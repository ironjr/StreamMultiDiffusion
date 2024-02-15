from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import DiffusionPipeline, LCMScheduler, AutoencoderTiny

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange

from typing import Tuple, List, Literal, Optional, Union
from PIL import Image

from util import gaussian_lowpass, blend


class StreamMagicDraw(nn.Module):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        sd_version: Literal['1.5', 'xl'] = '1.5',
        hf_key: Optional[str] = None,
        lora_key: Optional[str] = None,
        use_tiny_vae: bool = True,
        t_index_list: List[int] = [0, 16, 32, 45], # Magic number.
        width: int = 512,
        height: int = 512,
        frame_buffer_size: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        cfg_type: Literal['none', 'full', 'self', 'initialize'] = 'self',
        seed: int = 2024,
    ) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == 'xl':
            model_key = 'stabilityai/stable-diffusion-xl-base-1.0'
            lora_key = 'latent-consistency/lcm-lora-sdxl'
        elif self.sd_version == '1.5':
            model_key = 'runwayml/stable-diffusion-v1-5'
            lora_key = 'latent-consistency/lcm-lora-sdv1-5'
        else:
            raise ValueError(f'Stable Diffusion version {self.sd_version} not supported.')

        # Create model
        self.i2t_processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
        self.i2t_model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')

        self.pipe = DiffusionPipeline.from_pretrained(model_key, variant='fp16', torch_dtype=dtype).to(self.device)
        self.pipe.load_lora_weights(lora_key, adapter_name='lcm')
        self.pipe.fuse_lora(
            fuse_unet=True,
            fuse_text_encoder=True,
            lora_scale=1.0,
            safe_fusing=False,
        )
        self.pipe.enable_xformers_memory_efficient_attention()

        self.vae = (
            AutoencoderTiny.from_pretrained('madebyollin/taesd').to(device=self.device, dtype=self.dtype)
            if use_tiny_vae else self.pipe.vae
        )
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.set_timesteps(num_inference_steps)

        # StreamDiffusion setting.
        self.generator = None

        self.height = height
        self.width = width
        self.latent_height = int(height // self.pipe.vae_scale_factor)
        self.latent_width = int(width // self.pipe.vae_scale_factor)

        self.vae_scale_factor = self.pipe.vae_scale_factor

        self.t_list = t_index_list
        assert len(self.t_list) > 1, 'Current version only supports diffusion models with multiple steps.'
        self.frame_bff_size = frame_buffer_size  # f
        self.denoising_steps_num = len(self.t_list)  # t=2
        self.cfg_type = cfg_type
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = 1.0 if self.cfg_type == 'none' else guidance_scale
        self.delta = delta

        self.batch_size = self.denoising_steps_num * frame_buffer_size  # T = t*f
        if self.cfg_type == 'initialize':
            self.trt_unet_batch_size = (self.denoising_steps_num + 1) * self.frame_bff_size
        elif self.cfg_type == 'full':
            self.trt_unet_batch_size = 2 * self.denoising_steps_num * self.frame_bff_size
        else:
            self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size

        print(f'[INFO] Model is loaded!')

    def prepare(self, generator: Optional[torch.Generator] = None, seed: Optional[int] = None) -> None:
        generator = torch.Generator(self.device) if generator is None else generator
        seed = self.seed if seed is None else seed
        self.generator = generator
        self.generator.manual_seed(seed)

        # initialize x_t_latent (it can be any random tensor)
        b = (self.denoising_steps_num - 1) * self.frame_bff_size
        self.x_t_latent_buffer = torch.zeros(
            (b, 4, self.latent_height, self.latent_width), dtype=self.dtype, device=self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.timesteps = self.scheduler.timesteps.to(self.device)
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])
        sub_timesteps_tensor = torch.tensor(self.sub_timesteps, dtype=torch.long, device=self.device)
        self.sub_timesteps_tensor = sub_timesteps_tensor.repeat_interleave(self.frame_bff_size, dim=0)

        self.init_noise = torch.randn((self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator, device=self.device, dtype=self.dtype)
        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)
        self.c_skip = torch.stack(c_skip_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)
        self.c_out = torch.stack(c_out_list).view(len(self.t_list), 1, 1, 1).to(dtype=self.dtype, device=self.device)

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (torch.stack(alpha_prod_t_sqrt_list).view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device))
        beta_prod_t_sqrt = (torch.stack(beta_prod_t_sqrt_list).view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device))
        self.alpha_prod_t_sqrt = alpha_prod_t_sqrt.repeat_interleave(self.frame_bff_size, dim=0)
        self.beta_prod_t_sqrt = beta_prod_t_sqrt.repeat_interleave(self.frame_bff_size, dim=0)

    @torch.no_grad()
    def get_text_prompts(self, image: Image.Image) -> str:
        question = 'Question: What are in the image? Answer:'
        inputs = self.i2t_processor(image, question, return_tensors='pt')
        out = self.i2t_model.generate(**inputs)
        prompt = self.i2t_processor.decode(out[0], skip_special_tokens=True).strip()
        return prompt

    @torch.no_grad()
    def encode_imgs(self, imgs: torch.Tensor, generator: Optional[torch.Generator] = None, add_noise: bool = False):
        def _retrieve_latents(
            encoder_output: torch.Tensor,
            generator: Optional[torch.Generator] = None,
            sample_mode: str = 'sample',
        ):
            if hasattr(encoder_output, 'latent_dist') and sample_mode == 'sample':
                return encoder_output.latent_dist.sample(generator)
            elif hasattr(encoder_output, 'latent_dist') and sample_mode == 'argmax':
                return encoder_output.latent_dist.mode()
            elif hasattr(encoder_output, 'latents'):
                return encoder_output.latents
            else:
                raise AttributeError('Could not access latents of provided encoder_output')

        imgs = 2 * imgs - 1
        latents = self.vae.config.scaling_factor * _retrieve_latents(self.vae.encode(imgs), generator=generator)
        if add_noise:
            latents = self.alpha_prod_t_sqrt[0] * latents + self.beta_prod_t_sqrt[0] * self.init_noise[0]
        return latents

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def process_mask(
        self,
        images: Union[Image.Image, List[Image.Image]],
        std: float = 10.0,
    ) -> torch.Tensor:
        if not isinstance(images, (tuple, list)):
            images = [images]

        noise_lvs = (1 - self.scheduler.alphas_cumprod) ** 0.5
        noise_lvs = noise_lvs.to(self.device)[self.sub_timesteps_tensor]

        size = (images[0].size[1] // self.vae_scale_factor, images[0].size[0] // self.vae_scale_factor)

        masks = []
        for image in images:
            mask = (T.ToTensor()(image) < 0.5)[None, :1].float().to(self.device)
            mask = gaussian_lowpass(mask, std)

            mask_stack = mask.repeat(len(noise_lvs), 1, 1, 1) > noise_lvs[:, None, None, None]
            mask_stack = F.interpolate(mask_stack.float(), size=size, mode='nearest').to(self.dtype)
            masks.append(mask_stack)
        fg_masks = torch.stack(masks, dim=0)

        bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
        bg_mask = bg_mask * (bg_mask > 0)

        self.counts = fg_masks.sum(dim=0)  # (T, 1, h, w)
        self.masks = torch.cat([bg_mask, fg_masks], dim=0)  # (p, T, 1, h, w)
        self.bg_mask = bg_mask[0]  # (T, 1, h, w)

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        # TODO: use t_list to select beta_prod_t_sqrt
        if idx is None:
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt_ * model_pred_batch) / self.alpha_prod_t_sqrt_
            denoised_batch = self.c_out_ * F_theta + self.c_skip_ * x_t_latent_batch
        else:
            F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
        return denoised_batch

    def unet_step(
        self,
        x_t_latent: torch.Tensor,  # (T, 4, h, w)
        idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = len(self.prompts)
        x_t_latent = x_t_latent.repeat_interleave(p, dim=0)  # (T * p, 4, h, w)
        t_list = self.sub_timesteps_tensor_  # (T * p,)
        if self.guidance_scale > 1.0 and self.cfg_type == 'initialize':
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:p], x_t_latent], dim=0)  # (T * p + 1, 4, h, w)
            t_list = torch.concat([t_list[0:p], t_list], dim=0)  # (T * p + 1, 4, h, w)
        elif self.guidance_scale > 1.0 and self.cfg_type == 'full':
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)  # (2 * T * p, 4, h, w)
            t_list = torch.concat([t_list, t_list], dim=0)  # (2 * T * p,)
        else:
            x_t_latent_plus_uc = x_t_latent  # (T * p, 4, h, w)

        model_pred = self.unet(
            x_t_latent_plus_uc,  # (B, 4, h, w)
            t_list,  # (B,)
            encoder_hidden_states=self.prompt_embeds,  # (B, 77, 768)
            return_dict=False,
        )[0]  # (B, 4, h, w)

        # noise_pred_text, noise_pred_uncond: (T * p, 4, h, w)
        # self.stock_noise, init_noise: (T, 4, h, w)
        if self.guidance_scale > 1.0 and self.cfg_type == 'initialize':
            noise_pred_text = model_pred[p:]
            self.stock_noise_ = torch.concat([model_pred[0:p], self.stock_noise_[p:]], dim=0)
        elif self.guidance_scale > 1.0 and self.cfg_type == 'full':
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (self.cfg_type == 'self' or self.cfg_type == 'initialize'):
            noise_pred_uncond = self.stock_noise_ * self.delta

        if self.guidance_scale > 1.0 and self.cfg_type != 'none':
            model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
        if self.cfg_type == 'self' or self.cfg_type == 'initialize':
            scaled_noise = self.beta_prod_t_sqrt_ * self.stock_noise_
            delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)

            # Do mask edit.

            alpha_next = torch.concat([self.alpha_prod_t_sqrt_[p:], torch.ones_like(self.alpha_prod_t_sqrt_[0:p])], dim=0)
            delta_x = alpha_next * delta_x
            beta_next = torch.concat([self.beta_prod_t_sqrt_[p:], torch.ones_like(self.beta_prod_t_sqrt_[0:p])], dim=0)
            delta_x = delta_x / beta_next
            init_noise = torch.concat([self.init_noise_[p:], self.init_noise_[0:p]], dim=0)
            self.stock_noise_ = init_noise + delta_x

        p2 = len(self.t_list) - 1
        background = torch.concat([
            self.scheduler.add_noise(
                self.background_latent.repeat(p2, 1, 1, 1),
                self.stock_noise[1:],
                torch.tensor(self.t_list[1:], device=self.device),
            ),
            self.background_latent,
        ], dim=0)

        denoised_batch = rearrange(denoised_batch, '(t p) c h w -> p t c h w', p=p)
        latent = (self.masks * denoised_batch).sum(dim=0)  # (T, 4, h, w)
        latent = torch.where(self.counts > 0, latent / self.counts, latent)
        latent = (
            (1 - self.bg_mask) * self.mask_strength * latent
            + ((1 - self.bg_mask) * (1.0 - self.mask_strength) + self.bg_mask) * background
        )
        return latent

    @torch.no_grad()
    def register_background(
        self,
        background: Image.Image,
        background_prompt: Optional[str] = None,
        background_negative_prompt: str = '',
    ) -> None:
        background = background.resize((self.width, self.height))
        self.background_image = background
        self.background_latent = self.encode_imgs(T.ToTensor()(background)[None].to(self.device, self.dtype)) # (1, 3, H, W)
        self.background_prompt = self.get_text_prompts(background) if background_prompt is None else background_prompt
        self.background_negative_prompt = background_negative_prompt

    @torch.no_grad()
    def register_prompts(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = '',
        suffix: Optional[str] = ', background is ',
    ) -> None:
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        fg_prompt = [p + suffix + self.background_prompt if suffix is not None else p for p in prompts]
        self.prompts = [self.background_prompt] + fg_prompt
        self.negative_prompts = [self.background_negative_prompt] + negative_prompts

        encoder_output = self.pipe.encode_prompt(
            prompt=self.prompts,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=(self.guidance_scale > 1.0),
            negative_prompt=self.negative_prompts,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)  # (T * p, 77, 768)

        if self.cfg_type == 'full':
            uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)  # (T * p, 77, 768)
        elif self.cfg_type == 'initialize':
            uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)  # (f * p, 77, 768)
        if self.guidance_scale > 1.0 and (self.cfg_type == 'initialize' or self.cfg_type == 'full'):
            self.prompt_embeds = torch.cat([uncond_prompt_embeds, self.prompt_embeds], dim=0)  # (2 * T * p, 77, 768)

        p = len(self.prompts)
        self.sub_timesteps_tensor_ = self.sub_timesteps_tensor.repeat_interleave(p)  # (T * p,)
        # self.init_noise_ = self.init_noise.repeat_interleave(p, dim=0)  # (T * p, 77, 768)
        # self.stock_noise_ = self.stock_noise.repeat_interleave(p, dim=0)  # (T * p, 77, 768)
        self.c_out_ = self.c_out.repeat_interleave(p, dim=0)  # (T * p, 1, 1, 1)
        self.c_skip_ = self.c_skip.repeat_interleave(p, dim=0)  # (T * p, 1, 1, 1)
        self.beta_prod_t_sqrt_ = self.beta_prod_t_sqrt.repeat_interleave(p, dim=0)  # (T * p, 1, 1, 1)
        self.alpha_prod_t_sqrt_ = self.alpha_prod_t_sqrt.repeat_interleave(p, dim=0)  # (T * p, 1, 1, 1)

    @torch.no_grad()
    def register_masks(
        self,
        masks: Image.Image,
        mask_std: float = 10.0,
        mask_strength: float = 1.0,
    ) -> None:
        self.mask_images = masks
        self.process_mask(masks, mask_std)  # [p + 1, t + 1, 1, H, W]
        self.mask_std = mask_std
        self.mask_strength = mask_strength

    @torch.no_grad()
    def register_all(
        self,
        prompts: Union[str, List[str]],
        masks: Image.Image,
        background: Image.Image,
        background_prompt: Optional[str] = None,
        background_negative_prompt: str = '',
        negative_prompts: Union[str, List[str]] = '',
        suffix: Optional[str] = ', background is ',
        mask_std: float = 10.0,
        mask_strength: float = 1.0,
    ) -> None:
        # The order of this registration should not be changed!
        self.register_background(background, background_prompt, background_negative_prompt)
        self.register_prompts(prompts, negative_prompts)
        self.register_masks(masks, mask_std, mask_strength)

    @torch.no_grad()
    def __call__(self) -> Image.Image:
        latent = torch.randn((1, self.unet.in_channels, self.latent_height, self.latent_width),
            dtype=self.dtype, device=self.device)  # (1, 4, h, w)
        latent = torch.cat((latent, self.x_t_latent_buffer), dim=0)  # (t, 4, h, w)
        self.stock_noise = torch.cat((self.init_noise[0:1], self.stock_noise[:-1]), dim=0)  # (t, 4, h, w)

        x_0_pred_batch = self.unet_step(latent)

        latent = x_0_pred_batch[-1:]
        self.x_t_latent_buffer = (
            self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
            + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
        )

        imgs = self.decode_latents(latent.half())  # (1, 3, H, W)
        img = T.ToPILImage()(imgs[0].cpu())
        return img