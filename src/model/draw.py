from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import DiffusionPipeline, LCMScheduler, DDIMScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange

from typing import Tuple, List, Literal, Optional, Union
from tqdm import tqdm
from PIL import Image

from util import gaussian_lowpass, blend, get_panorama_views


class StableMultiDiffusion(nn.Module):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        sd_version: Literal['1.5', '2.0', '2.1', 'xl'] = '1.5',
        hf_key: Optional[str] = None,
        lora_key: Optional[str] = None,
        load_from_local: bool = False, # Turn on if you have already downloaed LoRA & Hugging Face hub is down.
        default_mask_std: float = 8.0,
        default_mask_strength: float = 1.0,
        default_prompt_strength: float = 0.95,
        t_index_list: List[int] = [0, 5, 16, 18, 20, 37], # [0, 12, 25, 37], # Magic number.
        mask_type: Literal['discrete', 'semi-continuous', 'continuous'] = 'continuous',
    ) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.sd_version = sd_version

        self.default_mask_std = default_mask_std
        self.default_mask_strength = default_mask_strength
        self.default_prompt_strength = default_prompt_strength
        self.default_t_list = t_index_list
        self.mask_type = mask_type

        print(f'[INFO] Loading Stable Diffusion...')
        variant = None
        lora_weight_name = None
        if hf_key is not None:
            print(f'[INFO] Using Hugging Face custom model key: {hf_key}')
            model_key = hf_key
        # elif self.sd_version == 'xl':
        #     model_key = 'stabilityai/stable-diffusion-xl-base-1.0'
        #     lora_key = 'latent-consistency/lcm-lora-sdxl'
        #     variant = 'fp16'
        #     lora_weight_name = 'pytorch_lora_weights.safetensors'
        elif self.sd_version == '2.1':
            model_key = 'stabilityai/stable-diffusion-2-1-base'
            variant = 'fp16'
        elif self.sd_version == '2.0':
            model_key = 'stabilityai/stable-diffusion-2-base'
            variant = 'fp16'
        elif self.sd_version == '1.5':
            model_key = 'runwayml/stable-diffusion-v1-5'
            lora_key = 'latent-consistency/lcm-lora-sdv1-5'
            variant = 'fp16'
            lora_weight_name = 'pytorch_lora_weights.safetensors'
        else:
            raise ValueError(f'Stable Diffusion version {self.sd_version} not supported.')

        # Create model
        self.i2t_processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
        self.i2t_model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')

        self.pipe = DiffusionPipeline.from_pretrained(model_key, variant=variant, torch_dtype=dtype).to(self.device)
        if lora_key is None:
            print(f'[INFO] LCM LoRA is not available for SD version {sd_version}. Using DDIM Scheduler instead...')
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            self.scheduler = self.pipe.scheduler
            self.default_num_inference_steps = 50
            self.default_guidance_scale = 7.5
        else:
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            self.scheduler = self.pipe.scheduler
            self.pipe.load_lora_weights(lora_key, weight_name=lora_weight_name, adapter_name='lcm')
            self.default_num_inference_steps = 4
            self.default_guidance_scale = 1.0

            self.prepare_lcm_schedule(t_index_list, 50)

        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.vae_scale_factor = self.pipe.vae_scale_factor

        print(f'[INFO] Model is loaded!')

    def prepare_lcm_schedule(
        self,
        t_index_list: Optional[List[int]] = None,
        num_inference_steps: Optional[int] = None,
    ) -> None:
        if t_index_list is None:
            t_index_list = self.default_t_list
        if num_inference_steps is None:
            num_inference_steps = self.default_num_inference_steps

        self.scheduler.set_timesteps(num_inference_steps)
        self.timesteps = torch.as_tensor([
            self.scheduler.timesteps[t] for t in t_index_list
        ], dtype=torch.long)

        shape = (len(t_index_list), 1, 1, 1)

        c_skip_list = []
        c_out_list = []
        for timestep in self.timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)
        self.c_skip = torch.stack(c_skip_list).view(*shape).to(dtype=self.dtype, device=self.device)
        self.c_out = torch.stack(c_out_list).view(*shape).to(dtype=self.dtype, device=self.device)

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (torch.stack(alpha_prod_t_sqrt_list).view(*shape)
            .to(dtype=self.dtype, device=self.device))
        beta_prod_t_sqrt = (torch.stack(beta_prod_t_sqrt_list).view(*shape)
            .to(dtype=self.dtype, device=self.device))
        self.alpha_prod_t_sqrt = alpha_prod_t_sqrt
        self.beta_prod_t_sqrt = beta_prod_t_sqrt

        noise_lvs = (1 - self.scheduler.alphas_cumprod[self.timesteps].to(self.device)) ** 0.5
        self.noise_lvs = noise_lvs[None, :, None, None, None]
        self.next_noise_lvs = torch.cat([noise_lvs[1:], noise_lvs.new_zeros(1)])[None, :, None, None, None]

    @torch.no_grad()
    def get_text_embeds(self, prompt: str, negative_prompt: str) -> torch.Tensor:
        kwargs = dict(padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
    
        # Tokenize text and get embeddings.
        text_input = self.tokenizer(prompt, truncation=True, **kwargs)
        text_embeds = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.tokenizer(negative_prompt, **kwargs)
        uncond_embeds = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Final embedding is a concatenation.
        # text_embeds = torch.cat([uncond_embeds, text_embeds])
        return uncond_embeds, text_embeds

    @torch.no_grad()
    def get_text_prompts(self, image: Image.Image) -> str:
        question = 'Question: What are in the image? Answer:'
        inputs = self.i2t_processor(image, question, return_tensors='pt')
        out = self.i2t_model.generate(**inputs)
        prompt = self.i2t_processor.decode(out[0], skip_special_tokens=True).strip()
        return prompt

    @torch.no_grad()
    def encode_imgs(self, imgs: torch.Tensor, generator: Optional[torch.Generator] = None):
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
        return latents

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clip_(0, 1)
        return imgs

    @torch.no_grad()
    def process_mask(
        self,
        masks: Union[torch.Tensor, Image.Image, List[Image.Image]],
        strength: Optional[Union[torch.Tensor, float]] = None,
        std: Optional[Union[torch.Tensor, float]] = None,
        height: int = 512,
        width: int = 512,
        use_boolean_mask: bool = True,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        if isinstance(masks, Image.Image):
            masks = [masks]
        if isinstance(masks, (tuple, list)):
            # Assumes white background for Image.Image;
            # inverted boolean masks with shape (1, 1, H, W) for torch.Tensor.
            if use_boolean_mask:
                proc = lambda m: T.ToTensor()(m)[None, -1:] < 0.5
            else:
                proc = lambda m: 1.0 - T.ToTensor()(m)[None, -1:]
            masks = torch.cat([proc(mask) for mask in masks], dim=0).float().clip_(0, 1)
        masks = F.interpolate(masks, size=(height, width), mode='bilinear', align_corners=False)
        masks = masks.to(self.device)

        if timesteps is None:
            noise_lvs = self.noise_lvs
            next_noise_lvs = self.next_noise_lvs
        else:
            noise_lvs_ = (1 - self.scheduler.alphas_cumprod[timesteps].to(self.device)) ** 0.5
            noise_lvs = noise_lvs_[None, :, None, None, None]
            next_noise_lvs = torch.cat([noise_lvs_[1:], noise_lvs_.new_zeros(1)])[None, :, None, None, None]

        if std is None:
            std = self.default_mask_std
        if isinstance(std, (int, float)):
            std = [std] * len(masks)
        if isinstance(std, (list, tuple)):
            std = torch.as_tensor(std, dtype=torch.float, device=self.device)

        if strength is None:
            strength = self.default_mask_strength
        if isinstance(strength, (int, float)):
            strength = [strength] * len(masks)
        if isinstance(strength, (list, tuple)):
            strength = torch.as_tensor(strength, dtype=torch.float, device=self.device)

        if (std > 0).any():
            std = torch.where(std > 0, std, 1e-5)
            masks = gaussian_lowpass(masks, std)
        masks_blurred = masks

        # NOTE: This `strength` aligns with `denoising strength`. However, with LCM, using strength < 0.96
        #       gives unpleasant results.
        masks = masks * strength[:, None, None, None]
        masks = masks.unsqueeze(1).repeat(1, noise_lvs.shape[1], 1, 1, 1)

        if self.mask_type == 'discrete':
            # Discrete mode.
            masks = masks > self.noise_lvs
        elif self.mask_type == 'semi-continuous':
            # Semi-continuous mode (continuous at the last step only).
            masks = torch.cat((
                masks[:, :-1] > noise_lvs[:, :-1],
                (
                    (masks[:, -1:] - next_noise_lvs[:, -1:]) / (noise_lvs[:, -1:] - next_noise_lvs[:, -1:])
                ).clip_(0, 1),
            ), dim=1)
        elif self.mask_type == 'continuous':
            # Continuous mode: Have the exact same `1` coverage with discrete mode, but the mask gradually
            #                  decreases continuously after the discrete mode boundary to become `0` at the
            #                  next lower threshold.
            masks = ((masks - next_noise_lvs) / (noise_lvs - next_noise_lvs)).clip_(0, 1)

        # NOTE: Post processing mask strength does not align with conventional 'denoising_strength'. However,
        #       fine-grained mask alpha channel tuning is available with this form.
        # masks = masks * strength[None, :, None, None, None]

        h = height // self.vae_scale_factor
        w = width // self.vae_scale_factor
        masks = rearrange(masks.float(), 'p t () h w -> (p t) () h w')
        masks = F.interpolate(masks, size=(h, w), mode='nearest')
        masks = rearrange(masks.to(self.dtype), '(p t) () h w -> p t () h w', p=len(std))
        return masks, masks_blurred, std

    def scheduler_step(
        self,
        model_pred: torch.Tensor,
        x_t_latent: torch.Tensor,
        idx: int,
    ) -> torch.Tensor:
        F_theta = (x_t_latent - self.beta_prod_t_sqrt[idx] * model_pred) / self.alpha_prod_t_sqrt[idx]
        denoised = self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent
        return denoised

    @torch.no_grad()
    def sample(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = '',
        height: int = 512,
        width: int = 512,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        batch_size: int = 1,
    ) -> Image.Image:
        if num_inference_steps is None:
            num_inference_steps = self.default_num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.default_guidance_scale
        self.scheduler.set_timesteps(num_inference_steps)

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Calculate text embeddings.
        uncond_embeds, text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]
        text_embeds = torch.cat([uncond_embeds.mean(dim=0, keepdim=True), text_embeds.mean(dim=0, keepdim=True)])
        h = height // self.vae_scale_factor
        w = width // self.vae_scale_factor
        latent = torch.randn((batch_size, self.unet.config.in_channels, h, w), dtype=self.dtype, device=self.device)

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):
                # Expand the latents if we are doing classifier-free guidance.
                latent_model_input = torch.cat([latent] * 2)

                # Perform one step of the reverse diffusion.
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent = self.scheduler.step(noise_pred, t, latent)['prev_sample']

        # Return PIL Image.
        latent = latent.to(dtype=self.dtype)
        imgs = [T.ToPILImage()(self.decode_latents(l[None])[0]) for l in latents]
        return imgs

    @torch.no_grad()
    def sample_panorama(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = '',
        height: int = 512,
        width: int = 2048,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        window_size: int = 512,
    ) -> Image.Image:
        if num_inference_steps is None:
            num_inference_steps = self.default_num_inference_steps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.timesteps
            use_custom_timesteps = False
        else:
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            use_custom_timesteps = True
        if guidance_scale is None:
            guidance_scale = self.default_guidance_scale

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Calculate text embeddings.
        uncond_embeds, text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]
        text_embeds = torch.cat([uncond_embeds.mean(dim=0, keepdim=True), text_embeds.mean(dim=0, keepdim=True)])

        # Define panorama grid and get views
        h = height // self.vae_scale_factor
        w = width // self.vae_scale_factor
        latent = torch.randn((1, self.unet.config.in_channels, h, w), dtype=self.dtype, device=self.device)

        views, masks = get_panorama_views(height, width, window_size // self.vae_scale_factor)
        masks = masks.to(dtype=self.dtype, device=self.device)
        value = torch.zeros_like(latent)
        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(timesteps)):
                value.zero_()

                for j, (h_start, h_end, w_start, w_end) in enumerate(views):
                    # TODO we can support batches, and pass multiple views at once to the unet
                    latent_view = latent[:, :, h_start:h_end, w_start:w_end]

                    # Expand the latents if we are doing classifier-free guidance.
                    latent_model_input = torch.cat([latent_view] * 2)

                    # Perform one step of the reverse diffusion.
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # Compute the denoising step.
                    latents_view_denoised = self.scheduler_step(noise_pred, latent_view, i) # (1, 4, h, w)
                    mask = masks[..., j:j + 1, h_start:h_end, w_start:w_end] # (1, 1, h, w)
                    value[..., h_start:h_end, w_start:w_end] += mask * latents_view_denoised # (1, 1, h, w)

                # Update denoised latent.
                latent = value.clone()

                if i < len(timesteps) - 1:
                    j = i + 1
                    latent = (
                        self.alpha_prod_t_sqrt[j] * latent
                        + self.beta_prod_t_sqrt[j] * torch.randn_like(latent)
                    )

        # Return PIL Image.
        imgs = self.decode_latents(latent)
        img = T.ToPILImage()(imgs[0].cpu())
        return img

    @torch.no_grad()
    def __call__(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        negative_prompts: Union[str, List[str]] = '',
        suffix: Optional[str] = None, #', background is ',
        background: Optional[Union[torch.Tensor, Image.Image]] = None,
        background_prompt: Optional[str] = None,
        background_negative_prompt: str = '',
        height: int = 512,
        width: int = 512,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        prompt_strengths: Optional[Union[torch.Tensor, float, List[float]]] = None,
        masks: Optional[Union[Image.Image, List[Image.Image]]] = None,
        mask_strengths: Optional[Union[torch.Tensor, float, List[float]]] = None,
        mask_stds: Optional[Union[torch.Tensor, float, List[float]]] = None,
        use_boolean_mask: bool = True,
        do_blend: bool = True,
    ) -> Image.Image:

        ### Simplest cases

        # prompts is None: return background.
        # masks is None but prompts is not None: return prompts
        # masks is not None and prompts is not None: Do MagicDraw.

        if prompts is None or (isinstance(prompts, (list, tuple, str)) and len(prompts) == 0):
            if background is None and background_prompt is not None:
                return sample(background_prompt, background_negative_prompt, height, width, num_inference_steps, guidance_scale)
            return background
        elif masks is None or (isinstance(masks, (list, tuple)) and len(masks) == 0):
            return sample(prompts, negative_prompts, height, width, num_inference_steps, guidance_scale)


        ### Prepare generation

        h = height // self.vae_scale_factor
        w = width // self.vae_scale_factor
        latent = torch.randn((1, self.unet.config.in_channels, h, w), dtype=self.dtype, device=self.device)

        if num_inference_steps is None:
            num_inference_steps = self.default_num_inference_steps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.timesteps
            use_custom_timesteps = False
        else:
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            use_custom_timesteps = True
        if guidance_scale is None:
            guidance_scale = self.default_guidance_scale


        ### Prompts & Masks

        # asserts #m > 0 and #p > 0.
        # #m == #p == #n > 0: We happily generate according to the prompts & masks.
        # #m != #p: #p should be 1 and we will broadcast text embeds of p through m masks.
        # #p != #n: #n should be 1 and we will broadcast negative embeds n through p prompts.

        if isinstance(masks, Image.Image):
            masks = [masks]
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        num_masks = len(masks)
        num_prompts = len(prompts)
        num_nprompts = len(negative_prompts)
        assert num_prompts in (num_masks, 1), \
            f'The number of prompts {num_prompts} should match the number of masks {num_masks}!'
        assert num_nprompts in (num_prompts, 1), \
            f'The number of negative prompts {num_nprompts} should match the number of prompts {num_prompts}!'

        masks, masks_g, std = self.process_mask(
            masks,
            mask_strengths,
            mask_stds,
            use_boolean_mask=use_boolean_mask,
            timesteps=timesteps if use_custom_timesteps else None,
        )  # (p, t, 1, H, W)
        counts = masks.sum(dim=0)  # (T, 1, h, w)
        bg_masks = (1 - counts).clip_(0, 1)  # (T, 1, h, w)
        has_background = bg_masks.sum() > 0


        ### Background

        # background == None && background_prompt == None: Initialize with white background.
        # background == None && background_prompt != None: Generate background *along with other prompts*.
        # background != None && background_prompt == None: Retrieve text prompt using BLIP.
        # background != None && background_prompt != None: Use the given arguments.

        # not has_background: no effect of prompt_strength (the mix ratio between fg prompt & bg prompt)
        # has_background && prompt_strength != 1: mix only for this case.

        if has_background:
            if background is None and background_prompt is not None:
                masks = torch.cat((bg_masks[None], masks), dim=0)
                if suffix is not None:
                    prompts = [p + suffix + background_prompt for p in prompts]
                prompts = [background_prompt] + prompts
                negative_prompts = [background_negative_prompt] + negative_prompts
                has_background = False # Regard that background does not exist.
            else:
                if background is None and background_prompt is None:
                    background = torch.ones(1, 3, height, width, dtype=self.dtype, device=self.device)
                    background_prompt = 'simple white background image'
                elif background is not None and background_prompt is None:
                    background_prompt = self.get_text_prompts(background)
                if suffix is not None:
                    prompts = [p + suffix + background_prompt for p in prompts]
                prompts = [background_prompt] + prompts
                negative_prompts = [background_negative_prompt] + negative_prompts
                if isinstance(background, Image.Image):
                    background = T.ToTensor()(background).to(dtype=self.dtype, device=self.device)[None]
                bg_latent = self.encode_imgs(background)


        ### Prepare text embeddings (optimized for the minimal encoder batch size)

        uncond_embeds, text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2 * len(prompts), 77, 768]
        if has_background:
            # First channel is background prompt text embeds. Background prompt itself is not used for generation.
            s = prompt_strengths
            if prompt_strengths is None:
                s = self.default_prompt_strength
            if isinstance(s, (int, float)):
                s = [s] * num_prompts
            if isinstance(s, (list, tuple)):
                assert len(s) == num_prompts, \
                    f'The number of prompt strengths {len(s)} should match the number of prompts {num_prompts}!'
                s = torch.as_tensor(s, dtype=self.dtype, device=self.device)
            s = s[:, None, None]

            be = text_embeds[:1]
            bu = uncond_embeds[:1]
            fe = text_embeds[1:]
            fu = uncond_embeds[1:]
            if num_prompts > num_nprompts:
                # # negative prompts = 1; # prompts > 1.
                assert fu.shape[0] == 1 and fe.shape == num_prompts
                fu = fu.repeat(num_prompts, 1, 1)
            text_embeds = torch.lerp(be, fe, s)  # (p, 77, 768)
            uncond_embeds = torch.lerp(bu, fu, s)  # (n, 77, 768)
        elif num_prompts > num_nprompts:
            # # negative prompts = 1; # prompts > 1.
            assert uncond_embeds.shape[0] == 1 and text_embeds.shape[0] == num_prompts
            uncond_embeds = uncond_embeds.repeat(num_prompts, 1, 1)
        assert uncond_embeds.shape[0] == text_embeds.shape[0] == num_prompts
        if num_masks > num_prompts:
            assert masks.shape[0] == num_masks and num_prompts == 1
            text_embeds = text_embeds.repeat(num_masks, 1, 1)
            uncond_embeds = uncond_embeds.repeat(num_masks, 1, 1)
        text_embeds = torch.cat([uncond_embeds, text_embeds])


        ### Run

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(timesteps)):
                # Expand the latents if we are doing classifier-free guidance.
                latent_model_input = torch.cat([latent.repeat(num_masks, 1, 1, 1)] * 2)

                # Perform one step of the reverse diffusion.
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                if use_custom_timesteps:
                    latent_out = self.scheduler.step(noise_pred, t, latent)['prev_sample']

                    # Mix the latents.
                    mask = masks[:, i]
                    count = counts[i:i + 1]
                    bg_mask = bg_masks[i:i + 1]
                    latent = (mask * latent_out).sum(dim=0, keepdim=True)
                    latent = torch.where(count > 0, latent / count, latent)

                    # Mix foreground and background.
                    if has_background:
                        if i < len(timesteps) - 1:
                            s = timesteps[i + 1]
                            bg_noise = torch.randn_like(bg_latent)
                            bg = self.scheduler.add_noise(bg_latent, bg_noise, torch.as_tensor([s]))
                        else:
                            bg = bg_latent
                        latent = (1 - bg_mask) * latent + bg_mask * bg
                else:
                    latent_out = self.scheduler_step(noise_pred, latent, i)

                    # Mix the latents.
                    mask = masks[:, i]
                    count = counts[i:i + 1]
                    bg_mask = bg_masks[i:i + 1]
                    latent = (mask * latent_out).sum(dim=0, keepdim=True)
                    latent = torch.where(count > 0, latent / count, latent)

                    # Mix foreground and background.
                    if has_background:
                        latent = (1 - bg_mask) * latent + bg_mask * bg_latent

                    if i < len(timesteps) - 1:
                        j = i + 1
                        latent = (
                            self.alpha_prod_t_sqrt[j] * latent
                            + self.beta_prod_t_sqrt[j] * torch.randn_like(latent)
                        )

        # Return PIL Image.
        image = self.decode_latents(latent.to(dtype=self.dtype))[0]
        if has_background and do_blend:
            fg_mask = torch.sum(masks_g, dim=0).clip_(0, 1)
            image = blend(image, background[0], fg_mask, std)
        else:
            image = T.ToPILImage()(image)
        return image
