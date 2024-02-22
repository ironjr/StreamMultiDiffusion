from transformers import CLIPTextModel, CLIPTokenizer, Blip2Processor, Blip2ForConditionalGeneration, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# Suppress partial model loading warning.
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange

from typing import Tuple, List, Literal, Optional, Union
from tqdm import tqdm
from PIL import Image

from util import gaussian_lowpass, blend


class MagicDraw(nn.Module):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        sd_version: Literal['1.5', 'xl'] = '1.5',
        hf_key: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.sd_version = sd_version

        print(f'[INFO] Loading Stable Diffusion...')
        if hf_key is not None:
            print(f'[INFO] Using Hugging Face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = 'stabilityai/stable-diffusion-2-1-base'
        elif self.sd_version == '2.0':
            model_key = 'stabilityai/stable-diffusion-2-base'
        elif self.sd_version == '1.5':
            model_key = 'runwayml/stable-diffusion-v1-5'
        else:
            raise ValueError(f'Stable Diffusion version {self.sd_version} not supported.')

        # Create model
        self.i2t_processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
        self.i2t_model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')

        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder='vae', torch_dtype=dtype).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder='tokenizer', torch_dtype=dtype)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder='text_encoder', torch_dtype=dtype).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder='unet', torch_dtype=dtype).to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder='scheduler')
        self.vae_scale_factor = 2 ** sum([
            1 for b in self.vae.encoder.down_blocks if hasattr(b, 'downsamplers') and b.downsamplers is not None])

        print(f'[INFO] Model is loaded!')

    @torch.no_grad()
    def get_text_embeds(self, prompt: str, negative_prompt: str) -> torch.Tensor:
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

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
        noise_lvs = noise_lvs[self.scheduler.timesteps]

        size = (images[0].size[1] // self.vae_scale_factor, images[0].size[0] // self.vae_scale_factor)

        masks = []
        masks_g = []
        for image in images:
            mask = (T.ToTensor()(image) < 0.5).float()[None, :1].to(noise_lvs.device)
            mask = gaussian_lowpass(mask, std)

            mask_stack = mask.repeat(len(noise_lvs), 1, 1, 1) > noise_lvs[:, None, None, None]
            mask_stack = F.interpolate(mask_stack.float(), size=size, mode='nearest')
            masks.append(mask_stack)
            masks_g.append(mask[0, 0])
        fg_masks = torch.stack(masks, dim=0)
        masks_g = torch.stack(masks_g, dim=0)

        bg_mask = 1 - torch.sum(fg_masks, dim=0, keepdim=True)
        bg_mask = bg_mask * (bg_mask > 0)

        masks = torch.cat([bg_mask, fg_masks], dim=0).to(self.dtype)
        return masks, masks_g

    @torch.no_grad()
    def __call__(
        self,
        background: Image.Image,
        masks: Union[Image.Image, List[Image.Image]],
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = '',
        background_negative_prompt: str = '',
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        mask_std: float = 10.0,
        mask_strength: float = 1.0,
    ) -> Image.Image:
        self.scheduler.set_timesteps(num_inference_steps)
        masks, masks_g = self.process_mask(masks, mask_std)  # (len(prompts) + 1, num_inference_steps + 1, 1, H, W)

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        bg_prompt = self.get_text_prompts(background)
        fg_prompt = [f'{p} with background of {bg_prompt}' for p in prompts]
        prompts = [bg_prompt] + fg_prompt
        negative_prompts = [background_negative_prompt] + negative_prompts

        # Calculate text embeddings.
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2 * len(prompts), 77, 768]
        latent = torch.randn(
            (1, self.unet.config.in_channels, height // 8, width // 8), dtype=self.dtype, device=self.device)
        bg_latent = self.encode_imgs(T.ToTensor()(background)[None].to(dtype=self.dtype, device=self.device))
        bg_noise = latent.clone()

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.scheduler.timesteps)):

                # Expand the latents if we are doing classifier-free guidance.
                latent_model_input = torch.cat([latent.repeat(len(prompts), 1, 1, 1)] * 2)

                # Perform one step of the reverse diffusion.
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeds)['sample']
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent_out = self.scheduler.step(noise_pred, t, latent)['prev_sample']

                # Mix the latents.
                if i < len(self.scheduler.timesteps) - 1:
                    s = self.scheduler.timesteps[i + 1]
                    bg = self.scheduler.add_noise(bg_latent, bg_noise, torch.as_tensor([s]))
                else:
                    bg = bg_latent

                mask = masks[:, i].to(latent.device)
                count = mask[1:].sum(dim=0, keepdims=True)
                latent = (mask * latent_out).sum(dim=0)
                latent = torch.where(count > 0, latent / count, latent)
                latent = (
                    (1 - mask[0]) * mask_strength * latent
                    + (1 - mask[0]) * (1.0 - mask_strength) * bg + mask[0] * bg
                )

        # Return PIL Image.
        fg_mask = torch.sum(masks_g, dim=0, keepdim=True).clip_(0, 1)
        image = T.ToPILImage()(self.decode_latents(latent.to(dtype=self.dtype))[0])
        image = blend(image, background, fg_mask, mask_std)
        return image

    @torch.no_grad()
    def sample(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = '',
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        self.scheduler.set_timesteps(num_inference_steps)

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Calculate text embeddings.
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]
        latent = torch.randn((1, self.unet.config.in_channels, height // 8, width // 8), device=self.device)

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
        img = T.ToPILImage()(self.decode_latents(latent.to(dtype=self.dtype))[0])
        return img