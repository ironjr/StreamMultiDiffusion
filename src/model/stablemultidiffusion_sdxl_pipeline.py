# Copyright (c) 2024 Jaerin Lee

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from diffusers import (
    AutoencoderTiny,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.loaders import (
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange

from typing import Tuple, List, Literal, Optional, Union
from tqdm import tqdm
from PIL import Image

from util import load_model, gaussian_lowpass, blend, get_panorama_views, shift_to_mask_bbox_center


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class StableMultiDiffusionSDXLPipeline(nn.Module):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        hf_key: Optional[str] = None,
        lora_key: Optional[str] = None,
        load_from_local: bool = False, # Turn on if you have already downloaed LoRA & Hugging Face hub is down.
        default_mask_std: float = 1.0, # 8.0
        default_mask_strength: float = 1.0,
        default_prompt_strength: float = 1.0, # 8.0
        default_bootstrap_steps: int = 1,
        default_boostrap_mix_steps: float = 1.0,
        default_bootstrap_leak_sensitivity: float = 0.2,
        default_preprocess_mask_cover_alpha: float = 0.3,
        t_index_list: List[int] = [0, 4, 12, 25, 37], # [0, 5, 16, 18, 20, 37], # # [0, 12, 25, 37], # Magic number.
        mask_type: Literal['discrete', 'semi-continuous', 'continuous'] = 'discrete',
        has_i2t: bool = True,
        lora_weight: float = 1.0,
    ) -> None:
        r"""Stabilized MultiDiffusion for fast sampling.

        Accelrated region-based text-to-image synthesis with Latent Consistency
        Model while preserving mask fidelity and quality.

        Args:
            device (torch.device): Specify CUDA device.
            hf_key (Optional[str]): Custom StableDiffusion checkpoint for
                stylized generation.
            lora_key (Optional[str]): Custom Lightning LoRA for acceleration.
            load_from_local (bool): Turn on if you have already downloaed LoRA 
                & Hugging Face hub is down.
            default_mask_std (float): Preprocess mask with Gaussian blur with
                specified standard deviation.
            default_mask_strength (float): Preprocess mask by multiplying it
                globally with the specified variable. Caution: extremely
                sensitive. Recommended range: 0.98-1.
            default_prompt_strength (float): Preprocess foreground prompts
                globally by linearly interpolating its embedding with the
                background prompt embeddint with specified mix ratio. Useful
                control handle for foreground blending. Recommended range:
                0.5-1.
            default_bootstrap_steps (int): Bootstrapping stage steps to
                encourage region separation. Recommended range: 1-3.
            default_boostrap_mix_steps (float): Bootstrapping background is a
                linear interpolation between background latent and the white
                image latent. This handle controls the mix ratio. Available
                range: 0-(number of bootstrapping inference steps). For
                example, 2.3 means that for the first two steps, white image
                is used as a bootstrapping background and in the third step,
                mixture of white (0.3) and registered background (0.7) is used
                as a bootstrapping background.
            default_bootstrap_leak_sensitivity (float): Postprocessing at each
                inference step by masking away the remaining bootstrap
                backgrounds t Recommended range: 0-1.
            default_preprocess_mask_cover_alpha (float): Optional preprocessing
                where each mask covered by other masks is reduced in its alpha
                value by this specified factor.
            t_index_list (List[int]): The default scheduling for LCM scheduler.
            mask_type (Literal['discrete', 'semi-continuous', 'continuous']):
                defines the mask quantization modes. Details in the codes of
                `self.process_mask`. Basically, this (subtly) controls the
                smoothness of foreground-background blending. More continuous
                means more blending, but smaller generated patch depending on
                the mask standard deviation.
            has_i2t (bool): Automatic background image to text prompt con-
                version with BLIP-2 model. May not be necessary for the non-
                streaming application.
            lora_weight (float): Adjusts weight of the LCM/Lightning LoRA.
                Heavily affects the overall quality!
        """
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.default_mask_std = default_mask_std
        self.default_mask_strength = default_mask_strength
        self.default_prompt_strength = default_prompt_strength
        self.default_t_list = t_index_list
        self.default_bootstrap_steps = default_bootstrap_steps
        self.default_boostrap_mix_steps = default_boostrap_mix_steps
        self.default_bootstrap_leak_sensitivity = default_bootstrap_leak_sensitivity
        self.default_preprocess_mask_cover_alpha = default_preprocess_mask_cover_alpha
        self.mask_type = mask_type

        # Create model.
        print(f'[INFO] Loading Stable Diffusion...')
        variant = None
        model_ckpt = None
        lora_ckpt = None
        lightning_repo = 'ByteDance/SDXL-Lightning'
        if hf_key is not None:
            print(f'[INFO] Using Hugging Face custom model key: {hf_key}')
            model_key = hf_key
            lora_ckpt = 'sdxl_lightning_4step_lora.safetensors'

            self.pipe = load_model(model_key, 'xl', self.device, self.dtype)
            self.pipe.load_lora_weights(hf_hub_download(lightning_repo, lora_ckpt), adapter_name='lightning')
            self.pipe.set_adapters(["lightning"], adapter_weights=[lora_weight])
            self.pipe.fuse_lora()
        else:
            model_key = 'stabilityai/stable-diffusion-xl-base-1.0'
            variant = 'fp16'
            model_ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

            unet = UNet2DConditionModel.from_config(model_key, subfolder='unet').to(self.device, self.dtype)
            unet.load_state_dict(load_file(hf_hub_download(lightning_repo, model_ckpt), device=self.device))
            self.pipe = StableDiffusionXLPipeline.from_pretrained(model_key, unet=unet, torch_dtype=self.dtype, variant=variant).to(self.device)

        # Create model
        if has_i2t:
            self.i2t_processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
            self.i2t_model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')

        # Use SDXL-Lightning LoRA by default.
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing")
        self.scheduler = self.pipe.scheduler
        self.default_num_inference_steps = 4
        self.default_guidance_scale = 0.0

        if t_index_list is None:
            self.prepare_lightning_schedule(
                list(range(self.default_num_inference_steps)),
                self.default_num_inference_steps,
            )
        else:
            self.prepare_lightning_schedule(t_index_list, 50)

        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer_2 = self.pipe.tokenizer_2
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2
        self.unet = self.pipe.unet
        self.vae_scale_factor = self.pipe.vae_scale_factor

        # Prepare white background for bootstrapping.
        self.get_white_background(1024, 1024)

        print(f'[INFO] Model is loaded!')

    def prepare_lightning_schedule(
        self,
        t_index_list: Optional[List[int]] = None,
        num_inference_steps: Optional[int] = None,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
    ) -> None:
        r"""Set up different inference schedule for the diffusion model.

        You do not have to run this explicitly if you want to use the default
        setting, but if you want other time schedules, run this function
        between the module initialization and the main call.

        Note:
          - Recommended t_index_lists for LCMs:
              - [0, 12, 25, 37]: Default schedule for 4 steps. Best for
                  panorama. Not recommended if you want to use bootstrapping.
                  Because bootstrapping stage affects the initial structuring
                  of the generated image & in this four step LCM, this is done
                  with only at the first step, the structure may be distorted.
              - [0, 4, 12, 25, 37]: Recommended if you would use 1-step boot-
                  strapping. Default initialization in this implementation.
              - [0, 5, 16, 18, 20, 37]: Recommended if you would use 2-step
                  bootstrapping.
          - Due to the characteristic of SD1.5 LCM LoRA, setting
            `num_inference_steps` larger than 20 may results in overly blurry
            and unrealistic images. Beware!

        Args:
            t_index_list (Optional[List[int]]): The specified scheduling step
                regarding the maximum timestep as `num_inference_steps`, which
                is by default, 50. That means that
                `t_index_list=[0, 12, 25, 37]` is a relative time indices basd
                on the full scale of 50. If None, reinitialize the module with
                the default value.
            num_inference_steps (Optional[int]): The maximum timestep of the
                sampler. Defines relative scale of the `t_index_list`. Rarely
                used in practice. If None, reinitialize the module with the
                default value.
        """
        if t_index_list is None:
            t_index_list = self.default_t_list
        if num_inference_steps is None:
            num_inference_steps = self.default_num_inference_steps

        self.scheduler.set_timesteps(num_inference_steps)
        self.timesteps = self.scheduler.timesteps[torch.tensor(t_index_list)]

        # EulerDiscreteScheduler

        self.sigmas = self.scheduler.sigmas[torch.tensor(t_index_list)]
        self.sigmas_next = torch.cat([self.sigmas, self.sigmas.new_zeros(1)])[1:]
        sigma_mask = torch.logical_and(s_tmin <= self.sigmas, self.sigmas <= s_tmax)
        # self.gammas = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) * sigma_mask
        self.gammas = min(s_churn / (num_inference_steps - 1), 2**0.5 - 1) * sigma_mask
        self.sigma_hats = self.sigmas * (self.gammas + 1)
        self.dt = self.sigmas_next - self.sigma_hats

        noise_lvs = self.sigmas * (self.sigmas**2 + 1)**(-0.5)
        self.noise_lvs = noise_lvs[None, :, None, None, None]
        self.next_noise_lvs = torch.cat([noise_lvs[1:], noise_lvs.new_zeros(1)])[None, :, None, None, None]

    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
                FusedAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    @torch.no_grad()
    def get_text_prompts(self, image: Image.Image) -> str:
        r"""A convenient method to extract text prompt from an image.

        This is called if the user does not provide background prompt but only
        the background image. We use BLIP-2 to automatically generate prompts.

        Args:
            image (Image.Image): A PIL image.

        Returns:
            A single string of text prompt.
        """
        if hasattr(self, 'i2t_model'):
            question = 'Question: What are in the image? Answer:'
            inputs = self.i2t_processor(image, question, return_tensors='pt')
            out = self.i2t_model.generate(**inputs, max_new_tokens=77)
            prompt = self.i2t_processor.decode(out[0], skip_special_tokens=True).strip()
            return prompt
        else:
            return ''

    @torch.no_grad()
    def encode_imgs(
        self,
        imgs: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        vae: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        r"""A wrapper function for VAE encoder of the latent diffusion model.

        Args:
            imgs (torch.Tensor): An image to get StableDiffusion latents.
                Expected shape: (B, 3, H, W). Expected pixel scale: [0, 1].
            generator (Optional[torch.Generator]): Seed for KL-Autoencoder.
            vae (Optional[nn.Module]): Explicitly specify VAE (used for
                the demo application with TinyVAE).

        Returns:
            An image latent embedding with 1/8 size (depending on the auto-
            encoder. Shape: (B, 4, H//8, W//8).
        """
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

        vae = self.vae if vae is None else vae
        imgs = 2 * imgs - 1
        latents = vae.config.scaling_factor * _retrieve_latents(vae.encode(imgs), generator=generator)
        return latents

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, vae: Optional[nn.Module] = None) -> torch.Tensor:
        r"""A wrapper function for VAE decoder of the latent diffusion model.

        Args:
            latents (torch.Tensor): An image latent to get associated images.
                Expected shape: (B, 4, H//8, W//8).
            vae (Optional[nn.Module]): Explicitly specify VAE (used for
                the demo application with TinyVAE).

        Returns:
            An image latent embedding with 1/8 size (depending on the auto-
            encoder. Shape: (B, 3, H, W).
        """
        vae = self.vae if vae is None else vae
        latents = 1 / vae.config.scaling_factor * latents
        imgs = vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clip_(0, 1)
        return imgs

    @torch.no_grad()
    def get_white_background(self, height: int, width: int) -> torch.Tensor:
        r"""White background image latent for bootstrapping or in case of
        absent background.

        Additionally stores the maximally-sized white latent for fast retrieval
        in the future. By default, we initially call this with 1024x1024 sized
        white image, so the function is rarely visited twice.

        Args:
            height (int): The height of the white *image*, not its latent.
            width (int): The width of the white *image*, not its latent.

        Returns:
            A white image latent of size (1, 4, height//8, width//8). A cropped
            version of the stored white latent is returned if the requested
            size is smaller than what we already have created.
        """
        if not hasattr(self, 'white') or self.white.shape[-2] < height or self.white.shape[-1] < width:
            white = torch.ones(1, 3, height, width, dtype=self.dtype, device=self.device)
            self.white = self.encode_imgs(white)
            return self.white
        return self.white[..., :(height // self.vae_scale_factor), :(width // self.vae_scale_factor)]

    @torch.no_grad()
    def process_mask(
        self,
        masks: Union[torch.Tensor, Image.Image, List[Image.Image]],
        strength: Optional[Union[torch.Tensor, float]] = None,
        std: Optional[Union[torch.Tensor, float]] = None,
        height: int = 1024,
        width: int = 1024,
        use_boolean_mask: bool = True,
        timesteps: Optional[torch.Tensor] = None,
        preprocess_mask_cover_alpha: Optional[float] = None,
    ) -> Tuple[torch.Tensor]:
        r"""Fast preprocess of masks for region-based generation with fine-
        grained controls.

        Mask preprocessing is done in four steps:
         1. Resizing: Resize the masks into the specified width and height by
            nearest neighbor interpolation.
         2. (Optional) Ordering: Masks with higher indices are considered to
            cover the masks with smaller indices. Covered masks are decayed
            in its alpha value by the specified factor of
            `preprocess_mask_cover_alpha`.
         3. Blurring: Gaussian blur is applied to the mask with the specified
            standard deviation (isotropic). This results in gradual increase of
            masked region as the timesteps evolve, naturally blending fore-
            ground and the predesignated background. Not strictly required if
            you want to produce images from scratch withoout background.
         4. Quantization: Split the real-numbered masks of value between [0, 1]
            into predefined noise levels for each quantized scheduling step of
            the diffusion sampler. For example, if the diffusion model sampler
            has noise level of [0.9977, 0.9912, 0.9735, 0.8499, 0.5840], which
            is the default noise level of this module with schedule [0, 4, 12,
            25, 37], the masks are split into binary masks whose values are
            greater than these levels. This results in tradual increase of mask
            region as the timesteps increase. Details are described in our
            paper at https://arxiv.org/pdf/2403.09055.pdf.

        On the Three Modes of `mask_type`:
            `self.mask_type` is predefined at the initialization stage of this
            pipeline. Three possible modes are available: 'discrete', 'semi-
            continuous', and 'continuous'. These define the mask quantization
            modes we use. Basically, this (subtly) controls the smoothness of
            foreground-background blending. Continuous modes produces nonbinary
            masks to further blend foreground and background latents by linear-
            ly interpolating between them. Semi-continuous masks only applies
            continuous mask at the last step of the LCM sampler. Due to the
            large step size of the LCM scheduler, we find that our continuous
            blending helps generating seamless inpainting and editing results.

        Args:
            masks (Union[torch.Tensor, Image.Image, List[Image.Image]]): Masks.
            strength (Optional[Union[torch.Tensor, float]]): Mask strength that
                overrides the default value. A globally multiplied factor to
                the mask at the initial stage of processing. Can be applied
                seperately for each mask.
            std (Optional[Union[torch.Tensor, float]]): Mask blurring Gaussian
                kernel's standard deviation. Overrides the default value. Can
                be applied seperately for each mask.
            height (int): The height of the expected generation. Mask is
                resized to (height//8, width//8) with nearest neighbor inter-
                polation.
            width (int): The width of the expected generation. Mask is resized
                to (height//8, width//8) with nearest neighbor interpolation.
            use_boolean_mask (bool): Specify this to treat the mask image as
                a boolean tensor. The retion with dark part darker than 0.5 of
                the maximal pixel value (that is, 127.5) is considered as the
                designated mask.
            timesteps (Optional[torch.Tensor]): Defines the scheduler noise
                levels that acts as bins of mask quantization.
            preprocess_mask_cover_alpha (Optional[float]): Optional pre-
                processing where each mask covered by other masks is reduced in
                its alpha value by this specified factor. Overrides the default
                value.

        Returns: A tuple of tensors.
          - masks: Preprocessed (ordered, blurred, and quantized) binary/non-
                binary masks (see the explanation on `mask_type` above) for
                region-based image synthesis.
          - masks_blurred: Gaussian blurred masks. Used for optionally
                specified foreground-background blending after image
                generation.
          - std: Mask blur standard deviation. Used for optionally specified
                foreground-background blending after image generation.
        """
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
        masks = F.interpolate(masks.float(), size=(height, width), mode='bilinear', align_corners=False)
        masks = masks.to(self.device)

        # Background mask alpha is decayed by the specified factor where foreground masks covers it.
        if preprocess_mask_cover_alpha is None:
            preprocess_mask_cover_alpha = self.default_preprocess_mask_cover_alpha
        if preprocess_mask_cover_alpha > 0:
            masks = torch.stack([
                torch.where(
                    masks[i + 1:].sum(dim=0) > 0,
                    mask * preprocess_mask_cover_alpha,
                    mask,
                ) if i < len(masks) - 1 else mask
                for i, mask in enumerate(masks)
            ], dim=0)

        # Scheduler noise levels for mask quantization.
        if timesteps is None:
            noise_lvs = self.noise_lvs
            next_noise_lvs = self.next_noise_lvs
        else:
            noise_lvs_ = self.sigmas * (self.sigmas**2 + 1)**(-0.5)
            # noise_lvs_ = (1 - self.scheduler.alphas_cumprod[timesteps].to(self.device)) ** 0.5
            noise_lvs = noise_lvs_[None, :, None, None, None].to(masks.device)
            next_noise_lvs = torch.cat([noise_lvs_[1:], noise_lvs_.new_zeros(1)])[None, :, None, None, None]

        # Mask preprocessing parameters are fetched from the default settings.
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

        # Mask is quantized according to the current noise levels specified by the scheduler.
        if self.mask_type == 'discrete':
            # Discrete mode.
            masks = masks > noise_lvs
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

    def scheduler_scale_model_input(
        self,
        latent: torch.FloatTensor,
        idx: int,
    ) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        latent = latent / ((self.sigmas[idx]**2 + 1) ** 0.5)
        return latent

    def scheduler_step(
        self,
        noise_pred: torch.Tensor,
        idx: int,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        r"""Denoise-only step for reverse diffusion scheduler.
        
        Designed to match the interface of the original `pipe.scheduler.step`,
        which is a combination of this method and the following
        `scheduler_add_noise`.

        Args:
            noise_pred (torch.Tensor): Noise prediction results from the U-Net.
            idx (int): Instead of timesteps (in [0, 1000]-scale) use indices
                for the timesteps tensor (ranged in [0, len(timesteps)-1]).
            latent (torch.Tensor): Noisy latent.

        Returns:
            A denoised tensor with the same size as latent.
        """
        # Upcast to avoid precision issues when computing prev_sample.
        latent = latent.to(torch.float32)

        # 1. Compute predicted original sample (x_0) from sigma-scaled predicted noise.
        assert self.scheduler.config.prediction_type == 'epsilon', 'Only supports `prediction_type` of `epsilon` for now.'
        # pred_original_sample = latent - self.sigma_hats[idx] * noise_pred
        # prev_sample = pred_original_sample + noise_pred * (self.dt[i] + self.sigma_hats[i])
        # return pred_original_sample.to(self.dtype)

        # 2. Convert to an ODE derivative.
        prev_sample = latent + noise_pred * self.dt[idx]
        return prev_sample.to(self.dtype)

    def scheduler_add_noise(
        self,
        latent: torch.Tensor,
        noise: Optional[torch.Tensor],
        idx: int,
        s_noise: float = 1.0,
        initial: bool = False,
    ) -> torch.Tensor:
        r"""Separated noise-add step for the reverse diffusion scheduler.
        
        Designed to match the interface of the original
        `pipe.scheduler.add_noise`.

        Args:
            latent (torch.Tensor): Denoised latent.
            noise (torch.Tensor): Added noise. Can be None. If None, a random
                noise is newly sampled for addition.
            idx (int): Instead of timesteps (in [0, 1000]-scale) use indices
                for the timesteps tensor (ranged in [0, len(timesteps)-1]).

        Returns:
            A noisy tensor with the same size as latent.
        """
        if initial:
            if idx < len(self.sigmas) and idx >= 0:
                noise = torch.randn_like(latent) if noise is None else noise
                return latent + self.sigmas[idx] * noise
            else:
                return latent
        else:
            # 3. Post-add noise.
            noise_lv = (self.sigma_hats[idx]**2 - self.sigmas[idx]**2) ** 0.5
            if self.gammas[idx] > 0 and noise_lv > 0 and s_noise > 0 and idx < len(self.sigmas) and idx >= 0:
                noise = torch.randn_like(latent) if noise is None else noise
                eps = noise * s_noise * noise_lv
                latent = latent + eps
                # pred_original_sample = pred_original_sample + eps
            return latent

    @torch.no_grad()
    def __call__(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        negative_prompts: Union[str, List[str]] = '',
        suffix: Optional[str] = None, #', background is ',
        background: Optional[Union[torch.Tensor, Image.Image]] = None,
        background_prompt: Optional[str] = None,
        background_negative_prompt: str = '',
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        prompt_strengths: Optional[Union[torch.Tensor, float, List[float]]] = None,
        masks: Optional[Union[Image.Image, List[Image.Image]]] = None,
        mask_strengths: Optional[Union[torch.Tensor, float, List[float]]] = None,
        mask_stds: Optional[Union[torch.Tensor, float, List[float]]] = None,
        use_boolean_mask: bool = True,
        do_blend: bool = True,
        tile_size: int = 1024,
        bootstrap_steps: Optional[int] = None,
        boostrap_mix_steps: Optional[float] = None,
        bootstrap_leak_sensitivity: Optional[float] = None,
        preprocess_mask_cover_alpha: Optional[float] = None,
    ) -> Image.Image:
        r"""Arbitrary-size image generation from multiple pairs of (regional)
        text prompt-mask pairs.

        This is a main routine for this pipeline.

        Example:
            >>> device = torch.device('cuda:0')
            >>> smd = StableMultiDiffusionPipeline(device)
            >>> prompts = {... specify prompts}
            >>> masks = {... specify mask tensors}
            >>> height, width = masks.shape[-2:]
            >>> image = smd(
            >>>     prompts, masks=masks.float(), height=height, width=width)
            >>> image.save('my_beautiful_creation.png')

        Args:
            prompts (Union[str, List[str]]): A text prompt.
            negative_prompts (Union[str, List[str]]): A negative text prompt.
            suffix (Optional[str]): One option for blending foreground prompts
                with background prompts by simply appending background prompt
                to the end of each foreground prompt with this `middle word` in
                between. For example, if you set this as `, background is`,
                then the foreground prompt will be changed into
                `(fg), background is (bg)` before conditional generation.
            background (Optional[Union[torch.Tensor, Image.Image]]): a
                background image, if the user wants to draw in front of the
                specified image. Background prompt will automatically generated
                with a BLIP-2 model.
            background_prompt (Optional[str]): The background prompt is used
                for preprocessing foreground prompt embeddings to blend
                foreground and background.
            background_negative_prompt (Optional[str]): The negative background
                prompt.
            height (int): Height of a generated image. It is tiled if larger
                than `tile_size`.
            width (int): Width of a generated image. It is tiled if larger
                than `tile_size`.
            num_inference_steps (Optional[int]): Number of inference steps.
                Default inference scheduling is used if none is specified.
            guidance_scale (Optional[float]): Classifier guidance scale.
                Default value is used if none is specified.
            prompt_strength (float): Overrides default value. Preprocess
                foreground prompts globally by linearly interpolating its
                embedding with the background prompt embeddint with specified
                mix ratio. Useful control handle for foreground blending.
                Recommended range: 0.5-1.
            masks (Optional[Union[Image.Image, List[Image.Image]]]): a list of
                mask images. Each mask associates with each of the text prompts
                and each of the negative prompts. If specified as an image, it
                regards the image as a boolean mask. Also accepts torch.Tensor
                masks, which can have nonbinary values for fine-grained
                controls in mixing regional generations.
            mask_strengths (Optional[Union[torch.Tensor, float, List[float]]]):
                Overrides the default value. an be assigned for each mask
                separately. Preprocess mask by multiplying it globally with the
                specified variable. Caution: extremely sensitive. Recommended
                range: 0.98-1.
            mask_stds (Optional[Union[torch.Tensor, float, List[float]]]):
                Overrides the default value. Can be assigned for each mask
                separately. Preprocess mask with Gaussian blur with specified
                standard deviation. Recommended range: 0-64.
            use_boolean_mask (bool): Turn this off if you want to treat the
                mask image as nonbinary one. The module will use the last
                channel of the given image in `masks` as the mask value.
            do_blend (bool): Blend the generated foreground and the optionally
                predefined background by smooth boundary obtained from Gaussian
                blurs of the foreground `masks` with the given `mask_stds`.
            tile_size (Optional[int]): Tile size of the panorama generation.
                Works best with the default training size of the Stable-
                Diffusion model, i.e., 1024 or 1024 for SD1.5 and 1024 for SDXL.
            bootstrap_steps (int): Overrides the default value. Bootstrapping
                stage steps to encourage region separation. Recommended range:
                1-3.
            boostrap_mix_steps (float): Overrides the default value.
                Bootstrapping background is a linear interpolation between
                background latent and the white image latent. This handle
                controls the mix ratio. Available range: 0-(number of
                bootstrapping inference steps). For example, 2.3 means that for
                the first two steps, white image is used as a bootstrapping
                background and in the third step, mixture of white (0.3) and
                registered background (0.7) is used as a bootstrapping
                background.
            bootstrap_leak_sensitivity (float): Overrides the default value.
                Postprocessing at each inference step by masking away the
                remaining bootstrap backgrounds t Recommended range: 0-1.
            preprocess_mask_cover_alpha (float): Overrides the default value.
                Optional preprocessing where each mask covered by other masks
                is reduced in its alpha value by this specified factor.

        Returns: A PIL.Image image of a panorama (large-size) image.
        """

        ### Simplest cases

        # prompts is None: return background.
        # masks is None but prompts is not None: return prompts
        # masks is not None and prompts is not None: Do StableMultiDiffusion.

        if prompts is None or (isinstance(prompts, (list, tuple, str)) and len(prompts) == 0):
            if background is None and background_prompt is not None:
                return sample(background_prompt, background_negative_prompt, height, width, num_inference_steps, guidance_scale)
            return background
        elif masks is None or (isinstance(masks, (list, tuple)) and len(masks) == 0):
            return sample(prompts, negative_prompts, height, width, num_inference_steps, guidance_scale)


        ### Prepare generation

        if num_inference_steps is not None:
            # self.prepare_lcm_schedule(list(range(num_inference_steps)), num_inference_steps)
            self.prepare_lightning_schedule(list(range(num_inference_steps)), num_inference_steps)

        if guidance_scale is None:
            guidance_scale = self.default_guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0


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

        fg_masks, masks_g, std = self.process_mask(
            masks,
            mask_strengths,
            mask_stds,
            height=height,
            width=width,
            use_boolean_mask=use_boolean_mask,
            timesteps=self.timesteps,
            preprocess_mask_cover_alpha=preprocess_mask_cover_alpha,
        )  # (p, t, 1, H, W)
        bg_masks = (1 - fg_masks.sum(dim=0)).clip_(0, 1)  # (T, 1, h, w)
        has_background = bg_masks.sum() > 0

        h = (height + self.vae_scale_factor - 1) // self.vae_scale_factor
        w = (width + self.vae_scale_factor - 1) // self.vae_scale_factor


        ### Background

        # background == None && background_prompt == None: Initialize with white background.
        # background == None && background_prompt != None: Generate background *along with other prompts*.
        # background != None && background_prompt == None: Retrieve text prompt using BLIP.
        # background != None && background_prompt != None: Use the given arguments.

        # not has_background: no effect of prompt_strength (the mix ratio between fg prompt & bg prompt)
        # has_background && prompt_strength != 1: mix only for this case.

        bg_latent = None
        if has_background:
            if background is None and background_prompt is not None:
                fg_masks = torch.cat((bg_masks[None], fg_masks), dim=0)
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
                background = F.interpolate(background, size=(height, width), mode='bicubic', align_corners=False)
                bg_latent = self.encode_imgs(background)

        # Bootstrapping stage preparation.

        if bootstrap_steps is None:
            bootstrap_steps = self.default_bootstrap_steps
        if boostrap_mix_steps is None:
            boostrap_mix_steps = self.default_boostrap_mix_steps
        if bootstrap_leak_sensitivity is None:
            bootstrap_leak_sensitivity = self.default_bootstrap_leak_sensitivity
        if bootstrap_steps > 0:
            height_ = min(height, tile_size)
            width_ = min(width, tile_size)
            white = self.get_white_background(height, width) # (1, 4, h, w)


        ### Prepare text embeddings (optimized for the minimal encoder batch size)

        # SDXL pipeline settings.
        batch_size = 1
        output_type = 'pil'

        guidance_rescale = 0.7

        prompt_2 = None
        device = self.device
        num_images_per_prompt = 1
        negative_prompt_2 = None

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        negative_crops_coords_top_left = (0, 0)
        negative_original_size = None
        negative_target_size = None
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        text_encoder_lora_scale = None

        prompt_embeds = None
        negative_prompt_embeds = None

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompts,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompts,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

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

            be = prompt_embeds[:1]
            fe = prompt_embeds[1:]
            prompt_embeds = torch.lerp(be, fe, s)  # (p, 77, 1024)

            if negative_prompt_embeds is not None:
                bu = negative_prompt_embeds[:1]
                fu = negative_prompt_embeds[1:]
                if num_prompts > num_nprompts:
                    # # negative prompts = 1; # prompts > 1.
                    assert fu.shape[0] == 1 and fe.shape == num_prompts
                    fu = fu.repeat(num_prompts, 1, 1)
                negative_prompt_embeds = torch.lerp(bu, fu, s)  # (n, 77, 1024)

            be = pooled_prompt_embeds[:1]
            fe = pooled_prompt_embeds[1:]
            pooled_prompt_embeds = torch.lerp(be, fe, s[..., 0])  # (p, 1280)

            if negative_pooled_prompt_embeds is not None:
                bu = negative_pooled_prompt_embeds[:1]
                fu = negative_pooled_prompt_embeds[1:]
                if num_prompts > num_nprompts:
                    # # negative prompts = 1; # prompts > 1.
                    assert fu.shape[0] == 1 and fe.shape == num_prompts
                    fu = fu.repeat(num_prompts, 1)
                negative_pooled_prompt_embeds = torch.lerp(bu, fu, s[..., 0])  # (n, 1280)
        elif negative_prompt_embeds is not None and num_prompts > num_nprompts:
            # # negative prompts = 1; # prompts > 1.
            assert negative_prompt_embeds.shape[0] == 1 and prompt_embeds.shape[0] == num_prompts
            negative_prompt_embeds = negative_prompt_embeds.repeat(num_prompts, 1, 1)

            assert negative_pooled_prompt_embeds.shape[0] == 1 and pooled_prompt_embeds.shape[0] == num_prompts
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(num_prompts, 1)
        # assert negative_prompt_embeds.shape[0] == prompt_embeds.shape[0] == num_prompts
        if num_masks > num_prompts:
            assert masks.shape[0] == num_masks and num_prompts == 1
            prompt_embeds = prompt_embeds.repeat(num_masks, 1, 1)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.repeat(num_masks, 1, 1)

            pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_masks, 1)
            if negative_pooled_prompt_embeds is not None:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(num_masks, 1)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        # SDXL pipeline settings.
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        del negative_prompt_embeds, negative_pooled_prompt_embeds, negative_add_time_ids

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)


        ### Run

        # Latent initialization.
        noise = torch.randn((1, self.unet.config.in_channels, h, w), dtype=self.dtype, device=self.device)
        if self.timesteps[0] < 999 and has_background:
            latent = self.scheduler_add_noise(bg_latent, noise, 0, initial=True)
        else:
            noise = torch.randn((1, self.unet.config.in_channels, h, w), dtype=self.dtype, device=self.device)
            latent = noise * self.scheduler.init_noise_sigma

        if has_background:
            noise_bg_latents = [
                self.scheduler_add_noise(bg_latent, noise, i, initial=True) for i in range(len(self.timesteps))
            ] + [bg_latent]

        # Tiling (if needed).
        if height > tile_size or width > tile_size:
            t = (tile_size + self.vae_scale_factor - 1) // self.vae_scale_factor
            views, tile_masks = get_panorama_views(h, w, t)
            tile_masks = tile_masks.to(self.device)
        else:
            views = [(0, h, 0, w)]
            tile_masks = latent.new_ones((1, 1, h, w))
        value = torch.zeros_like(latent)
        count_all = torch.zeros_like(latent)

        with torch.autocast('cuda'):
            for i, t in enumerate(tqdm(self.timesteps)):
                fg_mask = fg_masks[:, i]
                bg_mask = bg_masks[i:i + 1]

                value.zero_()
                count_all.zero_()
                for j, (h_start, h_end, w_start, w_end) in enumerate(views):
                    fg_mask_ = fg_mask[..., h_start:h_end, w_start:w_end]
                    latent_ = latent[..., h_start:h_end, w_start:w_end].repeat(num_masks, 1, 1, 1)

                    # Additional arguments for the SDXL pipeline.
                    add_time_ids_input = add_time_ids.clone()
                    add_time_ids_input[:, 2] = h_start * self.vae_scale_factor
                    add_time_ids_input[:, 3] = w_start * self.vae_scale_factor
                    add_time_ids_input = add_time_ids_input.repeat_interleave(num_prompts, dim=0)

                    # Bootstrap for tight background.
                    if i < bootstrap_steps:
                        mix_ratio = min(1, max(0, boostrap_mix_steps - i))
                        # Treat the first foreground latent as the background latent if one does not exist.
                        bg_latent_ = noise_bg_latents[i][..., h_start:h_end, w_start:w_end] if has_background else latent_[:1]
                        white_ = white[..., h_start:h_end, w_start:w_end]
                        white_ = self.scheduler_add_noise(white_, noise, i, initial=True)
                        bg_latent_ = mix_ratio * white_ + (1.0 - mix_ratio) * bg_latent_
                        latent_ = (1.0 - fg_mask_) * bg_latent_ + fg_mask_ * latent_

                        # Centering.
                        latent_ = shift_to_mask_bbox_center(latent_, fg_mask_, reverse=True)

                    latent_model_input = torch.cat([latent_] * 2) if do_classifier_free_guidance else latent_
                    latent_model_input = self.scheduler_scale_model_input(latent_model_input, i)

                    # Perform one step of the reverse diffusion.
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids_input}
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=None,
                        cross_attention_kwargs=None,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=guidance_rescale)

                    latent_ = self.scheduler_step(noise_pred, i, latent_)

                    if i < bootstrap_steps:
                        # Uncentering.
                        latent_ = shift_to_mask_bbox_center(latent_, fg_mask_)

                        # Remove leakage (optional).
                        leak = (latent_ - bg_latent_).pow(2).mean(dim=1, keepdim=True)
                        leak_sigmoid = torch.sigmoid(leak / bootstrap_leak_sensitivity) * 2 - 1
                        fg_mask_ = fg_mask_ * leak_sigmoid

                    # Mix the latents.
                    fg_mask_ = fg_mask_ * tile_masks[:, j:j + 1, h_start:h_end, w_start:w_end]
                    value[..., h_start:h_end, w_start:w_end] += (fg_mask_ * latent_).sum(dim=0, keepdim=True)
                    count_all[..., h_start:h_end, w_start:w_end] += fg_mask_.sum(dim=0, keepdim=True)

                latent = torch.where(count_all > 0, value / count_all, value)
                bg_mask = (1 - count_all).clip_(0, 1)  # (T, 1, h, w)
                if has_background:
                    latent = (1 - bg_mask) * latent + bg_mask * noise_bg_latents[i + 1] # bg_latent

                # Noise is added after mixing.
                if i < len(self.timesteps) - 1:
                    latent = self.scheduler_add_noise(latent, None, i + 1)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latent = latent.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latent.device, latent.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latent.device, latent.dtype)
                )
                latent = latent * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latent = latent / self.vae.config.scaling_factor

            image = self.vae.decode(latent, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latent

        # Return PIL Image.
        image = image[0].clip_(-1, 1) * 0.5 + 0.5
        if has_background and do_blend:
            fg_mask = torch.sum(masks_g, dim=0).clip_(0, 1)
            image = blend(image, background[0], fg_mask)
        else:
            image = T.ToPILImage()(image)
        return image
