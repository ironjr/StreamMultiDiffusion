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

import copy
from typing import Optional, Union
from PIL import Image
import torch


class BackgroundObject:
    def __init__(
        self,
        image: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> None:
        self.image = image
        self.prompt = prompt
        self.negative_prompt = negative_prompt

    @property
    def is_empty(self) -> bool:
        return (
            self.image is None and
            self.prompt is None and
            self.negative_prompt is None
        )

    def extra_repr(self) -> str:
        return ''

    def __repr__(self) -> str:
        strings = []
        if self.image is not None:
            if isinstance(self.image, Image.Image):
                image_str = f'Image(size={str(self.image.size)})'
            else:
                image_str = f'Tensor(shape={str(self.image.shape)})'
            strings.append(f'image={image_str}')
        if self.prompt is not None:
            strings.append(f'prompt="{self.prompt}"')
        if self.negative_prompt is not None:
            strings.append(f'negative_prompt="{self.negative_prompt}"')
        extra_repr = self.extra_repr()
        if extra_repr != '':
            strings.append(extra_repr)
        return f'{type(self).__name__}({", ".join(strings)})'


class LayerObject:
    def __init__(
        self,
        idx: Optional[int] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        suffix: Optional[str] = None,
        prompt_strength: Optional[float] = None,
        mask: Optional[Union[torch.Tensor, Image.Image]] = None,
        mask_std: Optional[float] = None,
        mask_strength: Optional[float] = None,
    ) -> None:
        self.idx = idx
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.suffix = suffix
        self.prompt_strength = prompt_strength
        self.mask = mask
        self.mask_std = mask_std
        self.mask_strength = mask_strength

    @property
    def is_empty(self) -> bool:
        return (
            self.prompt is None and
            self.negative_prompt is None and
            self.prompt_strength is None and
            self.mask is None and
            self.mask_strength is None and
            self.mask_std is None
        )

    def merge(self, other: 'LayerObject') -> bool: # Overriden or not.
        if self.idx != other.idx:
            # Merge only the modification requests for the same layer.
            return False

        if self.prompt is None and other.prompt is not None:
            self.prompt = copy.deepcopy(other.prompt)
        if self.negative_prompt is None and other.negative_prompt is not None:
            self.negative_prompt = copy.deepcopy(other.negative_prompt)
        if self.suffix is None and other.suffix is not None:
            self.suffix = copy.deepcopy(other.suffix)
        if self.prompt_strength is None and other.prompt_strength is not None:
            self.prompt_strength = copy.deepcopy(other.prompt_strength)
        if self.mask is None and other.mask is not None:
            self.mask = copy.deepcopy(other.mask)
        if self.mask_strength is None and other.mask_strength is not None:
            self.mask_strength = copy.deepcopy(other.mask_strength)
        if self.mask_std is None and other.mask_std is not None:
            self.mask_std = copy.deepcopy(other.mask_std)
        return True

    def extra_repr(self) -> str:
        return ''

    def __repr__(self) -> str:
        strings = []
        if self.idx is not None:
            strings.append(f'idx={self.idx}')
        if self.prompt is not None:
            strings.append(f'prompt="{self.prompt}"')
        if self.negative_prompt is not None:
            strings.append(f'negative_prompt="{self.negative_prompt}"')
        if self.suffix is not None:
            strings.append(f'suffix="{self.suffix}"')
        if self.prompt_strength is not None:
            strings.append(f'prompt_strength={self.prompt_strength}')
        if self.mask is not None:
            if isinstance(self.mask, Image.Image):
                mask_str = f'Image(size={str(self.mask.size)})'
            else:
                mask_str = f'Tensor(shape={str(self.mask.shape)})'
            strings.append(f'mask={mask_str}')
        if self.mask_std is not None:
            strings.append(f'mask_std={self.mask_std}')
        if self.mask_strength is not None:
            strings.append(f'mask_strength={self.mask_strength}')
        extra_repr = self.extra_repr()
        if extra_repr != '':
            strings.append(extra_repr)
        return f'{type(self).__name__}({", ".join(strings)})'


class BackgroundState(BackgroundObject):
    def __init__(
        self,
        image: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        latent: Optional[torch.Tensor] = None,
        embed: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(image, prompt, negative_prompt)
        self.latent = latent
        self.embed = embed

    @property
    def is_incomplete(self) -> bool:
        return (
            self.image is None or
            self.prompt is None or
            self.negative_prompt is None or
            self.latent is None or
            self.embed is None
        )

    def extra_repr(self) -> str:
        strings = []
        if self.latent is not None:
            strings.append(f'latent=Tensor(shape={str(self.latent.shape)})')
        if self.embed is not None:
            strings.append(f'embed=Tuple[Tensor(shape={str(self.embed[0].shape)})]')
        return ', '.join(strings)


# TODO
# class LayerState:
#     def __init__(
#         self,
#         prompst: List[str] = [],
#         negative_prompts: List[str] = [],
#         suffix: List[str] = [],
#         masks: Optional[torch.Tensor] = None,
#         mask_std: Optional[torch.Tensor] = None,
#         mask_strength: Optional[torch.Tensor] = None,
#         original_masks: Optional[Union[torch.Tensor, List[Image.Image]]] = None,
#     ) -> None:
#         self.prompts = prompts
#         self.negative_prompts = negative_prompts
#         self.suffix = suffix
#         self.masks = masks
#         self.mask_std = mask_std
#         self.mask_strength = mask_strength
#         self.original_masks = original_masks

#     def __len__(self) -> int:
#         self.check_integrity(True)
#         return len(self.prompts)

#     @property
#     def is_empty(self) -> bool:
#         self.check_integrity(True)
#         return len(self.prompt) == 0

#     def check_integrity(self, throw_error: bool = True) -> bool:
#         p = len(self.prompts)
#         flag = (
#             p != len(self.negative_prompts) or
#             p != len(self.suffix) or
#             p != len(self.masks) or
#             p != len(self.mask_std) or
#             p != len(self.mask_strength) or
#             p != len(self.original_masks)
#         )
#         if flag and throw_error:
#             print(
#                 f'LayerState(\n\tlen(prompts): {p},\n\tlen(negative_prompts): {len(self.negative_prompts)},\n\t'
#                 f'len(suffix): {len(self.suffix)},\n\tlen(masks): {len(self.masks)},\n\t'
#                 f'len(mask_std): {len(self.mask_std)},\n\tlen(mask_strength): {len(self.mask_strength)},\n\t'
#                 f'len(original_masks): {len(self.original_masks)}\n)'
#             )
#             raise ValueError('LayerState is corrupted!')
#         return not flag

#     def extra_repr(self) -> str:
#         strings = []
#         if self.idx is not None:
#             strings.append(f'idx={self.idx}')
#         if self.prompt is not None:
#             strings.append(f'prompt="{self.prompt}"')
#         if self.negative_prompt is not None:
#             strings.append(f'negative_prompt="{self.negative_prompt}"')
#         if self.suffix is not None:
#             strings.append(f'suffix="{self.suffix}"')
#         if self.mask is not None:
#             if isinstance(self.mask, Image.Image):
#                 mask_str = f'PIL.Image.Image(size={str(self.mask.size)})'
#             else:
#                 mask_str = f'torch.Tensor(shape={str(self.mask.shape)})'
#             strings.append(f'mask={mask_str}')
#         if self.mask_std is not None:
#             strings.append(f'mask_std={self.mask_std}')
#         if self.mask_strength is not None:
#             strings.append(f'mask_strength={self.mask_strength}')
#         return f'{type(self).__name__}({", ".join(strings)})'