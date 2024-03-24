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

import concurrent.futures
import time
from typing import Any, Callable, List, Literal, Tuple, Union

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_model(
    model_key: str,
    sd_version: Literal['1.5', 'xl'],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    if model_key.endswith('.safetensors'):
        if sd_version == '1.5':
            pipeline = StableDiffusionPipeline
        elif sd_version == 'xl':
            pipeline = StableDiffusionXLPipeline
        else:
            raise ValueError(f'Stable Diffusion version {sd_version} not supported.')
        return pipeline.from_single_file(model_key, torch_dtype=dtype).to(device)
    try:
        return DiffusionPipeline.from_pretrained(model_key, variant='fp16', torch_dtype=dtype).to(device)
    except:
        return DiffusionPipeline.from_pretrained(model_key, variant=None, torch_dtype=dtype).to(device)


def get_cutoff(cutoff: float = None, scale: float = None) -> float:
    if cutoff is not None:
        return cutoff

    if scale is not None and cutoff is None:
        return 0.5 / scale

    raise ValueError('Either one of `cutoff`, or `scale` should be specified.')


def get_scale(cutoff: float = None, scale: float = None) -> float:
    if scale is not None:
        return scale

    if cutoff is not None and scale is None:
        return 0.5 / cutoff

    raise ValueError('Either one of `cutoff`, or `scale` should be specified.')


def filter_2d_by_kernel_1d(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    assert len(k.shape) in (1,), 'Kernel size should be one of (1,).'
    #  assert len(k.shape) in (1, 2), 'Kernel size should be one of (1, 2).'

    b, c, h, w = x.shape
    ks = k.shape[-1]
    k = k.view(1, 1, -1).repeat(c, 1, 1)

    x = x.permute(0, 2, 1, 3)
    x = x.reshape(b * h, c, w)
    x = F.pad(x, (ks // 2, (ks - 1) // 2), mode='replicate')
    x = F.conv1d(x, k, groups=c)
    x = x.reshape(b, h, c, w).permute(0, 3, 2, 1).reshape(b * w, c, h)
    x = F.pad(x, (ks // 2, (ks - 1) // 2), mode='replicate')
    x = F.conv1d(x, k, groups=c)
    x = x.reshape(b, w, c, h).permute(0, 2, 3, 1)
    return x


def filter_2d_by_kernel_2d(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    assert len(k.shape) in (2, 3), 'Kernel size should be one of (2, 3).'

    x = F.pad(x, (
        k.shape[-2] // 2, (k.shape[-2] - 1) // 2,
        k.shape[-1] // 2, (k.shape[-1] - 1) // 2,
    ), mode='replicate')

    b, c, _, _ = x.shape
    if len(k.shape) == 2 or (len(k.shape) == 3 and k.shape[0] == 1):
        k = k.view(1, 1, *k.shape[-2:]).repeat(c, 1, 1, 1)
        x = F.conv2d(x, k, groups=c)
    elif len(k.shape) == 3:
        assert k.shape[0] == b, \
            'The number of kernels should match the batch size.'

        k = k.unsqueeze(1)
        x = F.conv2d(x.permute(1, 0, 2, 3), k, groups=b).permute(1, 0, 2, 3)
    return x


@amp.autocast(False)
def filter_by_kernel(
    x: torch.Tensor,
    k: torch.Tensor,
    is_batch: bool = False,
) -> torch.Tensor:
    k_dim = len(k.shape)
    if k_dim == 1 or k_dim == 2 and is_batch:
        return filter_2d_by_kernel_1d(x, k)
    elif k_dim == 2 or k_dim == 3 and is_batch:
        return filter_2d_by_kernel_2d(x, k)
    else:
        raise ValueError('Kernel size should be one of (1, 2, 3).')


def gen_gauss_lowpass_filter_2d(
    std: torch.Tensor,
    window_size: int = None,
) -> torch.Tensor:
    # Gaussian kernel size is odd in order to preserve the center.
    if window_size is None:
        window_size = (
            2 * int(np.ceil(3 * std.max().detach().cpu().numpy())) + 1)

    y = torch.arange(
        window_size, dtype=std.dtype, device=std.device
    ).view(-1, 1).repeat(1, window_size)
    grid = torch.stack((y.t(), y), dim=-1)
    grid -= 0.5 * (window_size - 1) # (W, W)
    var = (std * std).unsqueeze(-1).unsqueeze(-1)
    distsq = (grid * grid).sum(dim=-1).unsqueeze(0).repeat(*std.shape, 1, 1)
    k = torch.exp(-0.5 * distsq / var)
    k /= k.sum(dim=(-2, -1), keepdim=True)
    return k


def gaussian_lowpass(
    x: torch.Tensor,
    std: Union[float, Tuple[float], torch.Tensor] = None,
    cutoff: Union[float, torch.Tensor] = None,
    scale: Union[float, torch.Tensor] = None,
) -> torch.Tensor:
    if std is None:
        cutoff = get_cutoff(cutoff, scale)
        std = 0.5 / (np.pi * cutoff)
    if isinstance(std, (float, int)):
        std = (std, std)
    if isinstance(std, torch.Tensor):
        """Using nn.functional.conv2d with Gaussian kernels built in runtime is
        80% faster than transforms.functional.gaussian_blur for individual
        items.
        
        (in GPU); However, in CPU, the result is exactly opposite. But you
        won't gonna run this on CPU, right?
        """
        if len(list(s for s in std.shape if s != 1)) >= 2:
            raise NotImplementedError(
                'Anisotropic Gaussian filter is not currently available.')

        # k.shape == (B, W, W).
        k = gen_gauss_lowpass_filter_2d(std=std.view(-1))
        if k.shape[0] == 1:
            return filter_by_kernel(x, k[0], False)
        else:
            return filter_by_kernel(x, k, True)
    else:
        # Gaussian kernel size is odd in order to preserve the center.
        window_size = tuple(2 * int(np.ceil(3 * s)) + 1 for s in std)
        return TF.gaussian_blur(x, window_size, std)


def blend(
    fg: Union[torch.Tensor, Image.Image],
    bg: Union[torch.Tensor, Image.Image],
    mask: Union[torch.Tensor, Image.Image],
    std: float = 0.0,
) -> Image.Image:
    if not isinstance(fg, torch.Tensor):
        fg = T.ToTensor()(fg)
    if not isinstance(bg, torch.Tensor):
        bg = T.ToTensor()(bg)
    if not isinstance(mask, torch.Tensor):
        mask = (T.ToTensor()(mask) < 0.5).float()[:1]
    if std > 0:
        mask = gaussian_lowpass(mask[None], std)[0].clip_(0, 1)
    return T.ToPILImage()(fg * mask + bg * (1 - mask))


def get_panorama_views(
    panorama_height: int,
    panorama_width: int,
    window_size: int = 64,
) -> tuple[List[Tuple[int]], torch.Tensor]:
    stride = window_size // 2
    is_horizontal = panorama_width > panorama_height
    num_blocks_height = (panorama_height - window_size + stride - 1) // stride + 1
    num_blocks_width = (panorama_width - window_size + stride - 1) // stride + 1
    total_num_blocks = num_blocks_height * num_blocks_width

    half_fwd = torch.linspace(0, 1, (window_size + 1) // 2)
    half_rev = half_fwd.flip(0)
    if window_size % 2 == 1:
        half_rev = half_rev[1:]
    c = torch.cat((half_fwd, half_rev))
    one = torch.ones_like(c)
    f = c.clone()
    f[:window_size // 2] = 1
    b = c.clone()
    b[-(window_size // 2):] = 1

    h = [one] if num_blocks_height == 1 else [f] + [c] * (num_blocks_height - 2) + [b]
    w = [one] if num_blocks_width == 1 else [f] + [c] * (num_blocks_width - 2) + [b]

    views = []
    masks = torch.zeros(total_num_blocks, panorama_height, panorama_width) # (n, h, w)
    for i in range(total_num_blocks):
        hi, wi = i // num_blocks_width, i % num_blocks_width
        h_start = hi * stride
        h_end = min(h_start + window_size, panorama_height)
        w_start = wi * stride
        w_end = min(w_start + window_size, panorama_width)
        views.append((h_start, h_end, w_start, w_end))

        h_width = h_end - h_start
        w_width = w_end - w_start
        masks[i, h_start:h_end, w_start:w_end] = h[hi][:h_width, None] * w[wi][None, :w_width]

    # Sum of the mask weights at each pixel `masks.sum(dim=1)` must be unity.
    return views, masks[None] # (1, n, h, w)


def shift_to_mask_bbox_center(im: torch.Tensor, mask: torch.Tensor, reverse: bool = False) -> List[int]:
    h, w = mask.shape[-2:]
    device = mask.device
    mask = mask.reshape(-1, h, w)
    # assert mask.shape[0] == im.shape[0]
    h_occupied = mask.sum(dim=-2) > 0
    w_occupied = mask.sum(dim=-1) > 0
    l = torch.argmax(h_occupied * torch.arange(w, 0, -1).to(device), 1, keepdim=True).cpu()
    r = torch.argmax(h_occupied * torch.arange(w).to(device), 1, keepdim=True).cpu()
    t = torch.argmax(w_occupied * torch.arange(h, 0, -1).to(device), 1, keepdim=True).cpu()
    b = torch.argmax(w_occupied * torch.arange(h).to(device), 1, keepdim=True).cpu()
    tb = (t + b + 1) // 2
    lr = (l + r + 1) // 2
    shifts = (tb - (h // 2), lr - (w // 2))
    shifts = torch.cat(shifts, dim=1) # (p, 2)
    if reverse:
        shifts = shifts * -1
    return torch.stack([i.roll(shifts=s.tolist(), dims=(-2, -1)) for i, s in zip(im, shifts)], dim=0)


class Streamer:
    def __init__(self, fn: Callable, ema_alpha: float = 0.9) -> None:
        self.fn = fn
        self.ema_alpha = ema_alpha

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = self.executor.submit(fn)
        self.image = None

        self.prev_exec_time = 0
        self.ema_exec_time = 0

    @property
    def throughput(self) -> float:
        return 1.0 / self.ema_exec_time if self.ema_exec_time else float('inf')

    def timed_fn(self) -> Any:
        start = time.time()
        res = self.fn()
        end = time.time()
        self.prev_exec_time = end - start
        self.ema_exec_time = self.ema_exec_time * self.ema_alpha + self.prev_exec_time * (1 - self.ema_alpha)
        return res

    def __call__(self) -> Any:
        if self.future.done() or self.image is None:
            # get the result (the new image) and start a new task
            image = self.future.result()
            self.future = self.executor.submit(self.timed_fn)
            self.image = image
            return image
        else:
            # if self.fn() is not ready yet, use the previous image
            # NOTE: This assumes that we have access to a previously generated image here.
            # If there's no previous image (i.e., this is the first invocation), you could fall 
            # back to some default image or handle it differently based on your requirements.
            return self.image