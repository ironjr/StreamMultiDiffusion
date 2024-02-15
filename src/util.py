from typing import Tuple, Union
from PIL import Image

import numpy as np

import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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
    std: float = 10.0,
) -> Image.Image:
    if not isinstance(fg, torch.Tensor):
        fg = T.ToTensor()(fg)
    if not isinstance(bg, torch.Tensor):
        bg = T.ToTensor()(bg)
    if not isinstance(mask, torch.Tensor):
        mask = (T.ToTensor()(mask) < 0.5).float()[:1]
    mask = gaussian_lowpass(mask[None], std)[0]
    return T.ToPILImage()(fg * mask + bg * (1 - mask))