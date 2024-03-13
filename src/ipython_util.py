# Copyright (c) 2024 StreamMultiDiffusion Authors

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

from typing import List, Union

import numpy as np
from IPython.display import display

import torch
import torchvision.transforms as T
from diffusers.utils import make_image_grid


def dispt(
    tensors: Union[torch.Tensor, List[torch.Tensor]],
    size_idx: int = -1,
    row: int = 1,
    col: int = -1,
) -> None:
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]

    tf = lambda x: T.ToPILImage()(x.float().clip_(0, 1))

    ims = []
    for t in tensors:
        if len(t.shape) == 4:
            if t.shape[0] == 1:
                ims.append(t[0])
            else:
                ims.extend([s[0] for s in t.chunk(t.shape[0])])
        elif len(t.shape) == 3:
            ims.append(t)
        elif len(t.shape) == 2:
            ims.append(t[None])
        else:
            raise ValueError(
                f'Tensor of shape {t.shape} cannot be processed!'
                ' Only 2D, 3D, 4D tensors are allowed.'
            )
    ims = list(map(tf, ims))
    if size_idx >= 0:
        ims = [im.resize(ims[size_idx].size) for im in ims]

    if row <= 0 and col <= 0:
        row = 1
        col = len(ims)
    elif row <= 0:
        row = int(np.ceil(len(ims) / col))
    elif col <= 0:
        col = int(np.ceil(len(ims) / row))

    display(make_image_grid(ims, row, col))