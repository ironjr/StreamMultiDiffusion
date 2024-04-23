<div align="center">

<h1>StreamMultiDiffusion: Real-Time Interactive Generation</br>with Region-Based Semantic Control</h1>

[**Jaerin Lee**](http://jaerinlee.com/) · [**Daniel Sungho Jung**](https://dqj5182.github.io/) · [**Kanggeon Lee**](https://github.com/dlrkdrjs97/) · [**Kyoung Mu Lee**](https://cv.snu.ac.kr/index.php/~kmlee/)


<p align="center">
    <img src="assets/logo_cvlab.png" height=60>
</p>


[![Project](https://img.shields.io/badge/Project-Page-green)](https://jaerinlee.com/research/streammultidiffusion)
[![ArXiv](https://img.shields.io/badge/Arxiv-2403.09055-red)](https://arxiv.org/abs/2403.09055)
[![Github](https://img.shields.io/github/stars/ironjr/StreamMultiDiffusion)](https://github.com/ironjr/StreamMultiDiffusion)
[![X](https://img.shields.io/twitter/url?label=_ironjr_&url=https%3A%2F%2Ftwitter.com%2F_ironjr_)](https://twitter.com/_ironjr_)
[![LICENSE](https://img.shields.io/badge/license-MIT-lightgrey)](https://github.com/ironjr/StreamMultiDiffusion/blob/main/LICENSE)
[![HFPaper](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2403.09055)

[![HFDemo1](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-SD1.5-yellow)](https://huggingface.co/spaces/ironjr/SemanticPalette)
[![HFDemo2](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-SDXL-yellow)](https://huggingface.co/spaces/ironjr/SemanticPaletteXL)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/SemanticPalette-jupyter/blob/main/SemanticPalette_jupyter.ipynb)

</div>

<p align="center">
  <img src="./assets/figure_one.png" width=100%>
</p>

tl;dr: StreamMultiDiffusion is a *real-time* *interactive* *multiple*-text-to-image generation from user-assigned *regional* text prompts.
In other words, **you can now draw ✍️ using brushes 🖌️ that paints *meanings* 🧠 in addition to *colors*** 🌈!

<details>
  
<summary>What's the paper about?</summary>
Our paper is mainly about establishing the compatibility between region-based controlling techniques of <a href="https://multidiffusion.github.io/">MultiDiffusion</a> and acceleration techniques of <a href="https://latent-consistency-models.github.io/">LCM</a> and <a href="https://github.com/cumulo-autumn/StreamDiffusion">StreamDiffusion</a>.
To our surprise, these works were not compatible before, limiting the possible applications from both branches of works.
The effect of acceleration and stabilization of multiple region-based text-to-image generation technique is demonstrated using <a href="https://huggingface.co/runwayml/stable-diffusion-v1-5">StableDiffusion v1.5</a> in the video below ⬇️:

https://github.com/ironjr/MagicDraw/assets/12259041/9dda9740-58ba-4a96-b8c1-d40765979bd7

The video means that this project finally lets you work with **large size image generation with fine-grained regional prompt control**.
Previously, this was not feasible at all.
Taking an hour per trial means that you cannot sample multiple times to pick the best generation you want or to tune the generation process to realize your intention.
However, we have decreased the latency **from an hour to a minute**, making the technology workable for creators (hopefully).

</details>

---

- [⭐️ Features](#---features)
- [🚩 Updates](#---updates)
- [🤖 Installation](#---installation)
- [⚡ Usage](#---usage)
  * [Overview](#overview)
  * [Basic Usage (Python)](#basic-usage--python-)
  * [Streaming Generation Process](#streaming-generation-process)
  * [Region-Based Multi-Text-to-Image Generation](#region-based-multi-text-to-image-generation)
  * [Larger Region-Based Multi-Text-to-Image Generation](#larger-region-based-multi-text-to-image-generation)
  * [Image Inpainting with Prompt Separation](#image-inpainting-with-prompt-separation)
  * [Panorama Generation](#panorama-generation)
  * [Basic StableDiffusion](#basic-stablediffusion)
  * [Basic Usage (GUI)](#basic-usage--gui-)
  * [Demo Application (Semantic Palette)](#demo-application--semantic-palette-)
  * [Basic Usage (CLI)](#basic-usage--cli-)
- [💼 Further Information](#---further-information)
  * [User Interface (GUI)](#user-interface--gui-)
  * [Demo Application Architecture](#demo-application-architecture)
- [🙋 FAQ](#---faq)
  * [What is Semantic Palette Anyway?](#what-is--semantic-palette--anyway-)
- [🚨 Notice](#---notice)
- [🌏 Citation](#---citation)
- [🤗 Acknowledgement](#---acknowledgement)
- [📧 Contact](#---contact)

---

## ⭐️ Features


| ![usage1](./assets/feature1.gif) | ![usage2](./assets/feature3.gif) |  ![usage3](./assets/feature2.gif)  |
| :----------------------------: | :----------------------------: | :----------------------------: |

1. **Interactive image generation from scratch with fine-grained region control.** In other words, you paint images using meainings.

2. **Prompt separation.** Be bothered no more by unintentional content mixing when generating two or more objects at the same time!

3. **Real-time image inpainting and editing.** Basically, you draw upon any uploaded photo or a piece of art you want.

---

## 🚩 Updates

![demo_v2](./assets/demo_v2.gif)

- 🔥 April 23, 2024: Real-time interactive generation demo is updated to [version 2](https://github.com/ironjr/StreamMultiDiffusion/tree/main/demo/stream_v2)! We now have fully responsive interface with `gradio.ImageEditor`. Huge thanks to [@pngwn](https://github.com/pngwn) and Hugging Face 🤗 Gradio team for the [great update (4.27)](https://www.gradio.app/changelog#4-27-0)!
- 🔥 March 24, 2024: Our new demo app _Semantic Palette SDXL_ is out at [Hugging Face Space](https://huggingface.co/spaces/ironjr/SemanticPaletteXL)! Great thanks to [Cagliostro Research Lab](https://cagliostrolab.net/) for the permission of [Animagine XL 3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1) model used in the demo!
- 🔥 March 24, 2024: We now (experimentally) support SDXL with [Lightning LoRA](https://huggingface.co/ByteDance/SDXL-Lightning) in our semantic palette demo! Streaming type with SDXL-Lighning is under development.
- 🔥 March 23, 2024: We now support `.safetensors` type models. Please see the instructions in Usage section.
- 🔥 March 22, 2024: Our demo app _Semantic Palette_ is now available on [Google Colab](https://colab.research.google.com/github/camenduru/SemanticPalette-jupyter/blob/main/SemanticPalette_jupyter.ipynb)! Huge thanks to [@camenduru](https://github.com/camenduru)!
- 🔥 March 22, 2024: The app _Semantic Palette_ is now included in the repository! Run `python src/demo/semantic_palette/app.py --model "your model here"` to run the app from your local machine.
- 🔥 March 19, 2024: Our first public demo of _semantic palette_ is out at [Hugging Face Space](https://huggingface.co/spaces/ironjr/SemanticPalette)! We would like to give our biggest thanks to the almighty Hugging Face 🤗 team for their help!
- 🔥 March 16, 2024: Added examples and instructions for region-based generation, panorama generation, and inpainting.
- 🔥 March 15, 2024: Added detailed instructions in this README for creators.
- 🔥 March 14, 2024: We have released our paper, StreamMultiDiffusion on [arXiv](https://arxiv.org/abs/2403.09055).
- 🔥 March 13, 2024: Code release!

---

## 🤖 Installation

```bash
conda create -n smd python=3.10 && conda activate smd
git clone https://github.com/ironjr/StreamMultiDiffusion
pip install -r requirements.txt
```

## ⚡ Usage

### Overview

StreamMultiDiffusion is served in serveral different forms.

1. The main GUI demo powered by Gradio is available at `demo/stream/app.py`. Just type the below line in your command prompt and open `https://localhost:8000` with any web browser will launch the app.

```bash
cd demo/stream
python app.py --model "your stable diffusion 1.5 checkpoint" --height 512 --width 512 --port 8000
```

2. The GUI demo _Semantic Palette_ for _SD1.5_ checkpoints is available at `demo/semantic_palette/app.py`. The public version can be found at [![HFDemo1](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-SD1.5-yellow)](https://huggingface.co/spaces/ironjr/SemanticPalette) and at [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/SemanticPalette-jupyter/blob/main/SemanticPalette_jupyter.ipynb).

```bash
cd demo/semantic_palette
python app.py --model "your stable diffusion 1.5 checkpoint" --height 512 --width 512 --port 8000
```

3. The GUI demo _Semantic Palette_ for _SDXL_ checkpoints is available at `demo/semantic_palette_sdxl/app.py`. The public version can be found at [![HFDemo2](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-SDXL-yellow)](https://huggingface.co/spaces/ironjr/SemanticPaletteXL).

```bash
cd demo/semantic_palette_sdxl
python app.py --model "your stable diffusion 1.5 checkpoint" --height 512 --width 512 --port 8000
```

4. Jupyter Lab demos are available in the `notebooks` directory. Simply type `jupyter lab` in the command prompt will open a Jupyter server.

5. As a python library by importing the `model` in `src`. For detailed examples and interfaces, please see the Usage section below.



### Basic Usage (Python)

The main python modules in our project is two-fold: (1) [`model.StableMultiDiffusionPipeline`](https://github.com/ironjr/StreamMultiDiffusion/blob/main/src/model/stablemultidiffusion_pipeline.py) for single-call generation (might be more preferable for CLI users), and (2) [`model.StreamMultiDiffusion`](https://github.com/ironjr/StreamMultiDiffusion/blob/main/src/model/streammultidiffusion.py) for streaming application such as the [one](https://github.com/ironjr/StreamMultiDiffusion/blob/main/src/app.py) in the main figure of this README page.
We provide minimal examples for the possible applications below.


### Streaming Generation Process

With [multi-prompt stream batch](https://arxiv.org/abs/2403.09055), our modification to the [original stream batch architecture](https://github.com/cumulo-autumn/StreamDiffusion) by [@cumulo_autumn](https://twitter.com/cumulo_autumn), we can stream this multi-prompt text-to-image generation process to generate images for ever.

**Result:**

| ![mask](./assets/zeus/prompt.png) | ![result](./assets/athena_stream.gif) |
| :----------------------------: | :----------------------------: |
| Semantic Brush Input | Generated Stream |

**Code:**

```python
import torch
from util import seed_everything, Streamer
from model import StreamMultiDiffusion

# The following packages are imported only for loading the images.
import torchvision.transforms as T
import requests
import time
import imageio # This is not included in our requirements.txt!
from functools import reduce
from io import BytesIO
from PIL import Image


seed = 2024
device = 0
height = 768
width = 512

# Load the module.
device = torch.device(f'cuda:{device}')
smd = StreamMultiDiffusion(
    device,
    hf_key='ironjr/BlazingDriveV11m',
    height=height,
    width=width,
    cfg_type='none',
    autoflush=True,
    use_tiny_vae=True,
    mask_type='continuous',
    bootstrap_steps=2,
    bootstrap_mix_steps=1.5,
    seed=seed,
)

# Load the masks.
masks = []
for i in range(1, 3):
    url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/zeus/prompt_p{i}.png'
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).convert('RGBA')
    mask = (T.ToTensor()(mask)[-1:] > 0.5).float()
    masks.append(mask)
# In this example, background is simply set as non-marked regions.
background = reduce(torch.logical_and, [m == 0 for m in masks])

# Register a background, prompts, and masks (this can be called multiple times).
smd.update_background(Image.new(size=(width, height), mode='RGB', color=(255, 255, 255)))
smd.update_single_layer(
    idx=0,
    prompt='a photo of Mount Olympus',
    negative_prompt='worst quality, bad quality, normal quality, cropped, framed',
    mask=background,
    mask_strength=1.0,
    mask_std=0.0,
    prompt_strength=1.0,
)
smd.update_single_layer(
    idx=1,
    prompt='1girl, looking at viewer, lifts arm, smile, happy, Greek goddess Athena',
    negative_prompt='worst quality, bad quality, normal quality, cropped, framed',
    mask=masks[0],
    mask_strength=1.0,
    mask_std=0.0,
    prompt_strength=1.0,
)
smd.update_single_layer(
    idx=2,
    prompt='a small, sitting owl',
    negative_prompt='worst quality, bad quality, normal quality, cropped, framed',
    mask=masks[1],
    mask_strength=1.0,
    mask_std=0.0,
    prompt_strength=1.0,
)


# Generate images... forever.
# while True:
#     image = smd()
#     image.save(f'{str(int(time.time() % 100000))}.png') # This will take up your hard drive pretty much soon.
#     display(image) # If `from IPython.display import display` is called.
#
#     You can also intercept the process in the middle of the generation by updating other background, prompts or masks.
#     smd.update_single_layer(
#         idx=2,
#         prompt='a small, sitting owl',
#         negative_prompt='worst quality, bad quality, normal quality, cropped, framed',
#         mask=masks[1],
#         mask_strength=1.0,
#         mask_std=0.0,
#         prompt_strength=1.0,
#     )

# Or make a video/gif from your generation stream (requires `imageio`)
frames = []
for _ in range(50):
    image = smd()
    frames.append(image)
imageio.mimsave('my_beautiful_creation.gif', frames, loop=0)
```

---

### Region-Based Multi-Text-to-Image Generation

We support arbitrary-sized image generation from arbitrary number of prompt-mask pairs.
The first example is a simple example of generation 
Notice that **our generation results also obeys strict prompt separation**.


**Result:**

| ![mask](./assets/timessquare/timessquare_full.png) | ![result](./assets/timessquare_generation.png) |
| :----------------------------: | :----------------------------: |
| Semantic Brush Input | Generated Image (10 sec) |

<p align="center">
    No more unwanted prompt mixing! Brown boy and pink girl generated simultaneously without a problem.
</p>

**Code:**

```python
import torch
from model import StableMultiDiffusionPipeline
from util import seed_everything

# The following packages are imported only for loading the images.
import torchvision.transforms as T
import requests
from functools import reduce
from io import BytesIO
from PIL import Image


seed = 2024
device = 0

# Load the module.
seed_everything(seed)
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionPipeline(
    device,
    hf_key='ironjr/BlazingDriveV11m',
)

# Load prompts.
prompts = [
    # Background prompt.
    '1girl, 1boy, times square',
    # Foreground prompts.
    '1boy, looking at viewer, brown hair, casual shirt',
    '1girl, looking at viewer, pink hair, leather jacket',
]
negative_prompts = [
    '',
    '1girl', # (Optional) The first prompt is a boy so we don't want a girl.
    '1boy', # (Optional) The first prompt is a girl so we don't want a boy.
]
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

# Load masks.
masks = []
for i in range(1, 3):
    url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/timessquare/timessquare_{i}.png'
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).convert('RGBA')
    mask = (T.ToTensor()(mask)[-1:] > 0.5).float()
    masks.append(mask)
# In this example, background is simply set as non-marked regions.
background = reduce(torch.logical_and, [m == 0 for m in masks])
masks = torch.stack([background] + masks, dim=0).float()

height, width = masks.shape[-2:] # (768, 768) in this example.

# Sample an image.
image = smd(
    prompts,
    negative_prompts,
    masks=masks,
    mask_strengths=1,
    mask_stds=0,
    height=height,
    width=width,
    bootstrap_steps=2,
)
image.save('my_beautiful_creation.png')
```

---

### (🔥NEW!) Region-Based Multi-Text-to-Image Generation with Custom SDXL

We support arbitrary-sized image generation from arbitrary number of prompt-mask pairs using custom SDXL models.
This is powered by [SDXL-Lightning LoRA](https://huggingface.co/ByteDance/SDXL-Lightning) and our stabilization trick for MultiDiffusion in conjunction with Lightning-type sampling algorithm.

#### Known Issue:

SDXL-Lightning support is currently experimental, so there can be additional issues I have not yet noticed.
Please open an issue or a pull request if you find any.
These are the currently known SDXL-Lightning-specific issues compared to SD1.5 models.
- The model tends to be less obedient to the text prompts. SDXL-Lightning-specific prompt engineering may be required. The problem is less severe in custom models, such as [this](https://huggingface.co/cagliostrolab/animagine-xl-3.1).
- The vanilla SDXL-Lightning model produces NaNs when used as a FP16 variant. Please use `dtype=torch.float32` option for initializing `StableMultiDiffusionSDXLPipeline` if you want the _vanilla_ version of the SDXL-Lightning. This is _not_ a problem when using a custom checkpoint. You can use `dtype=torch.float16`.

**Result:**

| ![mask](./assets/fantasy_large/fantasy_large_full.png) | ![result](./assets/fantasy_large_generation.png) |
| :----------------------------: | :----------------------------: |
| Semantic Brush Input | Generated Image (12 sec) |

<p align="center">
    1024x1024 image generated with <a href="https://huggingface.co/ByteDance/SDXL-Lightning">SDXL-Lightning LoRA</a> and <a href="https://huggingface.co/cagliostrolab/animagine-xl-3.1">Animagine XL 3.1</a> checkpoint.
</p>

**Code:**

```python
import torch
from model import StableMultiDiffusionSDXLPipeline
from util import seed_everything
from prompt_util import print_prompts, preprocess_prompts

# The following packages are imported only for loading the images.
import torchvision.transforms as T
import requests
from functools import reduce
from io import BytesIO
from PIL import Image


seed = 0
device = 0

# Load the module.
seed_everything(seed)
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionSDXLPipeline(
    device,
    hf_key='cagliostrolab/animagine-xl-3.1',
    has_i2t=False,
)

# Load prompts.
prompts = [
    # Background prompt.
    'purple sky, planets, planets, planets, stars, stars, stars',
    # Foreground prompts.
    'a photo of the dolomites, masterpiece, absurd quality, background, no humans',
    '1girl, looking at viewer, pretty face, blue hair, fantasy style, witch, magi, robe',
]
negative_prompts = [
    '1girl, 1boy, humans, humans, humans',
    '1girl, 1boy, humans, humans, humans',
    '',
]
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

# Preprocess prompts for better results.
prompts, negative_prompts = preprocess_prompts(
    prompts,
    negative_prompts,
    style_name='(None)',
    quality_name='Standard v3.1',
)

# Load masks.
masks = []
for i in range(1, 3):
    url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/fantasy_large/fantasy_large_{i}.png'
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).convert('RGBA')
    mask = (T.ToTensor()(mask)[-1:] > 0.5).float()
    masks.append(mask)
# In this example, background is simply set as non-marked regions.
background = reduce(torch.logical_and, [m == 0 for m in masks])
masks = torch.stack([background] + masks, dim=0).float()

height, width = masks.shape[-2:] # (1024, 1024) in this example.

# Sample an image.
image = smd(
    prompts,
    negative_prompts,
    masks=masks,
    mask_strengths=1,
    mask_stds=0,
    height=height,
    width=width,
    bootstrap_steps=2,
    guidance_scale=0,
)
image.save('my_beautiful_creation.png')
```

---

### *Larger* Region-Based Multi-Text-to-Image Generation

The below code reproduces the results in the [second video](https://github.com/ironjr/MagicDraw/assets/12259041/9dda9740-58ba-4a96-b8c1-d40765979bd7) of this README page.
The original MultiDiffusion pipeline using 50 step DDIM sampler takes roughly an hour to run the code, but we have reduced in down to **a minute**.

**Result:**

| ![mask](./assets/irworobongdo/irworobongdo_full.png) |
| :----------------------------: |
| Semantic Brush Input |
|  ![result](./assets/irworobongdo_generation.png) |
| Generated Image (59 sec) |

**Code:**

```python
import torch
from model import StableMultiDiffusionPipeline
from util import seed_everything

# The following packages are imported only for loading the images.
import torchvision.transforms as T
import requests
from functools import reduce
from io import BytesIO
from PIL import Image


seed = 2024
device = 0

# Load the module.
seed_everything(seed)
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionPipeline(device)

# Load prompts.
prompts = [
    # Background prompt.
    'clear deep blue sky',
    # Foreground prompts.
    'summer mountains',
    'the sun',
    'the moon',
    'a giant waterfall',
    'a giant waterfall',
    'clean deep blue lake',
    'a large tree',
    'a large tree',
]
negative_prompts = ['worst quality, bad quality, normal quality, cropped, framed'] * len(prompts)

# Load masks.
masks = []
for i in range(1, 9):
    url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/irworobongdo/irworobongdo_{i}.png'
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).convert('RGBA')
    mask = (T.ToTensor()(mask)[-1:] > 0.5).float()
    masks.append(mask)
# In this example, background is simply set as non-marked regions.
background = reduce(torch.logical_and, [m == 0 for m in masks])
masks = torch.stack([background] + masks, dim=0).float()

height, width = masks.shape[-2:] # (768, 1920) in this example.

# Sample an image.
image = smd(
    prompts,
    negative_prompts,
    masks=masks,
    mask_strengths=1,
    mask_stds=0,
    height=height,
    width=width,
    bootstrap_steps=2,
)
image.save('my_beautiful_creation.png')
```

---

### Image Inpainting with Prompt Separation

Our pipeline also enables editing and inpainting existing images.
We also support *any* SD 1.5 checkpoint models.
One exceptional advantage of ours is that we provide an easy separation of prompt
You can additionally trade-off between prompt separation and overall harmonization by changing the argument `bootstrap_steps` from 0 (full mixing) to 5 (full separation).
We recommend `1-3`.
The following code is a minimal example of performing prompt separated multi-prompt image inpainting using our pipeline on a custom model.

**Result:**

| ![mask](./assets/timessquare/timessquare.jpeg) | ![mask](./assets/timessquare/timessquare_full.png) | ![result](./assets/timessquare_inpainting.png) |
| :----------------------------: | :----------------------------: | :----------------------------: |
| Images to Inpaint | Semantic Brush Input | Inpainted Image (9 sec) |

**Code:**

```python
import torch
from model import StableMultiDiffusionPipeline
from util import seed_everything

# The following packages are imported only for loading the images.
import torchvision.transforms as T
import requests
from io import BytesIO
from PIL import Image


seed = 2
device = 0

# Load the module.
seed_everything(seed)
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionPipeline(
    device,
    hf_key='ironjr/BlazingDriveV11m',
)

# Load the background image you want to start drawing.
#   Although it works for any image, we recommend to use background that is generated
#   or at least modified by the same checkpoint model (e.g., preparing it by passing
#   it to the same checkpoint for an image-to-image pipeline with denoising_strength 0.2)
#   for the maximally harmonized results!
#   However, in this example, we choose to use a real-world image for the demo.
url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/timessquare/timessquare.jpeg'
response = requests.get(url)
background_image = Image.open(BytesIO(response.content)).convert('RGB')

# Load prompts and background prompts (explicitly).
background_prompt = '1girl, 1boy, times square'
prompts = [
    # Foreground prompts.
    '1boy, looking at viewer, brown hair, casual shirt',
    '1girl, looking at viewer, pink hair, leather jacket',
]
negative_prompts = [
    '1girl',
    '1boy',
]
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]
background_negative_prompt = negative_prompt_prefix

# Load masks.
masks = []
for i in range(1, 3):
    url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/timessquare/timessquare_{i}.png'
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).convert('RGBA')
    mask = (T.ToTensor()(mask)[-1:] > 0.5).float()
    masks.append(mask)
masks = torch.stack(masks, dim=0).float()
height, width = masks.shape[-2:] # (768, 768) in this example.

# Sample an image.
image = smd(
    prompts,
    negative_prompts,
    masks=masks,
    mask_strengths=1,
    # Use larger standard deviation to harmonize the inpainting result (Recommended: 8-32)!
    mask_stds=16.0,
    height=height,
    width=width,
    bootstrap_steps=2,
    bootstrap_leak_sensitivity=0.1,
    # This is for providing the image input.
    background=background_image,
    background_prompt=background_prompt,
    background_negative_prompt=background_negative_prompt,
)
image.save('my_beautiful_inpainting.png')
```

---

### Panorama Generation

Our [`model.StableMultiDiffusionPipeline`](https://github.com/ironjr/StreamMultiDiffusion/blob/main/src/model/stablemultidiffusion_pipeline.py) supports x10 faster generation of irregularly large size images such as panoramas.
For example, the following code runs in 10s with a single 2080 Ti GPU.

**Result:**

<p align="center">
  <img src="./assets/panorama_generation.png" width=100%>
</p>
<p align="center">
    512x3072 image generated in 10 seconds.
</p>

**Code:**

```python
import torch
from model import StableMultiDiffusionPipeline

device = 0

# Load the module.
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionPipeline(device)

# Sample a panorama image.
smd.sample_panorama('A photo of Alps', height=512, width=3072)
image.save('my_panorama_creation.png')
```

---

### Basic StableDiffusion

We also support standard single-prompt single-tile sampling of StableDiffusion checkpoint for completeness.
This behaves exactly the same as calling [`diffuser`](https://huggingface.co/docs/diffusers/en/index)'s [`StableDiffusionPipeline`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py).

**Result:**

<p align="left">
  <img src="./assets/dolomites_generation.png" width=50%>
</p>

**Code:**

```python
import torch
from model import StableMultiDiffusionPipeline

device = 0

# Load the module.
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionPipeline(device)

# Sample an image.
image = smd.sample('A photo of the dolomites')
image.save('my_creation.png')
```

---

### Demo Application (StreamMultiDiffusion)

<p align="center">
  <img src="./assets/demo_v2.gif" width=90%>
</p>

#### Features

- Drawing with _semantic palette_ with streaming interface.
- Fully web-based GUI, powered by Gradio.
- Supports any Stable Diffusion v1.5 checkpoint with option `--model`.
- Supports any-sized canvas (if your VRAM permits!) with opetion `--height`, `--width`.
- Supports 8 semantic brushes.

#### Run

```bash
cd src/demo/stream_v2
python app.py [other options]
```

#### Run with `.safetensors`

We now support `.safetensors` type local models.
You can run the demo app with your favorite checkpoint models as follows:
1. Save `<your model>.safetensors` or a [symbolic link](https://mangohost.net/blog/what-is-the-linux-equivalent-to-symbolic-links-in-windows/) to the actual file to `demo/stream/checkpoints`.
2. Run the demo with your model loaded with `python app.py --model <your model>.safetensors`

Done!

#### Other options

- `--model`: Optional. The path to your custom SDv1.5 checkpoint. Both Hugging Face model repository / local safetensor types are supported. e.g., `--model "KBlueLeaf/kohaku-v2.1"` or `--model "realcartoonPixar_v6.safetensors"` Please note that safetensors models should reside in `src/demo/stream/checkpoints`!
- `--height` (`-H`): Optional. Height of the canvas. Default: 768.
- `--width` (`-W`): Optional. Width of the canvas. Default: 1920.
- `--display_col`: Optional. Number of displays in a row. Useful for buffering the old frames. Default: 2.
- `--display_row`: Optional. Number of displays in a column. Useful for buffering the old frames. Default: 2.
- `--bootstrap_steps`: Optional. The number of bootstrapping steps that separate each of the different semantic regions. Best when 1-3. Larger value means better separation, but less harmony within the image. Default: 1.
- `--seed`: Optional. The default seed of the application. Almost never needed since you can modify the seed value in GUI. Default: 2024.
- `--device`: Optional. The number of GPU card (probably 0-7) you want to run the model. Only for multi-GPU servers. Default: 0.
- `--port`: Optional. The front-end port of the application. If the port is 8000, you can access your runtime through `https://localhost:8000` from any web browser. Default: 8000.


#### Instructions

| ![usage1](./assets/instruction1.png) | ![usage2](./assets/instruction2.png) |
| :----------------------------: | :----------------------------: |
| Upoad a background image | Type some text prompts |
| ![usage3](./assets/instruction3.png) | ![usage4](./assets/instruction4.png) |
| Draw | Press the play button and enjoy 🤩 |

1. (top-left) **Upload a background image.** You can start with a white background image, as well as any other images from your phone camera or other AI-generated artworks. You can also entirely cover the image editor with specific semantic brush to draw background image simultaneously from the text prompt.

2. (top-right) **Type some text prompts.** Click each semantic brush on the semantic palette on the left of the screen and type in text prompts in the interface below. This will create a new semantic brush for you.

3. (bottom-left) **Draw.** Select appropriate layer (*important*) that matches the order of the semantic palette. That is, ***Layer n*** corresponds to ***Prompt n***. I am not perfectly satisfied with the interface of the drawing interface. Importing professional Javascript-based online drawing tools instead of the default `gr.ImageEditor` will enable more responsive interface. We have released our code with MIT License, so please feel free to fork this repo and build a better user interface upon it. 😁

4. (bottom-right) **Press the play button and enjoy!** The buttons literally mean 'toggle stream/run single/run batch (4)'.

---

### Demo Application (Semantic Palette)

<div>

<p align="center">
  <img src="./assets/demo_semantic_draw_large.gif" width=90%>
</p>

</div>

Our first demo _[Semantic Palette](https://huggingface.co/spaces/ironjr/SemanticPalette)_ is now available in your local machine.

#### Features

- Fully web-based GUI, powered by Gradio.
- Supports any Stable Diffusion v1.5 checkpoint with option `--model`.
- Supports any-sized canvas (if your VRAM permits!) with opetion `--height`, `--width`.
- Supports 5 semantic brushes. If you want more brushes, you can use our python interface directly. Please see our Jupyter notebook references in the `notebooks` directory.

#### Run

```bash
cd src/demo/semantic_palette
python app.py [other options]
```

#### Run with `.safetensors`

We now support `.safetensors` type local models.
You can run the demo app with your favorite checkpoint models as follows:
1. Save `<your model>.safetensors` or a [symbolic link](https://mangohost.net/blog/what-is-the-linux-equivalent-to-symbolic-links-in-windows/) to the actual file to `demo/semantic_palette/checkpoints`.
2. Run the demo with your model loaded with `python app.py --model <your model>.safetensors`

Done!

#### Other options

- `--model`: Optional. The path to your custom SDv1.5 checkpoint. Both Hugging Face model repository / local safetensor types are supported. e.g., `--model "KBlueLeaf/kohaku-v2.1"` or `--model "realcartoonPixar_v6.safetensors"` Please note that safetensors models should reside in `src/demo/semantic_palette/checkpoints`!
- `--height` (`-H`): Optional. Height of the canvas. Default: 768.
- `--width` (`-W`): Optional. Width of the canvas. Default: 1920.
- `--bootstrap_steps`: Optional. The number of bootstrapping steps that separate each of the different semantic regions. Best when 1-3. Larger value means better separation, but less harmony within the image. Default: 1.
- `--seed`: Optional. The default seed of the application. Almost never needed since you can modify the seed value in GUI. Default: -1 (random).
- `--device`: Optional. The number of GPU card (probably 0-7) you want to run the model. Only for multi-GPU servers. Default: 0.
- `--port`: Optional. The front-end port of the application. If the port is 8000, you can access your runtime through `https://localhost:8000` from any web browser. Default: 8000.

#### Instructions

Instructions on how to use the app to create your images: Please see this twitter [thread](https://twitter.com/_ironjr_/status/1770245714591064066).

#### Tips

I have provided more tips in using the app in another twitter [thread](https://twitter.com/_ironjr_/status/1770716860948025539).

---

### Basic Usage (CLI)

Coming Soon!


---


## 💼 Further Information

We have provided detailed explanation of the application design and the expected usages in appendices of our [paper](https://arxiv.org/abs/2403.09055).
This section is a summary of its contents.
Although we expect everything to work fine, there may be unexpected bugs or missed features in the implementation.
We are always welcoming issues and pull requests from you to improve this project! 🤗


### User Interface (GUI)

<p align="center">
  <img src="./assets/user_interface.png" width=90%>
</p>

| No. | Component Name | Description |
| --- | -------------- | ----------- |
| 1 | *Semantic palette* | Creates and manages text prompt-mask pairs, a.k.a., _semantic brushes_. |
| 2 | Create new _semantic brush_ btn. | Creates a new text prompt-mask pair. |
| 3 | Main drawing pad | User draws at each semantic layers with a brush tool. |
| 4 | Layer selection | Each layer corresponds to each of the prompt mask in the *semantic palette*. |
| 5 | Background image upload | User uploads background image to start drawing. |
| 6 | Drawing tools | Using brushes and erasers to interactively edit the prompt masks. |
| 7 | Play button | Switches between streaming/step-by-step mode. |
| 8 | Display | Generated images are streamed through this component. |
| 9 | Mask alpha control | Changes the mask alpha value before quantization. Controls local content blending (simply means that you can use nonbinary masks for fine-grained controls), but extremely sensitive. Recommended: >0.95 |
| 10 | Mask blur std. dev. control | Changes the standard deviation of the quantized mask of the current semantic brush. Less sensitive than mask alpha control. |
| 11 | Seed control | Changes the seed of the application. May not be needed, since we generate infinite stream of images. |
| 12 | Prompt edit | User can interactively change the positive/negative prompts at need. |
| 13 | Prompt strength control | Prompt embedding mix ratio between the current & the background. Helps global content blending. Recommended: >0.75 |
| 14 | Brush name edit | Adds convenience by changing the name of the brush. Does not affect the generation. Just for preference. |

### Demo Application Architecture

There are two types of transaction data between the front-end and the back-end (`model.streammultidiffusion_pipeline.StreamMultiDiffusion`) of the application: a (1) background image object and a (2) list of text prompt-mask pairs.
We choose to call a pair of the latter as a _semantic brush_.
Despite its fancy name, a _semantic brush_ is just a pair of a text prompt and a regional mask assigned to the prompt, possibly with additional mask-controlling parameters.
Users interact with the application by registering and updating these two types of data to control the image generation stream.
The interface is summarized in the image below ⬇️:

<p align="center">
  <img src="./assets/app_design.png" width=90%>
</p>


---

## 🙋 FAQ

### What is _Semantic Palette_ Anyway?

**Semantic palette** basically means that you paint things with semantics, i.e., text prompts, just like how you may use brush tools in commercial image editing software, such as Adobe Photoshop, etc.
Our acceleration technique for region-based controlled image generation allows users to edit their prompt masks similarly to drawing.
We couldn't find a good preexisting name for this type of user interface, so we named it as _semantic palette_, hoping for it to make sense to you. 😄

### Can it run realistic models / anime-style models?

Of course. Both types of models are supported.
For realistic models and SDXL-type models, using `--bootstrap_steps=2` or `3` produces better (non-cropped) images.

https://github.com/ironjr/StreamMultiDiffusion/assets/12259041/9a6bb02b-7dca-4dd0-a1bc-6153dde1571d


## 🌏 Citation

Please cite us if you find our project useful!

```latex
@article{lee2024streammultidiffusion,
    title={{StreamMultiDiffusion:} Real-Time Interactive Generation with Region-Based Semantic Control},
    author={Lee, Jaerin and Jung, Daniel Sungho and Lee, Kanggeon and Lee, Kyoung Mu},
    journal={arXiv preprint arXiv:2403.09055},
    year={2024}
}
```

## 🚨 Notice

Please note that we do not host separate pages for this project other than our [official project page](https://jaerinlee.com/research/streammultidiffusion) and [hugging face](https://huggingface.co/spaces/ironjr/SemanticPalette) [space demos](https://huggingface.co/spaces/ironjr/SemanticPaletteXL).
For example [https://streammultidiffusion.net](https://streammultidiffusion.net/) is not related to us!
(By the way thanks for hosting)

We do welcome anyone who wants to use our framework/code/app for any personal or commercial purpose (we have opened the code here for free with MIT License).
However, **we'd be much happier if you cite us in any format in your application**.
We are very open to discussion, so if you find any issue with this project, including commercialization of the project, please contact [us](https://twitter.com/_ironjr_) or post an issue.


## 🤗 Acknowledgement

Our code is based on the projects: [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion), [MultiDiffusion](https://multidiffusion.github.io/), and [Latent Consistency Model](https://latent-consistency-models.github.io/). Thank you for sharing such amazing works!
We also give our huge thanks to [@br_d](https://twitter.com/br_d) and [@KBlueleaf](https://twitter.com/KBlueleaf) for the wonderful models [BlazingDriveV11m](https://civitai.com/models/121083?modelVersionId=236210) and [Kohaku V2](https://civitai.com/models/136268/kohaku-v2)!


## 📧 Contact

If you have any questions, please email `jarin.lee@gmail.com`.
