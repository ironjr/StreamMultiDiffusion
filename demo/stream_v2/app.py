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

import sys

sys.path.append('../../src')

import argparse
import random
import time
import json
import os
import glob
import pathlib
from functools import partial
from pprint import pprint

import numpy as np
from PIL import Image
import torch

import gradio as gr
from huggingface_hub import snapshot_download

# from model import StreamMultiDiffusionSDXL
from model import StreamMultiDiffusion
from util import seed_everything
from prompt_util import preprocess_prompts, _quality_dict, _style_dict


### Utils




def log_state(state):
    pprint(vars(opt))
    if isinstance(state, gr.State):
        state = state.value
    pprint(vars(state))


def is_empty_image(im: Image.Image) -> bool:
    if im is None:
        return True
    im = np.array(im)
    has_alpha = (im.shape[2] == 4)
    if not has_alpha:
        return False
    elif im.sum() == 0:
        return True
    else:
        return False


### Argument passing

# parser = argparse.ArgumentParser(description='Semantic Palette demo powered by StreamMultiDiffusion with SDXL support.')
# parser.add_argument('-H', '--height', type=int, default=1024)
# parser.add_argument('-W', '--width', type=int, default=1024)
parser = argparse.ArgumentParser(description='Semantic Palette demo powered by StreamMultiDiffusion.')
parser.add_argument('-H', '--height', type=int, default=768)
parser.add_argument('-W', '--width', type=int, default=768)
parser.add_argument('--model', type=str, default=None, help='Hugging face model repository or local path for a SD1.5 model checkpoint to run.')
parser.add_argument('--bootstrap_steps', type=int, default=1)
parser.add_argument('--guidance_scale', type=float, default=0) # 1.2
parser.add_argument('--run_time', type=float, default=60)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--port', type=int, default=8000)
opt = parser.parse_args()


### Global variables and data structures

device = f'cuda:{opt.device}' if opt.device >= 0 else 'cpu'


if opt.model is None:
    # opt.model = 'cagliostrolab/animagine-xl-3.1'
    # opt.model = 'ironjr/BlazingDriveV11m'
    opt.model = 'KBlueLeaf/kohaku-v2.1'
else:
    if opt.model.endswith('.safetensors'):
        opt.model = os.path.abspath(os.path.join('checkpoints', opt.model))

# model = StreamMultiDiffusionSDXL(
model = StreamMultiDiffusion(
    device,
    hf_key=opt.model,
    height=opt.height,
    width=opt.width,
    cfg_type="full",
    autoflush=True,
    use_tiny_vae=True,
    mask_type='continuous',
    bootstrap_steps=opt.bootstrap_steps,
    bootstrap_mix_steps=opt.bootstrap_steps,
    guidance_scale=opt.guidance_scale,
    seed=opt.seed,
)


prompt_suggestions = [
    '1girl, souryuu asuka langley, neon genesis evangelion, solo, upper body, v, smile, looking at viewer',
    '1boy, solo, portrait, looking at viewer, white t-shirt, brown hair',
    '1girl, arima kana, oshi no ko, solo, upper body, from behind',
]

opt.max_palettes = 3
opt.default_prompt_strength = 1.0
opt.default_mask_strength = 1.0
opt.default_mask_std = 8.0
opt.default_negative_prompt = (
    'nsfw, worst quality, bad quality, normal quality, cropped, framed'
)
opt.verbose = True
opt.colors = [
    '#000000',
    '#2692F3',
    '#F89E12',
    '#16C232',
    # '#F92F6C',
    # '#AC6AEB',
    # '#92C62C',
    # '#92C6EC',
    # '#FECAC0',
]
opt.excluded_keys = ['inpainting_mode', 'is_running', 'active_palettes', 'current_palette', 'model']
opt.prep_time = 20


### Event handlers

def add_palette(state):
    old_actives = state.active_palettes
    state.active_palettes = min(state.active_palettes + 1, opt.max_palettes)

    if opt.verbose:
        log_state(state)

    if state.active_palettes != old_actives:
        return [state] + [
            gr.update() if state.active_palettes != opt.max_palettes else gr.update(visible=False)
        ] + [
            gr.update() if i != state.active_palettes - 1 else gr.update(value=state.prompt_names[i + 1], visible=True)
            for i in range(opt.max_palettes)
        ]
    else:
        return [state] + [gr.update() for i in range(opt.max_palettes + 1)]


def select_palette(state, button, idx):
    if idx < 0 or idx > opt.max_palettes:
        idx = 0
    old_idx = state.current_palette
    if old_idx == idx:
        return [state] + [gr.update() for _ in range(opt.max_palettes + 7)]

    state.current_palette = idx

    if opt.verbose:
        log_state(state)

    updates = [state] + [
        gr.update() if i not in (idx, old_idx) else
        gr.update(variant='secondary') if i == old_idx else gr.update(variant='primary')
        for i in range(opt.max_palettes + 1)
    ]
    label = 'Background' if idx == 0 else f'Palette {idx}'
    updates.extend([
        gr.update(value=button, interactive=(idx > 0)),
        gr.update(value=state.prompts[idx], label=f'Edit Prompt for {label}'),
        gr.update(value=state.neg_prompts[idx], label=f'Edit Negative Prompt for {label}'),
        (
            gr.update(value=state.mask_strengths[idx - 1], interactive=True) if idx > 0 else
            gr.update(value=opt.default_mask_strength, interactive=False)
        ),
        (
            gr.update(value=state.prompt_strengths[idx - 1], interactive=True) if idx > 0 else
            gr.update(value=opt.default_prompt_strength, interactive=False)
        ),
        (
            gr.update(value=state.mask_stds[idx - 1], interactive=True) if idx > 0 else
            gr.update(value=opt.default_mask_std, interactive=False)
        ),
    ])
    return updates


def change_prompt_strength(state, strength):
    if state.current_palette == 0:
        return state

    state.prompt_strengths[state.current_palette - 1] = strength
    if opt.verbose:
        log_state(state)

    return state


def change_std(state, std):
    if state.current_palette == 0:
        return state

    state.mask_stds[state.current_palette - 1] = std
    if opt.verbose:
        log_state(state)

    return state


def change_mask_strength(state, strength):
    if state.current_palette == 0:
        return state

    state.mask_strengths[state.current_palette - 1] = strength
    if opt.verbose:
        log_state(state)

    return state


def reset_seed(state, seed):
    state.seed = seed
    if opt.verbose:
        log_state(state)

    return state


def rename_prompt(state, name):
    state.prompt_names[state.current_palette] = name
    if opt.verbose:
        log_state(state)

    return [state] + [
        gr.update() if i != state.current_palette else gr.update(value=name)
        for i in range(opt.max_palettes + 1)
    ]


def change_prompt(state, prompt):
    state.prompts[state.current_palette] = prompt
    if opt.verbose:
        log_state(state)

    return state


def change_neg_prompt(state, neg_prompt):
    state.neg_prompts[state.current_palette] = neg_prompt
    if opt.verbose:
        log_state(state)

    return state


# def select_style(state, style_name):
#     state.style_name = style_name
#     if opt.verbose:
#         log_state(state)

#     return state


# def select_quality(state, quality_name):
#     state.quality_name = quality_name
#     if opt.verbose:
#         log_state(state)

#     return state


def import_state(state, json_text):
    prev_state_dict = {k: v for k, v in vars(state).items() if k in opt.excluded_keys}
    state_dict = json.loads(json_text)
    for k in opt.excluded_keys:
        if k in state_dict:
            del state_dict[k]
    state_dict.update(prev_state_dict)
    state = argparse.Namespace(**state_dict)

    current_palette = state.current_palette
    state.active_palettes = opt.max_palettes
    return [state] + [
        gr.update(value=v, visible=True) for v in state.prompt_names
    ] + [
        # state.style_name,
        # state.quality_name,
        state.prompts[current_palette],
        state.prompt_names[current_palette],
        state.neg_prompts[current_palette],
        state.prompt_strengths[current_palette - 1],
        state.mask_strengths[current_palette - 1],
        state.mask_stds[current_palette - 1],
        state.seed,
    ]


### Main worker

def generate():
    return model()


def register(state, drawpad):
    seed_everything(state.seed if state.seed >=0 else np.random.randint(2147483647))
    print('Generate!')

    background = drawpad['background'].convert('RGBA')
    inpainting_mode = np.asarray(background).sum() != 0
    if not inpainting_mode:
        background = Image.new(size=(opt.width, opt.height), mode='RGB', color=(255, 255, 255))
    print('Inpainting mode: ', inpainting_mode)

    user_input = np.asarray(drawpad['layers'][0]) # (H, W, 4)
    foreground_mask = torch.tensor(user_input[..., -1])[None, None] # (1, 1, H, W)
    user_input = torch.tensor(user_input[..., :-1]) # (H, W, 3)

    palette = torch.tensor([
        tuple(int(s[i+1:i+3], 16) for i in (0, 2, 4))
        for s in opt.colors[1:]
    ]) # (N, 3)
    masks = (palette[:, None, None, :] == user_input[None]).all(dim=-1)[:, None, ...] # (N, 1, H, W)
    # has_masks = [i for i, m in enumerate(masks.sum(dim=(1, 2, 3)) == 0) if not m]
    has_masks = list(range(opt.max_palettes))
    print('Has mask: ', has_masks)
    masks = masks * foreground_mask
    masks = masks[has_masks]

    # if inpainting_mode:
    prompts = [state.prompts[v + 1] for v in has_masks]
    negative_prompts = [state.neg_prompts[v + 1] for v in has_masks]
    mask_strengths = [state.mask_strengths[v] for v in has_masks]
    mask_stds = [state.mask_stds[v] for v in has_masks]
    prompt_strengths = [state.prompt_strengths[v] for v in has_masks]
    # else:
        # masks = torch.cat([torch.ones_like(foreground_mask), masks], dim=0)
        # prompts = [state.prompts[0]] + [state.prompts[v + 1] for v in has_masks]
        # negative_prompts = [state.neg_prompts[0]] + [state.neg_prompts[v + 1] for v in has_masks]
        # mask_strengths = [1] + [state.mask_strengths[v] for v in has_masks]
        # mask_stds = [0] + [state.mask_stds[v] for v in has_masks]
        # prompt_strengths = [1] + [state.prompt_strengths[v] for v in has_masks]

    # prompts, negative_prompts = preprocess_prompts(
    #     prompts, negative_prompts, style_name=state.style_name, quality_name=state.quality_name)

    model.update_background(
        background.convert('RGB'),
        prompt=None,
        negative_prompt=None,
    )
    state.prompts[0] = model.background.prompt
    state.neg_prompts[0] = model.background.negative_prompt

    model.update_layers(
        prompts=prompts,
        negative_prompts=negative_prompts,
        masks=masks.to(device),
        mask_strengths=mask_strengths,
        mask_stds=mask_stds,
        prompt_strengths=prompt_strengths,
    )

    state.inpainting_mode = inpainting_mode
    return state


def run(state, drawpad):
    state = register(state, drawpad)
    state.is_running = True

    tic = time.time()
    while True:
        yield [state, generate()]
        toc = time.time()
        tdelta = toc - tic
        if tdelta > opt.run_time:
            state.is_running = False
            return [state, generate()]


def hide_element():
    return gr.update(visible=False)


def show_element():
    return gr.update(visible=True)


def draw(state, drawpad):
    if not state.is_running:
        return

    user_input = np.asarray(drawpad['layers'][0]) # (H, W, 4)
    foreground_mask = torch.tensor(user_input[..., -1])[None, None] # (1, 1, H, W)
    user_input = torch.tensor(user_input[..., :-1]) # (H, W, 3)

    palette = torch.tensor([
        tuple(int(s[i+1:i+3], 16) for i in (0, 2, 4))
        for s in opt.colors[1:]
    ]) # (N, 3)
    masks = (palette[:, None, None, :] == user_input[None]).all(dim=-1)[:, None, ...] # (N, 1, H, W)
    # has_masks = [i for i, m in enumerate(masks.sum(dim=(1, 2, 3)) == 0) if not m]
    has_masks = list(range(opt.max_palettes))
    print('Has mask: ', has_masks)
    masks = masks * foreground_mask
    masks = masks[has_masks]

    # if state.inpainting_mode:
    mask_strengths = [state.mask_strengths[v] for v in has_masks]
    mask_stds = [state.mask_stds[v] for v in has_masks]
    # else:
    #     masks = torch.cat([torch.ones_like(foreground_mask), masks], dim=0)
    #     mask_strengths = [1] + [state.mask_strengths[v] for v in has_masks]
    #     mask_stds = [0] + [state.mask_stds[v] for v in has_masks]

    for i in range(len(has_masks)):
        model.update_single_layer(
            idx=i,
            mask=masks[i:i + 1],
            mask_strength=mask_strengths[i],
            mask_std=mask_stds[i],
        )

### Load examples


root = pathlib.Path(__file__).parent
print(root)
example_root = os.path.join(root, 'examples')
example_images = glob.glob(os.path.join(example_root, '*.png'))
example_images = [Image.open(i) for i in example_images]

# with open(os.path.join(example_root, 'prompt_background_advanced.txt')) as f:
#     prompts_background = [l.strip() for l in f.readlines() if l.strip() != '']

# with open(os.path.join(example_root, 'prompt_girl.txt')) as f:
#     prompts_girl = [l.strip() for l in f.readlines() if l.strip() != '']

# with open(os.path.join(example_root, 'prompt_boy.txt')) as f:
#     prompts_boy = [l.strip() for l in f.readlines() if l.strip() != '']

# with open(os.path.join(example_root, 'prompt_props.txt')) as f:
#     prompts_props = [l.strip() for l in f.readlines() if l.strip() != '']
#     prompts_props = {l.split(',')[0].strip(): ','.join(l.split(',')[1:]).strip() for l in prompts_props}

# prompt_background = lambda: random.choice(prompts_background)
# prompt_girl = lambda: random.choice(prompts_girl)
# prompt_boy = lambda: random.choice(prompts_boy)
# prompt_props = lambda: np.random.choice(list(prompts_props.keys()), size=(opt.max_palettes - 2), replace=False).tolist()


### Main application

css = f"""
#run-button {{
    font-size: 18pt;
    background-image: linear-gradient(to right, #4338ca 0%, #26a0da 51%, #4338ca 100%);
    margin: 0;
    padding: 15px 45px;
    text-align: center;
//    text-transform: uppercase;
    transition: 0.5s;
    background-size: 200% auto;
    color: white;
    box-shadow: 0 0 20px #eee;
    border-radius: 10px;
//    display: block;
    background-position: right center;
}}

#run-button:hover {{
    background-position: left center;
    color: #fff;
    text-decoration: none;
}}

#run-anim {{
    padding: 40px 45px;
}}

#semantic-palette {{
    border-style: solid;
    border-width: 0.2em;
    border-color: #eee;
}}

#semantic-palette:hover {{
    box-shadow: 0 0 20px #eee;
}}

#output-screen {{
    width: 100%;
    aspect-ratio: {opt.width} / {opt.height};
}}
"""

for i in range(opt.max_palettes + 1):
    css = css + f"""
.secondary#semantic-palette-{i} {{
    background-image: linear-gradient(to right, #374151 0%, #374151 71%, {opt.colors[i]} 100%);
    color: white;
}}

.primary#semantic-palette-{i} {{
    background-image: linear-gradient(to right, #4338ca 0%, #4338ca 71%, {opt.colors[i]} 100%);
    color: white;
}}
"""

css = css + f"""
.mask-red {{
    left: 0;
    width: 0;
    color: #BE002A;
    -webkit-animation: text-red {opt.run_time + opt.prep_time:.1f}s ease infinite;
            animation: text-red {opt.run_time + opt.prep_time:.1f}s ease infinite;
    z-index: 2;
    background: transparent;
}}
.mask-white {{
    right: 0;
}}
/* Flames */
#red-flame {{
    opacity: 0;
    -webkit-animation: show-flames {opt.run_time + opt.prep_time:.1f}s ease infinite, red-flame 120ms ease infinite;
            animation: show-flames {opt.run_time + opt.prep_time:.1f}s ease infinite, red-flame 120ms ease infinite;
    transform-origin: center bottom;
}}
#yellow-flame {{
    opacity: 0;
    -webkit-animation: show-flames {opt.run_time + opt.prep_time:.1f}s ease infinite, yellow-flame 120ms ease infinite;
            animation: show-flames {opt.run_time + opt.prep_time:.1f}s ease infinite, yellow-flame 120ms ease infinite;
    transform-origin: center bottom;
}}
#white-flame {{
    opacity: 0;
    -webkit-animation: show-flames {opt.run_time + opt.prep_time:.1f}s ease infinite, red-flame 100ms ease infinite;
            animation: show-flames {opt.run_time + opt.prep_time:.1f}s ease infinite, red-flame 100ms ease infinite;
    transform-origin: center bottom;
}}
"""

with open(os.path.join(root, 'timer', 'style.css')) as f:
    added_css = ''.join(f.readlines())
css = css + added_css

# js = ''

# with open(os.path.join(root, 'timer', 'script.js')) as f:
#     added_js = ''.join(f.readlines())
# js = js + added_js

head = f"""
<link href='https://fonts.googleapis.com/css?family=Oswald' rel='stylesheet' type='text/css'>
<script src='https://code.jquery.com/jquery-2.2.4.min.js'></script>
"""


with gr.Blocks(theme=gr.themes.Soft(), css=css, head=head) as demo:

    iface = argparse.Namespace()

    def _define_state():
        state = argparse.Namespace()

        # Cursor.
        state.is_running = False
        state.inpainting_mode = False
        state.current_palette = 0 # 0: Background; 1,2,3,...: Layers
        state.model_id = opt.model
        state.style_name = '(None)'
        state.quality_name = 'Standard v3.1'

        # State variables (one-hot).
        state.active_palettes = 5

        # Front-end initialized to the default values.
        # prompt_props_ = prompt_props()
        state.prompt_names = [
            '🌄 Background',
            '👧 Girl',
            '🐶 Dog',
            '💐 Garden',
        ] + ['🎨 New Palette' for _ in range(opt.max_palettes - 3)]
        state.prompts = [
            '',
           'A girl smiling at viewer',
            'Doggy body part',
            'Flower garden',
        ] + ['' for _ in range(opt.max_palettes - 3)]
        state.neg_prompts = [
            opt.default_negative_prompt
            + (', humans, humans, humans' if i == 0 else '')
            for i in range(opt.max_palettes + 1)
        ]
        state.prompt_strengths = [opt.default_prompt_strength for _ in range(opt.max_palettes)]
        state.mask_strengths = [opt.default_mask_strength for _ in range(opt.max_palettes)]
        state.mask_stds = [opt.default_mask_std for _ in range(opt.max_palettes)]
        state.seed = opt.seed
        return state

    state = gr.State(value=_define_state)


    ### Demo user interface

    gr.HTML(
        """
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
    <div>
        <h1>🦦🦦 StreamMultiDiffusion: Real-Time Interactive Generation with Region-Based Semantic Control 🦦🦦</h1>
        <h5 style="margin: 0;">If you ❤️ our project, please visit our Github and give us a 🌟!</h5>
        </br>
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <a href='https://jaerinlee.com/research/StreamMultiDiffusion'>
                <img src='https://img.shields.io/badge/Project-Page-green' alt='Project Page'>
            </a>
            &nbsp;
            <a href='https://arxiv.org/abs/2403.09055'>
                <img src="https://img.shields.io/badge/arXiv-2403.09055-red">
            </a>
            &nbsp;
            <a href='https://github.com/ironjr/StreamMultiDiffusion'>
                <img src='https://img.shields.io/github/stars/ironjr/StreamMultiDiffusion?label=Github&color=blue'>
            </a>
            &nbsp;
            <a href='https://twitter.com/_ironjr_'>
                <img src='https://img.shields.io/twitter/url?label=_ironjr_&url=https%3A%2F%2Ftwitter.com%2F_ironjr_'>
            </a>
            &nbsp;
            <a href='https://github.com/ironjr/StreamMultiDiffusion/blob/main/LICENSE'>
                <img src='https://img.shields.io/badge/license-MIT-lightgrey'>
            </a>
            &nbsp;
            <a href='https://huggingface.co/spaces/ironjr/StreamMultiDiffusion'>
                <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Demo-StreamMultiDiffusion-yellow'>
            </a>
            &nbsp;
            <a href='https://huggingface.co/spaces/ironjr/SemanticPalette'>
                <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Demo-SemanticPaletteSD1.5-yellow'>
            </a>
            &nbsp;
            <a href='https://huggingface.co/spaces/ironjr/SemanticPaletteXL'>
                <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Demo-SemanticPaletteSDXL-yellow'>
            </a>
            &nbsp;
            <a href='https://colab.research.google.com/github/camenduru/SemanticPalette-jupyter/blob/main/SemanticPalette_jupyter.ipynb'>
                <img src='https://colab.research.google.com/assets/colab-badge.svg'>
            </a>
        </div>
    </div>
</div>
<div>
    </br>
</div>
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            with gr.Group(elem_id='semantic-palette'):

                gr.HTML(
                    """
<div style="justify-content: center; align-items: center;">
    <br/>
    <h3 style="margin: 0; text-align: center;"><b>🧠 Semantic Palette 🎨</b></h3>
    <br/>
</div>
                    """
                )

                iface.btn_semantics = [gr.Button(
                    value=state.value.prompt_names[0],
                    variant='primary',
                    elem_id='semantic-palette-0',
                )]
                for i in range(opt.max_palettes):
                    iface.btn_semantics.append(gr.Button(
                        value=state.value.prompt_names[i + 1],
                        variant='secondary',
                        visible=(i < state.value.active_palettes),
                        elem_id=f'semantic-palette-{i + 1}'
                    ))

                iface.btn_add_palette = gr.Button(
                    value='Create New Semantic Brush',
                    variant='primary',
                    visible=(state.value.active_palettes < opt.max_palettes),
                )

            with gr.Accordion(label='Import/Export Semantic Palette', open=True):
                iface.tbox_state_import = gr.Textbox(label='Put Palette JSON Here To Import')
                iface.json_state_export = gr.JSON(label='Exported Palette')
                iface.btn_export_state = gr.Button("Export Palette ➡️ JSON", variant='primary')
                iface.btn_import_state = gr.Button("Import JSON ➡️ Palette", variant='secondary')

            gr.HTML(
                """
<div>
</br>
</div>
<div style="justify-content: center; align-items: center;">
<h3 style="margin: 0; text-align: center;"><b>❓Usage❓</b></h3>
</br>
<div style="justify-content: center; align-items: left; text-align: left;">
    <p>1. (Optional) Uploading a background image. No background image means white background. It is <b>not mandatory</b>!</p>
    <p>2. Modify semantic palette (prompt & settings) as you want by clicking <b>Semantic Palette</b>. Export, import, and share semantic palette for fast configuration.</p>
    <p>3. Start drawing in the <b>Semantic Drawpad</b> tab. The brush <b>color</b>, not layer, is directly linked to the semantic brushes.</p>
    <p>4. Click [<b>Lemmy try!</b>] button to grant 1 minute of streaming demo.</p>
    <p>5. Continue drawing until your quota is over!</p>
</div>
</div>
                """
            )

            gr.HTML(
                """
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
<h5 style="margin: 0;"><b>... or run in your own 🤗 space!</b></h5>
</div>
                """
            )

            gr.DuplicateButton()

        with gr.Column(scale=4):

            with gr.Row():

                with gr.Column(scale=2):

                    iface.ctrl_semantic = gr.ImageEditor(
                        image_mode='RGBA',
                        sources=['upload', 'clipboard', 'webcam'],
                        transforms=['crop'],
                        crop_size=(opt.width, opt.height),
                        brush=gr.Brush(
                            colors=opt.colors[1:],
                            color_mode="fixed",
                        ),
                        type='pil',
                        label='Semantic Drawpad',
                        elem_id='drawpad',
                        layers=False,
                    )

#                     with gr.Accordion(label='Prompt Engineering', open=False):
#                         iface.quality_select = gr.Dropdown(
#                             label='Quality Presets',
#                             interactive=True,
#                             choices=list(_quality_dict.keys()),
#                             value='Standard v3.1',
#                         )

#                         iface.style_select = gr.Radio(
#                             label='Style Preset',
#                             container=True,
#                             interactive=True,
#                             choices=list(_style_dict.keys()),
#                             value='(None)',
#                         )

                with gr.Column(scale=2):

                    iface.image_slot = gr.Image(
                        interactive=False,
                        show_label=False,
                        show_download_button=True,
                        type='pil',
                        label='Generated Result',
                        elem_id='output-screen',
                        value=lambda: random.choice(example_images),
                    )

                    iface.btn_generate = gr.Button(
                        value=f'Lemme try! ({int(opt.run_time // 60)} min)',
                        variant='primary',
                        # scale=1,
                        elem_id='run-button'
                    )

                    iface.run_animation = gr.HTML(
                        f"""
<div id="deadline">
  <svg preserveAspectRatio="none" id="line" viewBox="0 0 581 158" enable-background="new 0 0 581 158">
    <g id="fire">
      <rect id="mask-fire-black" x="511" y="41" width="38" height="34"/>
      <g>
        <defs>
          <rect id="mask_fire" x="511" y="41" width="38" height="34"/>
        </defs>
        <clipPath id="mask-fire_1_">
          <use xlink:href="#mask_fire"  overflow="visible"/>
        </clipPath>
        <g id="group-fire" clip-path="url(#mask-fire_1_)">
          <path id="red-flame" fill="#B71342" d="M528.377,100.291c6.207,0,10.947-3.272,10.834-8.576 c-0.112-5.305-2.934-8.803-8.237-10.383c-5.306-1.581-3.838-7.9-0.79-9.707c-7.337,2.032-7.581,5.891-7.11,8.238 c0.789,3.951,7.56,4.402,5.077,9.48c-2.482,5.079-8.012,1.129-6.319-2.257c-2.843,2.233-4.78,6.681-2.259,9.703 C521.256,98.809,524.175,100.291,528.377,100.291z"/>
          <path id="yellow-flame" opacity="0.71" fill="#F7B523" d="M528.837,100.291c4.197,0,5.108-1.854,5.974-5.417 c0.902-3.724-1.129-6.207-5.305-9.931c-2.396-2.137-1.581-4.176-0.565-6.32c-4.401,1.918-3.384,5.304-2.482,6.658 c1.511,2.267,2.099,2.364,0.42,5.8c-1.679,3.435-5.42,0.764-4.275-1.527c-1.921,1.512-2.373,4.04-1.528,6.563 C522.057,99.051,525.994,100.291,528.837,100.291z"/>
          <path id="white-flame" opacity="0.81" fill="#FFFFFF" d="M529.461,100.291c-2.364,0-4.174-1.322-4.129-3.469 c0.04-2.145,1.117-3.56,3.141-4.198c2.022-0.638,1.463-3.195,0.302-3.925c2.798,0.821,2.89,2.382,2.711,3.332 c-0.301,1.597-2.883,1.779-1.938,3.834c0.912,1.975,3.286,0.938,2.409-0.913c1.086,0.903,1.826,2.701,0.864,3.924 C532.18,99.691,531.064,100.291,529.461,100.291z"/>
        </g>
      </g>
    </g>
    <g id="progress-trail">
      <path fill="#FFFFFF" d="M491.979,83.878c1.215-0.73-0.62-5.404-3.229-11.044c-2.583-5.584-5.034-10.066-7.229-8.878
                              c-2.854,1.544-0.192,6.286,2.979,11.628C487.667,80.917,490.667,84.667,491.979,83.878z"/>
      <path fill="#FFFFFF" d="M571,76v-5h-23.608c0.476-9.951-4.642-13.25-4.642-13.25l-3.125,4c0,0,3.726,2.7,3.625,5.125
                              c-0.071,1.714-2.711,3.18-4.962,4.125H517v5h10v24h-25v-5.666c0,0,0.839,0,2.839-0.667s6.172-3.667,4.005-6.333
                              s-7.49,0.333-9.656,0.166s-6.479-1.5-8.146,1.917c-1.551,3.178,0.791,5.25,5.541,6.083l-0.065,4.5H16c-2.761,0-5,2.238-5,5v17
                              c0,2.762,2.239,5,5,5h549c2.762,0,5-2.238,5-5v-17c0-2.762-2.238-5-5-5h-3V76H571z"/>
      <path fill="#FFFFFF" d="M535,65.625c1.125,0.625,2.25-1.125,2.25-1.125l11.625-22.375c0,0,0.75-0.875-1.75-2.125
                              s-3.375,0.25-3.375,0.25s-8.75,21.625-9.875,23.5S533.875,65,535,65.625z"/>
    </g>
    <g>
      <defs>
        <path id="SVGID_1_" d="M484.5,75.584c-3.172-5.342-5.833-10.084-2.979-11.628c2.195-1.188,4.646,3.294,7.229,8.878
                               c2.609,5.64,4.444,10.313,3.229,11.044C490.667,84.667,487.667,80.917,484.5,75.584z M571,76v-5h-23.608
                               c0.476-9.951-4.642-13.25-4.642-13.25l-3.125,4c0,0,3.726,2.7,3.625,5.125c-0.071,1.714-2.711,3.18-4.962,4.125H517v5h10v24h-25
                               v-5.666c0,0,0.839,0,2.839-0.667s6.172-3.667,4.005-6.333s-7.49,0.333-9.656,0.166s-6.479-1.5-8.146,1.917
                               c-1.551,3.178,0.791,5.25,5.541,6.083l-0.065,4.5H16c-2.761,0-5,2.238-5,5v17c0,2.762,2.239,5,5,5h549c2.762,0,5-2.238,5-5v-17
                               c0-2.762-2.238-5-5-5h-3V76H571z M535,65.625c1.125,0.625,2.25-1.125,2.25-1.125l11.625-22.375c0,0,0.75-0.875-1.75-2.125
                               s-3.375,0.25-3.375,0.25s-8.75,21.625-9.875,23.5S533.875,65,535,65.625z"/>
      </defs>
      <clipPath id="SVGID_2_">
        <use xlink:href="#SVGID_1_"  overflow="visible"/>
      </clipPath>
      <rect id="progress-time-fill" x="-100%" y="34" clip-path="url(#SVGID_2_)" fill="#BE002A" width="586" height="103"/>
    </g>

    <g id="death-group">
      <path id="death" fill="#BE002A" d="M-46.25,40.416c-5.42-0.281-8.349,3.17-13.25,3.918c-5.716,0.871-10.583-0.918-10.583-0.918
                                         C-67.5,49-65.175,50.6-62.083,52c5.333,2.416,4.083,3.5,2.084,4.5c-16.5,4.833-15.417,27.917-15.417,27.917L-75.5,84.75
                                         c-1,12.25-20.25,18.75-20.25,18.75s39.447,13.471,46.25-4.25c3.583-9.333-1.553-16.869-1.667-22.75
                                         c-0.076-3.871,2.842-8.529,6.084-12.334c3.596-4.22,6.958-10.374,6.958-15.416C-38.125,43.186-39.833,40.75-46.25,40.416z
                                         M-40,51.959c-0.882,3.004-2.779,6.906-4.154,6.537s-0.939-4.32,0.112-7.704c0.82-2.64,2.672-5.96,3.959-5.583
                                         C-39.005,45.523-39.073,48.8-40,51.959z"/>
      <path id="death-arm" fill="#BE002A" d="M-53.375,75.25c0,0,9.375,2.25,11.25,0.25s2.313-2.342,3.375-2.791
                                             c1.083-0.459,4.375-1.75,4.292-4.75c-0.101-3.627,0.271-4.594,1.333-5.043c1.083-0.457,2.75-1.666,2.75-1.666
                                             s0.708-0.291,0.5-0.875s-0.791-2.125-1.583-2.959c-0.792-0.832-2.375-1.874-2.917-1.332c-0.542,0.541-7.875,7.166-7.875,7.166
                                             s-2.667,2.791-3.417,0.125S-49.833,61-49.833,61s-3.417,1.416-3.417,1.541s-1.25,5.834-1.25,5.834l-0.583,5.833L-53.375,75.25z"/>
      <path id="death-tool" fill="#BE002A" d="M-20.996,26.839l-42.819,91.475l1.812,0.848l38.342-81.909c0,0,8.833,2.643,12.412,7.414
                                              c5,6.668,4.75,14.084,4.75,14.084s4.354-7.732,0.083-17.666C-10,32.75-19.647,28.676-19.647,28.676l0.463-0.988L-20.996,26.839z"/>
    </g>
    <path id="designer-body" fill="#FEFFFE" d="M514.75,100.334c0,0,1.25-16.834-6.75-16.5c-5.501,0.229-5.583,3-10.833,1.666
                                               c-3.251-0.826-5.084-15.75-0.834-22c4.948-7.277,12.086-9.266,13.334-7.833c2.25,2.583-2,10.833-4.5,14.167
                                               c-2.5,3.333-1.833,10.416,0.5,9.916s8.026-0.141,10,2.25c3.166,3.834,4.916,17.667,4.916,17.667l0.917,2.5l-4,0.167L514.75,100.334z
                                               "/>

    <circle id="designer-head" fill="#FEFFFE" cx="516.083" cy="53.25" r="6.083"/>

    <g id="designer-arm-grop">
      <path id="designer-arm" fill="#FEFFFE" d="M505.875,64.875c0,0,5.875,7.5,13.042,6.791c6.419-0.635,11.833-2.791,13.458-4.041s2-3.5,0.25-3.875
                                                s-11.375,5.125-16,3.25c-5.963-2.418-8.25-7.625-8.25-7.625l-2,1.125L505.875,64.875z"/>
      <path id="designer-pen" fill="#FEFFFE" d="M525.75,59.084c0,0-0.423-0.262-0.969,0.088c-0.586,0.375-0.547,0.891-0.547,0.891l7.172,8.984l1.261,0.453
                                                l-0.104-1.328L525.75,59.084z"/>
    </g>
  </svg>

  <div class="deadline-timer">
    Remaining <span class="day">{opt.run_time}</span> <span class="days">s</span>
  </div>

</div>
                        """,
                        elem_id='run-anim',
                        visible=False,
                    )

            with gr.Group(elem_id='control-panel'):

                with gr.Row():
                    iface.tbox_prompt = gr.Textbox(
                        label='Edit Prompt for Background',
                        info='What do you want to draw?',
                        value=state.value.prompts[0],
                        placeholder=lambda: random.choice(prompt_suggestions),
                        scale=2,
                    )

                    iface.slider_strength = gr.Slider(
                        label='Prompt Strength',
                        info='Blends fg & bg in the prompt level, >0.8 Preferred.',
                        minimum=0.5,
                        maximum=1.0,
                        value=opt.default_prompt_strength,
                        scale=1,
                    )

                with gr.Row():
                    iface.tbox_neg_prompt = gr.Textbox(
                        label='Edit Negative Prompt for Background',
                        info='Add unwanted objects for this semantic brush.',
                        value=opt.default_negative_prompt,
                        scale=2,
                    )

                    iface.tbox_name = gr.Textbox(
                        label='Edit Brush Name',
                        info='Just for your convenience.',
                        value=state.value.prompt_names[0],
                        placeholder='🌄 Background',
                        scale=1,
                    )

                with gr.Row():
                    iface.slider_alpha = gr.Slider(
                        label='Mask Alpha',
                        info='Factor multiplied to the mask before quantization. Extremely sensitive, >0.98 Preferred.',
                        minimum=0.5,
                        maximum=1.0,
                        value=opt.default_mask_strength,
                    )

                    iface.slider_std = gr.Slider(
                        label='Mask Blur STD',
                        info='Blends fg & bg in the latent level, 0 for generation, 8-32 for inpainting.',
                        minimum=0.0001,
                        maximum=100.0,
                        value=opt.default_mask_std,
                    )

                    iface.slider_seed = gr.Slider(
                        label='Seed',
                        info='The global seed.',
                        minimum=-1,
                        maximum=2147483647,
                        step=1,
                        value=opt.seed,
                    )

    ### Attach event handlers

    for idx, btn in enumerate(iface.btn_semantics):
        btn.click(
            fn=partial(select_palette, idx=idx),
            inputs=[state, btn],
            outputs=[state] + iface.btn_semantics + [
                iface.tbox_name,
                iface.tbox_prompt,
                iface.tbox_neg_prompt,
                iface.slider_alpha,
                iface.slider_strength,
                iface.slider_std,
            ],
            api_name=f'select_palette_{idx}',
        )

    iface.btn_add_palette.click(
        fn=add_palette,
        inputs=state,
        outputs=[state, iface.btn_add_palette] + iface.btn_semantics[1:],
        api_name='create_new',
    )

    run_event = iface.btn_generate.click(
        fn=hide_element,
        inputs=None,
        outputs=iface.btn_generate,
        api_name='hide_run_button',
    ).then(
        fn=show_element,
        inputs=None,
        outputs=iface.run_animation,
        api_name='show_run_animation',
    )

    run_event.then(
        fn=run,
        inputs=[state, iface.ctrl_semantic],
        outputs=[state, iface.image_slot],
        api_name='run',
    ).then(
        fn=hide_element,
        inputs=None,
        outputs=iface.run_animation,
        api_name='hide_run_animation',
    ).then(
        fn=show_element,
        inputs=None,
        outputs=iface.btn_generate,
        api_name='show_run_button',
    )

    run_event.then(
        fn=None,
        inputs=None,
        outputs=None,
        api_name='run_animation',
        js=f"""
async () => {{
    // timer arguments: 
    //   #1 - time of animation in mileseconds, 
    //   #2 - days to deadline
    const animationTime = {opt.run_time + opt.prep_time};
    const days = {opt.run_time};
    jQuery('#progress-time-fill, #death-group').css({{'animation-duration': animationTime+'s'}});
    var deadlineAnimation = function () {{
        setTimeout(function() {{
            jQuery('#designer-arm-grop').css({{'animation-duration': '1.5s'}});
        }}, 0);
        setTimeout(function() {{
            jQuery('#designer-arm-grop').css({{'animation-duration': '1.0s'}});
        }}, {int((opt.run_time + opt.prep_time) * 1000 * 0.2)});
        setTimeout(function() {{
            jQuery('#designer-arm-grop').css({{'animation-duration': '0.7s'}});
        }}, {int((opt.run_time + opt.prep_time) * 1000 * 0.4)});
        setTimeout(function() {{
            jQuery('#designer-arm-grop').css({{'animation-duration': '0.3s'}});
        }}, {int((opt.run_time + opt.prep_time) * 1000 * 0.6)});
        setTimeout(function() {{
            jQuery('#designer-arm-grop').css({{'animation-duration': '0.2s'}});
        }}, {int((opt.run_time + opt.prep_time) * 1000 * 0.75)});
    }};
    var deadlineTextBegin = function () {{
        var el = jQuery('.deadline-timer');
        var html = 'Preparing...';
        el.html(html);
    }};
    var deadlineTextFinished = function () {{
        var el = jQuery('.deadline-timer');
        var html = 'Done! Retry?';
        el.html(html);
    }};
    var deadlineText = function (remainingTime) {{
        var el = jQuery('.deadline-timer');
        var htmlBase = 'Remaining <span class="day">' + remainingTime + '</span> <span class="days">s</span>';
        el.html(html);
        var html = '<div class="mask-red"><div class="inner">' + htmlBase + '</div></div><div class="mask-white"><div class="inner">' + htmlBase + '</div></div>';
        el.html(html);
    }};
    function timer(totalTime, deadline) {{
        var time = totalTime * 1000;
        var dayDuration = time / (deadline + {opt.prep_time});
        var actualDay = deadline + {opt.prep_time};
        var timer = setInterval(countTime, dayDuration);
        function countTime() {{
            --actualDay;
            if (actualDay > deadline) {{
                deadlineTextBegin();
            }} else if (actualDay > 0) {{
                deadlineText(actualDay);
                // jQuery('.deadline-timer .day').text(actualDay - {opt.prep_time});
            }} else {{
                clearInterval(timer);
                // jQuery('.deadline-timer .day').text(deadline);
                deadlineTextFinished();
            }}
        }}
    }}
    var runAnimation = function() {{
        timer(animationTime, days);
        remation();
        deadlineText({opt.run_time});
        console.log('begin interval', animationTime * 1000);
    }};
    runAnimation();
}}
        """
    )

    iface.slider_alpha.input(
        fn=change_mask_strength,
        inputs=[state, iface.slider_alpha],
        outputs=state,
        api_name='change_alpha',
    )
    iface.slider_std.input(
        fn=change_std,
        inputs=[state, iface.slider_std],
        outputs=state,
        api_name='change_std',
    )
    iface.slider_strength.input(
        fn=change_prompt_strength,
        inputs=[state, iface.slider_strength],
        outputs=state,
        api_name='change_strength',
    )
    iface.slider_seed.input(
        fn=reset_seed,
        inputs=[state, iface.slider_seed],
        outputs=state,
        api_name='reset_seed',
    )

    iface.tbox_name.input(
        fn=rename_prompt,
        inputs=[state, iface.tbox_name],
        outputs=[state] + iface.btn_semantics,
        api_name='prompt_rename',
    )
    iface.tbox_prompt.input(
        fn=change_prompt,
        inputs=[state, iface.tbox_prompt],
        outputs=state,
        api_name='prompt_edit',
    )
    iface.tbox_neg_prompt.input(
        fn=change_neg_prompt,
        inputs=[state, iface.tbox_neg_prompt],
        outputs=state,
        api_name='neg_prompt_edit',
    )

    # iface.style_select.change(
    #     fn=select_style,
    #     inputs=[state, iface.style_select],
    #     outputs=state,
    #     api_name='style_select',
    # )
    # iface.quality_select.change(
    #     fn=select_quality,
    #     inputs=[state, iface.quality_select],
    #     outputs=state,
    #     api_name='quality_select',
    # )

    iface.btn_export_state.click(lambda x: {k: v for k, v in vars(x).items() if k not in opt.excluded_keys}, state, iface.json_state_export)
    iface.btn_import_state.click(import_state, [state, iface.tbox_state_import], [
        state,
        *iface.btn_semantics,
        # iface.style_select,
        # iface.quality_select,
        iface.tbox_prompt,
        iface.tbox_name,
        iface.tbox_neg_prompt,
        iface.slider_strength,
        iface.slider_alpha,
        iface.slider_std,
        iface.slider_seed,
    ])

    # Realtime user input.
    iface.ctrl_semantic.change(
        fn=draw,
        inputs=[state, iface.ctrl_semantic],
        outputs=None,
        api_name='draw',
    )


if __name__ == '__main__':
    demo.launch(server_port=opt.port)
