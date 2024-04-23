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

from model import StableMultiDiffusionPipeline
from util import seed_everything


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

parser = argparse.ArgumentParser(description='Semantic Palette demo powered by StreamMultiDiffusion.')
parser.add_argument('-H', '--height', type=int, default=768)
parser.add_argument('-W', '--width', type=int, default=1920)
parser.add_argument('--model', type=str, default=None, help='Hugging face model repository or local path for a SD1.5 model checkpoint to run.')
parser.add_argument('--bootstrap_steps', type=int, default=1)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--port', type=int, default=8000)
opt = parser.parse_args()


### Global variables and data structures

device = f'cuda:{opt.device}' if opt.device >= 0 else 'cpu'


if opt.model is None:
    model_dict = {
        'Blazing Drive V11m': 'ironjr/BlazingDriveV11m',
        # 'Real Cartoon Pixar V5': 'ironjr/RealCartoon-PixarV5',
        # 'Kohaku V2.1': 'KBlueLeaf/kohaku-v2.1',
        # 'Realistic Vision V5.1': 'ironjr/RealisticVisionV5-1',
        # 'Stable Diffusion V1.5': 'runwayml/stable-diffusion-v1-5',
    }
else:
    if opt.model.endswith('.safetensors'):
        opt.model = os.path.abspath(os.path.join('checkpoints', opt.model))
    model_dict = {os.path.splitext(os.path.basename(opt.model))[0]: opt.model}

models = {
    k: StableMultiDiffusionPipeline(device, sd_version='1.5', hf_key=v, has_i2t=False)
    for k, v in model_dict.items()
}


prompt_suggestions = [
    '1girl, souryuu asuka langley, neon genesis evangelion, solo, upper body, v, smile, looking at viewer',
    '1boy, solo, portrait, looking at viewer, white t-shirt, brown hair',
    '1girl, arima kana, oshi no ko, solo, upper body, from behind',
]

opt.max_palettes = 5
opt.default_prompt_strength = 1.0
opt.default_mask_strength = 1.0
opt.default_mask_std = 0.0
opt.default_negative_prompt = (
    'nsfw, worst quality, bad quality, normal quality, cropped, framed'
)
opt.verbose = True
opt.colors = [
    '#000000',
    '#2692F3',
    '#F89E12',
    '#16C232',
    '#F92F6C',
    '#AC6AEB',
    # '#92C62C',
    # '#92C6EC',
    # '#FECAC0',
]


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


def select_model(state, model_id):
    state.model_id = model_id
    if opt.verbose:
        log_state(state)

    return state


def import_state(state, json_text):
    current_palette = state.current_palette
    # active_palettes = state.active_palettes
    state = argparse.Namespace(**json.loads(json_text))
    state.active_palettes = opt.max_palettes
    return [state] + [
        gr.update(value=v, visible=True) for v in state.prompt_names
    ] + [
        state.model_id,
        state.prompts[current_palette],
        state.prompt_names[current_palette],
        state.neg_prompts[current_palette],
        state.prompt_strengths[current_palette - 1],
        state.mask_strengths[current_palette - 1],
        state.mask_stds[current_palette - 1],
        state.seed,
    ]


### Main worker

def generate(state, *args, **kwargs):
    return models[state.model_id](*args, **kwargs)



def run(state, drawpad):
    seed_everything(state.seed if state.seed >=0 else np.random.randint(2147483647))
    print('Generate!')

    background = drawpad['background'].convert('RGBA')
    inpainting_mode = np.asarray(background).sum() != 0
    print('Inpainting mode: ', inpainting_mode)

    user_input = np.asarray(drawpad['layers'][0]) # (H, W, 4)
    foreground_mask = torch.tensor(user_input[..., -1])[None, None] # (1, 1, H, W)
    user_input = torch.tensor(user_input[..., :-1]) # (H, W, 3)

    palette = torch.tensor([
        tuple(int(s[i+1:i+3], 16) for i in (0, 2, 4))
        for s in opt.colors[1:]
    ]) # (N, 3)
    masks = (palette[:, None, None, :] == user_input[None]).all(dim=-1)[:, None, ...] # (N, 1, H, W)
    has_masks = [i for i, m in enumerate(masks.sum(dim=(1, 2, 3)) == 0) if not m]
    print('Has mask: ', has_masks)
    masks = masks * foreground_mask
    masks = masks[has_masks]

    # if inpainting_mode:
    #     prompts = state.prompts[1:len(masks)+1]
    #     negative_prompts = state.neg_prompts[1:len(masks)+1]
    #     mask_strengths = state.mask_strengths[:len(masks)]
    #     mask_stds = state.mask_stds[:len(masks)]
    #     prompt_strengths = state.prompt_strengths[:len(masks)]
    # else:
    #     masks = torch.cat([torch.ones_like(foreground_mask), masks], dim=0)
    #     prompts = state.prompts[:len(masks)+1]
    #     negative_prompts = state.neg_prompts[:len(masks)+1]
    #     mask_strengths = [1] + state.mask_strengths[:len(masks)]
    #     mask_stds = [0] + [state.mask_stds[:len(masks)]
    #     prompt_strengths = [1] + state.prompt_strengths[:len(masks)]

    if inpainting_mode:
        prompts = [state.prompts[v + 1] for v in has_masks]
        negative_prompts = [state.neg_prompts[v + 1] for v in has_masks]
        mask_strengths = [state.mask_strengths[v] for v in has_masks]
        mask_stds = [state.mask_stds[v] for v in has_masks]
        prompt_strengths = [state.prompt_strengths[v] for v in has_masks]
    else:
        masks = torch.cat([torch.ones_like(foreground_mask), masks], dim=0)
        prompts = [state.prompts[0]] + [state.prompts[v + 1] for v in has_masks]
        negative_prompts = [state.neg_prompts[0]] + [state.neg_prompts[v + 1] for v in has_masks]
        mask_strengths = [1] + [state.mask_strengths[v] for v in has_masks]
        mask_stds = [0] + [state.mask_stds[v] for v in has_masks]
        prompt_strengths = [1] + [state.prompt_strengths[v] for v in has_masks]

    return generate(
        state,
        prompts,
        negative_prompts,
        masks=masks,
        mask_strengths=mask_strengths,
        mask_stds=mask_stds,
        prompt_strengths=prompt_strengths,
        background=background.convert('RGB'),
        background_prompt=state.prompts[0],
        background_negative_prompt=state.neg_prompts[0],
        height=opt.height,
        width=opt.width,
        bootstrap_steps=2,
    )



### Load examples


root = pathlib.Path(__file__).parent
print(root)
example_root = os.path.join(root, 'examples')
example_images = glob.glob(os.path.join(example_root, '*.png'))
example_images = [Image.open(i) for i in example_images]

with open(os.path.join(example_root, 'prompt_background_advanced.txt')) as f:
    prompts_background = [l.strip() for l in f.readlines() if l.strip() != '']

with open(os.path.join(example_root, 'prompt_girl.txt')) as f:
    prompts_girl = [l.strip() for l in f.readlines() if l.strip() != '']

with open(os.path.join(example_root, 'prompt_boy.txt')) as f:
    prompts_boy = [l.strip() for l in f.readlines() if l.strip() != '']

with open(os.path.join(example_root, 'prompt_props.txt')) as f:
    prompts_props = [l.strip() for l in f.readlines() if l.strip() != '']
    prompts_props = {l.split(',')[0].strip(): ','.join(l.split(',')[1:]).strip() for l in prompts_props}

prompt_background = lambda: random.choice(prompts_background)
prompt_girl = lambda: random.choice(prompts_girl)
prompt_boy = lambda: random.choice(prompts_boy)
prompt_props = lambda: np.random.choice(list(prompts_props.keys()), size=(opt.max_palettes - 2), replace=False).tolist()


### Main application

css = f"""
#run-button {{
    font-size: 30pt;
    background-image: linear-gradient(to right, #4338ca 0%, #26a0da 51%, #4338ca 100%);
    margin: 0;
    padding: 15px 45px;
    text-align: center;
    text-transform: uppercase;
    transition: 0.5s;
    background-size: 200% auto;
    color: white;
    box-shadow: 0 0 20px #eee;
    border-radius: 10px;
    display: block;
    background-position: right center;
}}

#run-button:hover {{
    background-position: left center;
    color: #fff;
    text-decoration: none;
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

.layer-wrap {{
    display: none;
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


with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:

    iface = argparse.Namespace()

    def _define_state():
        state = argparse.Namespace()

        # Cursor.
        state.current_palette = 0 # 0: Background; 1,2,3,...: Layers
        state.model_id = list(model_dict.keys())[0]

        # State variables (one-hot).
        state.active_palettes = 1

        # Front-end initialized to the default values.
        prompt_props_ = prompt_props()
        # state.prompt_names = [
        #     '🌄 Background',
        #     '👧 Girl',
        #     '👦 Boy',
        #     '🐶 Dog',
        #     '🚗 Car',
        #     '💐 Garden',
        # ] + ['🎨 New Palette' for _ in range(opt.max_palettes - 5)]
        # state.prompts = [
        #     'Maximalism, best quality, high quality, city lights, times square',
        #     '1girl, looking at viewer, pink hair, leather jacket',
        #     '1boy, looking at viewer, brown hair, casual shirt',
        #     'Doggy body part',
        #     'Car',
        #     'Flower garden',
        # ] + ['' for _ in range(opt.max_palettes - 5)]
        state.prompt_names = [
            '🌄 Background',
            '👧 Girl',
            '👦 Boy',
        ] + prompt_props_ + ['🎨 New Palette' for _ in range(opt.max_palettes - 5)]
        state.prompts = [
            prompt_background(),
            prompt_girl(),
            prompt_boy(),
        ] + [prompts_props[k] for k in prompt_props_] + ['' for _ in range(opt.max_palettes - 5)]
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
        <h1>🧠 Semantic Palette 🎨</h1>
        <h5 style="margin: 0;">powered by</h5>
        <h3>StreamMultiDiffusion: Real-Time Interactive Generation with Region-Based Semantic Control</h3>
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
            <a href='https://huggingface.co/papers/2403.09055'>
                <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Paper-StreamMultiDiffusion-yellow'>
            </a>
            &nbsp;
            <a href='https://huggingface.co/spaces/ironjr/StreamMultiDiffusion'>
                <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Demo-StreamMultiDiffusion-yellow'>
            </a>
            &nbsp;
            <a href='https://huggingface.co/spaces/ironjr/SemanticPalette'>
                <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Demo-SD1.5-yellow'>
            </a>
            &nbsp;
            <a href='https://huggingface.co/spaces/ironjr/SemanticPaletteXL'>
                <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Demo-SDXL-yellow'>
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

        iface.image_slot = gr.Image(
            interactive=False,
            show_label=False,
            show_download_button=True,
            type='pil',
            label='Generated Result',
            elem_id='output-screen',
            value=lambda: random.choice(example_images),
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
                )

            with gr.Accordion(label='Import/Export Semantic Palette', open=False):
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
    <p>1-1. Type in the background prompt. Background is not required if you paint the whole drawpad.</p>
    <p>1-2. (Optional: <em><b>Inpainting mode</b></em>) Uploading a background image will make the app into inpainting mode. Removing the image returns to the creation mode. In the inpainting mode, increasing the <em>Mask Blur STD</em> > 8 for every colored palette is recommended for smooth boundaries.</p>
    <p>2. Select a semantic brush by clicking onto one in the <b>Semantic Palette</b> above. Edit prompt for the semantic brush.</p>
    <p>2-1. If you are willing to draw more diverse images, try <b>Create New Semantic Brush</b>.</p>
    <p>3. Start drawing in the <b>Semantic Drawpad</b> tab. The brush color is directly linked to the semantic brushes.</p>
    <p>4. Click [<b>GENERATE!</b>] button to create your (large-scale) artwork!</p>
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

                with gr.Column(scale=3):

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
                    )

                with gr.Column(scale=1):

                    iface.btn_generate = gr.Button(
                        value='Generate!',
                        variant='primary',
                        # scale=1,
                        elem_id='run-button'
                    )

                    iface.model_select = gr.Radio(
                        list(model_dict.keys()),
                        label='Stable Diffusion Checkpoint',
                        info='Choose your favorite style.',
                        value=state.value.model_id,
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

                    iface.tbox_name = gr.Textbox(
                        label='Edit Brush Name',
                        info='Just for your convenience.',
                        value=state.value.prompt_names[0],
                        placeholder='🌄 Background',
                        scale=1,
                    )

                with gr.Row():
                    iface.tbox_neg_prompt = gr.Textbox(
                        label='Edit Negative Prompt for Background',
                        info='Add unwanted objects for this semantic brush.',
                        value=opt.default_negative_prompt,
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

    iface.btn_generate.click(
        fn=run,
        inputs=[state, iface.ctrl_semantic],
        outputs=iface.image_slot,
        api_name='run',
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

    iface.model_select.change(
        fn=select_model,
        inputs=[state, iface.model_select],
        outputs=state,
        api_name='model_select',
    )

    iface.btn_export_state.click(lambda x: vars(x), state, iface.json_state_export)
    iface.btn_import_state.click(import_state, [state, iface.tbox_state_import], [
        state,
        *iface.btn_semantics,
        iface.model_select,
        iface.tbox_prompt,
        iface.tbox_name,
        iface.tbox_neg_prompt,
        iface.slider_strength,
        iface.slider_alpha,
        iface.slider_std,
        iface.slider_seed,
    ])


if __name__ == '__main__':
    demo.launch(server_port=opt.port)
