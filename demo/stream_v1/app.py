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

import os
import argparse
import random
import time
from functools import partial
from pprint import pprint

import numpy as np
from PIL import Image
from emoji import emoji_count

import gradio as gr
from huggingface_hub import snapshot_download

from model import StreamMultiDiffusion


### Utils

def starts_with_emoji(s):
    return bool(emoji_count(s.strip()[0]))


def log_streamer():
    pprint(streamer.ready_checklist)
    pprint({
        'number_of_masks': len(streamer.masks) if hasattr(streamer, 'masks') else 0,
        'number_of_prompts': len(streamer.prompts) - 1 if hasattr(streamer, 'prompts') else 0,
        'number_of_negative_prompts': len(streamer.negative_prompts) - 1 if hasattr(streamer, 'negative_prompts') else 0,

        'background': str(streamer.background),
        'model_key': str(streamer.state['model_key']),

        'prompts': streamer.prompts if hasattr(streamer, 'prompts') else None,
        'negative_prompts': streamer.negative_prompts if hasattr(streamer, 'negative_prompts') else None,
        'prompt_strengths': streamer.prompt_strengths if hasattr(streamer, 'prompt_strengths') else None,

        'prompt_embedding_shape': streamer.prompt_embeds.shape if hasattr(streamer, 'prompt_embeds') else None,
        'mask_stds': streamer.mask_stds if hasattr(streamer, 'mask_stds') else None,
        'mask_strengths': streamer.mask_strengths if hasattr(streamer, 'mask_strengths') else None,

        'masks_shape': streamer.masks.shape if hasattr(streamer, 'masks') else None,

        'init_noise_shape': streamer.init_noise.shape if hasattr(streamer, 'init_noise') else None,
        'stock_noise_shape': streamer.stock_noise.shape if hasattr(streamer, 'stock_noise') else None,
        'x_t_latent_buffer_shape': streamer.x_t_latent_buffer.shape if hasattr(streamer, 'x_t_latent_buffer') else None,
        'noise_lvs': streamer.noise_lvs[:, 0, 0, 0].tolist(),
    })


def log_state(state):
    pprint(vars(opt))

    if isinstance(state, gr.State):
        state = state.value
    pprint(vars(state))

    log_streamer()

    print_state(state)


def print_state(state):
    prefix = lambda v: (('  [ v ] ' if v else '  [ x ] ') if v in (True, False) else f'  [ {v} ] ')
    print('[INFO]     Front-end state:')
    print(prefix(state.has_background) + 'has background')
    print(prefix(state.active_palettes) + 'number of active palettes')
    print(prefix(state.is_streaming) + 'is streaming')


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

parser = argparse.ArgumentParser(description='StreamMultiDiffusion demonstation.')
parser.add_argument('-H', '--height', type=int, default=768)
parser.add_argument('-W', '--width', type=int, default=768)
parser.add_argument('--display_col', type=int, default=2)
parser.add_argument('--display_row', type=int, default=2)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--bootstrap_steps', type=int, default=1)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--port', type=int, default=8000)
opt = parser.parse_args()


### Global variables and data structures

device = f'cuda:{opt.device}' if opt.device >= 0 else 'cpu'

if opt.model is not None:
    if opt.model.endswith('.safetensors'):
        opt.model = os.path.abspath(os.path.join('checkpoints', opt.model))
    model_name = os.path.splitext(os.path.basename(opt.model))[0]

streamer = StreamMultiDiffusion(
    device,
    sd_version='1.5',
    hf_key=opt.model,
    height=opt.height,
    width=opt.width,
    cfg_type="none",
    seed=opt.seed,
    bootstrap_steps=opt.bootstrap_steps,
)

prompt_suggestions = [
    'apple',
    'banana',
    'cherry',
]

opt.max_palettes = 6 # Must be greater than 3 (due to UI design issue).
opt.default_prompt_strength = 1.0
opt.default_mask_strength = 1.0
opt.default_mask_std = 8.0
opt.default_negative_prompt = (
    'worst quality, normal quality, bad anatomy, bad hand, split screen, collage, cropped, deformed body features'
)
opt.num_display = opt.display_row * opt.display_col
opt.sleep_interval = 0.2
opt.verbose = True
opt.is_deployed_public = False

opt.background_button_name = '🌄 Background'


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
            gr.update() if i != state.active_palettes - 1 else gr.update(visible=True)
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


def draw(state, drawpad, mask_strength, prompt_strength):

    ### Process background.

    prev_has_background = state.has_background
    has_changed = drawpad['background'] != streamer.background.image
    is_empty = is_empty_image(drawpad['background'])

    # Option 1: At state [0. Start], proceeds to itself.

    if not prev_has_background and not has_changed:
        return [state] + [gr.update() for _ in range(opt.max_palettes + 11)]

    # Option 2: At state [1], [2], [3], proceeds to [0]: Should be handled in 'clear()' method.

    if prev_has_background and is_empty and has_changed:
        # Reset UI and state.
        state.has_background = False
        state.is_streaming = False

        state.current_palette = 0
        return [
            state,

            gr.update(value='⏸️', variant='secondary', interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),

            gr.update(value=opt.background_button_name, interactive=False),
            gr.update(value=streamer.background.prompt, label=f'Edit Prompt for Background'),
            gr.update(value=streamer.background.negative_prompt, label=f'Edit Negative Prompt for Background'),
            gr.update(value=opt.default_mask_strength, interactive=False),
            gr.update(value=opt.default_prompt_strength, interactive=False),
            gr.update(value=opt.default_mask_std, interactive=False),
            gr.update(interactive=False),
        ] + [gr.update(variant='primary', interactive=False)] + [
            gr.update(variant='secondary', interactive=False) for _ in range(opt.max_palettes)
        ]

    # Options:
    # 3-1: prev_has_background and not is_empty and not has_changed: No update on the background image.
    # 3-1: prev_has_background and not is_empty and has_changed: Normal update on the background image.
    # 3-2: not prev_has_background and has_changed and not is_empty: Normal initial upload of the background.
    # 3-3: not prev_has_background and has_changed and is_empty or
    #      prev_has_background and not has_changed and is_empty:
    #      This gives an only way to hack this app: The background image changes from 'None' to 'empty image'
    #      once at the initial stage. After the background is registered, it never becomes 'None', so this
    #      option is not able to be reached once any interaction is made. It is not hard to block this action,
    #      but I'll leave this open, since it allows pure generation from the model.

    # States transitions:
    # 1. At state [0. Start], proceeds to [1. Background Ready]
    # 2: At state [1], [2], [3], proceeds to themselves.

    if has_changed:
        background = drawpad['background'].convert('RGB')
        if state.prompts[0] != '':
            # Use the user specified prompt.
            streamer.update(
                background=background,
                background_prompt=state.prompts[0],
                background_negative_prompt=state.neg_prompts[0],
            )
        else:
            # Enforce internal image-to-text captioning to generate background prompt.
            streamer.update_background(
                background,
                None if state.prompts[0] == '' else state.prompts[0],
                state.neg_prompts[0],
            )
            state.prompts[0] = streamer.background.prompt
        state.has_background = True

    is_pointing_background = state.current_palette == 0
    updated_background_prompt = is_pointing_background and has_changed

    ### Process drawing.

    for idx, layer in enumerate(drawpad['layers']):
        if idx >= state.active_palettes:
            # More than pre-specified maximum number of layers are not allowed.
            break

        # Take only alpha channel from the gradio.ImageEditor and use it as a Black-White mask.
        layer = Image.fromarray(np.repeat(255 - np.asarray(layer)[..., -1:], 3, axis=-1))
        streamer.update(
            idx=idx,
            mask=layer,
            mask_strength=state.mask_strengths[idx],
            mask_std=state.mask_stds[idx],
            prompt=state.prompts[idx + 1],
            negative_prompt=state.neg_prompts[idx + 1],
            prompt_strength=state.prompt_strengths[idx],
        )

    ### Modify the back-end state.

    if not prev_has_background:
        # Enforce initial commit to the streamer.
        # This asserts background_update == True.
        streamer.commit()

    if opt.verbose:
        log_state(state)

    ### Modify the front-end state.

    if not prev_has_background:
        # Option 3: At state [0. Start], proceeds to [1. Background Ready]
        # This asserts background_update == True.
        return [
            state,

            gr.update(value='⏸️', variant='secondary', interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),

            gr.update(interactive=(not is_pointing_background)),
            (
                gr.update(value=state.prompts[0], interactive=True) if updated_background_prompt
                else gr.update(interactive=True)
            ),
            gr.update(interactive=True),
            gr.update(interactive=(not is_pointing_background)),
            gr.update(interactive=(not is_pointing_background)),
            gr.update(interactive=(not is_pointing_background)),
            gr.update(interactive=True),
        ] + [
            gr.update(interactive=True) for _ in range(opt.max_palettes + 1)
        ]
    else:
        # Option 4: At state [1], [2], [3], proceeds to themselves.
        # This does not modifies things.
        return [
            state,

            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=state.prompts[0]) if updated_background_prompt else gr.update(),
        ] + [
            gr.update() for _ in range(opt.max_palettes + 6)
        ]


def toggle_play(state):
    if not state.has_background or state.active_palettes <= 0:
        return [state] + [gr.update() for _ in range(3)]

    state.is_streaming = not state.is_streaming

    if opt.verbose:
        log_state(state)

    updates = [state]
    if state.is_streaming:
        updates.extend([
            gr.update(value='▶️', variant='primary'),
            gr.update(interactive=False),
            gr.update(interactive=False),
        ])
    else:
        updates.extend([
            gr.update(value='⏸️', variant='secondary'),
            gr.update(interactive=True),
            gr.update(interactive=True),
        ])
    return updates


def run_single_step(state, *slots):
    if (not streamer.is_ready_except_flush or
        not state.has_background or
        state.active_palettes <= 0 or
        state.is_streaming
    ):
        return [gr.update() for _ in range(opt.num_display)]

    res = streamer()
    slots = [res] + list(slots[:-1])
    return slots


def run_single_round(state, *slots):
    if (not streamer.is_ready_except_flush or
        not state.has_background or
        state.active_palettes <= 0 or
        state.is_streaming
    ):
        return [gr.update() for _ in range(opt.num_display)]

    return [streamer() for _ in range(opt.num_display)]


def change_prompt_strength(state, strength):
    if state.current_palette == 0:
        return state

    idx = state.current_palette - 1
    streamer.update(idx=idx, prompt_strength=strength)
    state.prompt_strengths[idx] = strength

    if opt.verbose:
        log_state(state)

    return state


def change_std(state, std):
    if state.current_palette == 0:
        return state

    idx = state.current_palette - 1
    streamer.update(idx=idx, mask_std=std)
    state.mask_stds[idx] = std

    if opt.verbose:
        log_state(state)

    return state


def change_mask_strength(state, strength):
    if state.current_palette == 0:
        return state

    idx = state.current_palette - 1
    streamer.update(idx=idx, mask_strength=strength)
    state.mask_strengths[idx] = strength

    if opt.verbose:
        log_state(state)

    return state


def reset_seed(seed):
    streamer.reset_seed(seed=seed)


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

    # TODO change the Name of the palette according to the prompt using txt2txt model.

    if state.current_palette == 0:
        streamer.update(background_prompt=prompt)
    else:
        streamer.update(idx=(state.current_palette - 1), prompt=prompt)

    if opt.verbose:
        log_state(state)

    return state


def change_neg_prompt(state, neg_prompt):
    state.neg_prompts[state.current_palette] = neg_prompt

    if state.current_palette == 0:
        streamer.update(background_negative_prompt=neg_prompt)
    else:
        streamer.update(idx=(state.current_palette - 1), negative_prompt=neg_prompt)

    if opt.verbose:
        log_state(state)

    return state


### Scheduler deamon

def run(state, *slots):
    while (
        streamer.is_ready_except_flush and
        state.has_background and
        state.active_palettes > 0 and
        state.is_streaming
    ):
        res = streamer()
        slots = [res] + list(slots[:-1])
        time.sleep(opt.sleep_interval)
        yield slots + [gr.update() for _ in range(3)]

    return [gr.update() for _ in range(opt.num_display)] + [
        gr.update(value='⏸️', variant='secondary'),
        gr.update(interactive=True),
        gr.update(interactive=True),
    ]


### Main application

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    ### State definition

    #                       -(entry point)->         [0. Start] (blocks every controls except background insertion)
    # [0. Start]            -(inserts background)->  [1. Background Ready] (releases every controls)
    # [1. Background Ready] -(active_palettes > 0)-> [2. Ready-to-Run] (can run 1-step-wise or 1-batch-wise)
    # [2. Ready-to-Run]     -(toggles play to on)->  [3. Streaming] (runs in stream)

    # [2. Ready-to-Run]     -(removes background)->  [0. Start] (blocks every controls except background insertion)
    # [3. Streaming]        -(removes background)->  [0. Start] (blocks every controls except background insertion)
    # [3. Streaming]        -(toggles play to off)-> [2. Ready-to-Run] (can run 1-step-wise or 1-batch-wise)

    iface = argparse.Namespace()
    state = argparse.Namespace()

    # Cursor.
    state.current_palette = 0 # 0: Background; 1,2,3,...: Layers

    # State variables (one-hot).
    state.has_background = False
    state.active_palettes = 3
    state.is_streaming = False

    # Front-end initialized to the default values.
    state.prompt_names = [
        opt.background_button_name,
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
    state.neg_prompts = [opt.default_negative_prompt for _ in range(opt.max_palettes + 1)]
    state.prompt_strengths = [opt.default_prompt_strength for _ in range(opt.max_palettes)]
    state.mask_strengths = [opt.default_mask_strength for _ in range(opt.max_palettes)]
    state.mask_stds = [opt.default_mask_std for _ in range(opt.max_palettes)]

    state = gr.State(state)

    ### TODO: Request for quota.


    ### Demo user interface

    gr.HTML(
        """
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
    <div>
        <h1 >StreamMultiDiffusion: Real-Time Interactive Generation with Region-Based Semantic Control</h1>
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
            gr.HTML(
                """
<div style="justify-content: center; align-items: center;">
    <h5 style="margin: 0; text-align: center;"><b>Customizable Semantic Palette</b></h5>
</div>
                """
            )

            iface.btn_semantics = [gr.Button(
                value=state.value.prompt_names[0],
                variant='primary',
                interactive=False,
            )]
            for i in range(opt.max_palettes):
                iface.btn_semantics.append(gr.Button(
                    value=state.value.prompt_names[i + 1],
                    variant='secondary',
                    visible=(i < state.value.active_palettes),
                    interactive=False,
                ))

            iface.btn_add_palette = gr.Button(
                value='Create New Semantic Brush',
                variant='primary',
                interactive=False,
            )

            gr.HTML(
                """
<div>
    </br>
</div>
<div style="justify-content: center; align-items: center;">
    <h5 style="margin: 0; text-align: center;"><b>Usage</b></h5>
    </br>
    <div style="justify-content: center; align-items: left; text-align: left;">
        <p>1. Upload a background image. A background prompt will be automatically generated for you. You can change the prompt afterwards. Try our sample background images in the <b>Examples</b> below!</p>
        <p>2. Click one of the <b>Semantic Brushes</b> above.</p>
        <p>2-1. Or, you can create your own <b>Semantic Brush</b> by pressing <b>Create New Semantic Brush</b> button and typing in the prompt below.</p>
        <p>3. Start drawing in <b>Semantic Draw</b> tab. Layers indicate the semantic brushes (from top to bottom order).</p>
        <p>4. Click [⏭] button for a single image generation, [⏭ X 4] for a batched generation, or toggle [▶️/⏸️] for streaming!</p>
    </div>
</div>
                """
            )

#             gr.HTML(
#                 """
# <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
#     <h5 style="margin: 0;"><b>... or run in your own 🤗 space!</b></h5>
# </div>
#                 """
#             )

            # gr.DuplicateButton()

        with gr.Column(scale=4):

            with gr.Row():

                with gr.Column():

                    with gr.Tab('Semantic Draw'):
                        iface.ctrl_semantic = gr.ImageEditor(
                            image_mode='RGBA',
                            sources=['upload', 'clipboard', 'webcam'],
                            transforms=['crop'],
                            crop_size=(opt.width, opt.height),
                            type='pil',
                            label='StreamMultiDiffusion Interactive Input',
                            every=1.0,
                        )

                    # with gr.Tab('Color Draw'):
                    #     iface.ctrl_color = gr.ImageEditor(
                    #         type='pil',
                    #         label='Tile ControlNet Interactive Input',
                    #     )

                with gr.Column():
                    with gr.Row():
                        iface.btn_runstop = gr.Button(value='⏸️', scale=1, variant='secondary', interactive=False)
                        iface.btn_nextstep = gr.Button(value='⏭', interactive=False, scale=1)
                        iface.btn_nextround = gr.Button(value=f'⏭ X {opt.num_display}', interactive=False, scale=1)

                    iface.image_slots = []
                    for _ in range(opt.display_row):
                        with gr.Row():
                            for _ in range(opt.display_col):
                                iface.image_slots.append(gr.Image(
                                    interactive=False,
                                    show_label=False,
                                    show_download_button=True,
                                    type='pil',
                                ))

            with gr.Row():
                iface.slider_alpha = gr.Slider(
                    label='Mask Alpha',
                    minimum=0.5,
                    maximum=1.0,
                    value=opt.default_mask_strength,
                    interactive=False,
                )

                iface.slider_std = gr.Slider(
                    label='Mask Blur STD',
                    minimum=0.0001,
                    maximum=100.0,
                    value=opt.default_mask_std,
                    interactive=False,
                )

                iface.slider_seed = gr.Slider(
                    label='Seed',
                    minimum=0,
                    maximum=2147483647,
                    step=1,
                    value=opt.seed,
                    interactive=(not opt.is_deployed_public),
                )

            with gr.Row():
                iface.tbox_prompt = gr.Textbox(
                    label='Edit Prompt for Background',
                    placeholder=random.choice(prompt_suggestions),
                    scale=2,
                )

                iface.slider_strength = gr.Slider(
                    label='Prompt Strength (> 0.8 Preferred)',
                    minimum=0.5,
                    maximum=1.0,
                    value=opt.default_prompt_strength,
                    scale=1,
                    interactive=False,
                )

            with gr.Row():
                iface.tbox_neg_prompt = gr.Textbox(
                    label='Edit Negative Prompt for Background',
                    value=opt.default_negative_prompt,
                    scale=2,
                )

                iface.tbox_name = gr.Textbox(
                    label='Edit Brush Name',
                    placeholder='🌄 Background',
                    interactive=False,
                    scale=1,
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

    iface.ctrl_semantic.change(
        fn=draw,
        inputs=[state, iface.ctrl_semantic, iface.slider_alpha, iface.slider_strength],
        outputs=[
            state, iface.btn_runstop, iface.btn_nextstep, iface.btn_nextround,
            iface.tbox_name, iface.tbox_prompt, iface.tbox_neg_prompt,
            iface.slider_alpha, iface.slider_strength, iface.slider_std,
            iface.btn_add_palette,
        ] + iface.btn_semantics,
        api_name='draw',
    )

    iface.btn_runstop.click(
        fn=toggle_play,
        inputs=state,
        outputs=[state, iface.btn_runstop, iface.btn_nextstep, iface.btn_nextround],
        api_name='toggle_play',
    ).then(
        fn=run,
        inputs=[state] + iface.image_slots,
        outputs=iface.image_slots + [iface.btn_runstop, iface.btn_nextstep, iface.btn_nextround],
        api_name='stream_run',
    )
    iface.btn_nextstep.click(
        fn=run_single_step,
        inputs=[state] + iface.image_slots,
        outputs=iface.image_slots,
        api_name='run_single_step',
    )
    iface.btn_nextround.click(
        fn=run_single_round,
        inputs=[state] + iface.image_slots,
        outputs=iface.image_slots,
        api_name='run_single_round',
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
        inputs=iface.slider_seed,
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


if __name__ == '__main__':
    demo.launch(server_port=opt.port)