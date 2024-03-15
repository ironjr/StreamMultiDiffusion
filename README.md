# 🦦🪄 StreamMultiDiffusion


<p align="center">
  <img src="./assets/demo.gif" width=90%>
</p>


## [StreamMultiDiffusion: Real-Time Interactive Generation with Region-Based Semantic Control](https://arxiv.org/abs/2403.09055)
> ##### Authors: [Jaerin Lee](http://jaerinlee.com/), [Daniel Sungho Jung](https://dqj5182.github.io/), [Kanggeon Lee](https://github.com/dlrkdrjs97/), and [Kyoung Mu Lee](https://cv.snu.ac.kr/index.php/~kmlee/)

<p align="center">
    <img src="assets/logo_cvlab.png" height=60>
</p>


<div align="center">

[![Project](https://img.shields.io/badge/Project-Page-green)](https://jaerinlee.com/research/streammultidiffusion/)
[![ArXiv](https://img.shields.io/badge/Arxiv-2403.09055-red)](https://arxiv.org/abs/2403.09055)
[![Github](https://img.shields.io/github/stars/ironjr/StreamMultiDiffusion)](https://github.com/ironjr/StreamMultiDiffusion)
[![X](https://img.shields.io/twitter/url?label=_ironjr_&url=https%3A%2F%2Ftwitter.com%2F_ironjr_)](https://twitter.com/_ironjr_)
[![LICENSE](https://img.shields.io/badge/license-MIT-lightgrey)](https://github.com/ironjr/StreamMultiDiffusion/blob/main/LICENSE)

</div>

tl;dr: StreamMultiDiffusion is a *real-time* *interactive* *multiple*-text-to-image generation from user-assigned *regional* text prompts.

In other words, **you can now draw using brushes that paints *meanings* instead of *colors***!

Our paper is mainly about establishing the compatibility between region-based controlling techniques of [MultiDiffusion](https://multidiffusion.github.io/) and acceleration techniques of [LCM](https://latent-consistency-models.github.io/) and [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion).
To our surprise, these works were not compatible before, limiting the possible applications from both branches of works.
The effect of acceleration and stabilization of multiple region-based text-to-image generation technique is compared using [StableDiffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) in the video below ⬇️:

https://github.com/ironjr/MagicDraw/assets/12259041/9dda9740-58ba-4a96-b8c1-d40765979bd7


---

## 🤖 Install

```bash
conda create -n smd python=3.10 && conda activate smd
pip install -r requirements.txt
```

## ⚡ Usage

### Overview

StreamMultiDiffusion is served in three different forms.

1. The main interactive demo powered by Gradio is available at `src/app.py`. Just type the below line in your command prompt and open `https://localhost:8000` with any web browser will launch the app.

```bash
CUDA_VISIBLE_DEVICES=0 python app.py --model {your stable diffusion 1.5 checkpoint} --height 512 --width 512 --port 8000
```

2. Jupyter Lab demos are available in the `notebooks` directory. Simply type `jupyter lab` in the command prompt will open a Jupyter server.

3. Command line prompts by importing the `model` in `src`. For detailed examples and interfaces, please see the Jupyter demos.


### More Detailed Instruction

We have provided detailed explanation of the application design and the expected usages in appendices of our [paper](https://arxiv.org/abs/2403.09055).
This section is a summary of its contents.
Although we expect everything to work fine, there may be unexpected bugs or misssed features in the implementation.
We are always welcoming issues and pull requests from you to improve this project! 🤗

### What is _Semantic Palette_ Anyway?

**Semantic palette** basically means that you paint things with semantics, i.e., text prompts, just like how you may use brush tools in commercial image editing software, such as Adobe Photoshop, etc.
Our acceleration technique for region-based controlled image generation allows users to edit their prompt masks similarly to drawing.
We couldn't find a good preexisting name for this type of user interface, so we named it as _semantic palette_, hoping for it to make sense to you. 😄

### Demo Application Architecture

There are two types of transaction data between the front-end and the back-end (`model.streammultidiffusion_pipeline.StreamMultiDiffusion`) of the application: a (1) background image object and a (2) list of text prompt-mask pairs.
We choose to call a pair of the latter as a _semantic brush_.
Despite its fancy name, a _semantic brush_ is just a pair of a text prompt and a regional mask assigned to the prompt, possibly with additional mask-controlling parameters.
Users interact with the application by registering and updating these two types of data to control the image generation stream.
The interface is summarized in the image below ⬇️:

<p align="center">
  <img src="./assets/app_design.png" width=90%>
</p>

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
| 9 | Mask alpha control | Changes the mask alpha value before quantization. Controls local content blending, but extremely sensitive. Recommended: >0.95 |
| 10 | Mask blur std. dev. control | Changes the standard deviation of the quantized mask of the current semantic brush. Less sensitive than mask alpha control. |
| 11 | Seed control | Changes the seed of the application. May not be needed, since we generate infinite stream of images. |
| 12 | Prompt edit | User can interactively change the positive/negative prompts at need. |
| 13 | Prompt strength control | Prompt embedding mix ratio between the current & the background. Helps global content blending. Recommended: >0.75 |
| 14 | Brush name edit | Adds convenience by changing the name of the brush. Does not affect the generation. Just for preference. |

### User Interface (CLI)

Coming Soon!

### Basic Usage (GUI)

Work in progress!


### Basic Usage (CLI)

Coming Soon!

---

## 🚩 **Updates**

- 🏃 Project page and the detailed demo instruction coming very soon!
- ✅ March 14, 2023: We have released our paper, StreamMultiDiffusion on [arXiv](https://arxiv.org/abs/2403.09055).
- ✅ March 13, 2023: Code release!


## 🌏 Citation

Please cite us if you find our project useful!

```latex
@article{lee2024streammultidiffusion,
    title={StreamMultiDiffusion: Real-Time Interactive Generation with Region-Based Semantic Control},
    author={Lee, Jaerin and Jung, Daniel Sungho and Lee, Kanggeon and Lee, Kyoung Mu},
    journal={arXiv preprint arXiv:2403.09055},
    year={2024}
}
```


## 🤗 Acknowledgement

Our code is based on the projects: [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion), [MultiDiffusion](https://multidiffusion.github.io/), and [Latent Consistency Model](https://latent-consistency-models.github.io/). Thank you for sharing such amazing works!

## 📧 Contact

If you have any questions, please email `jarin.lee@gmail.com`.
