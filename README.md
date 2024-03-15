# 🦦🪄 StreamMultiDiffusion

https://github.com/ironjr/MagicDraw/assets/12259041/9dda9740-58ba-4a96-b8c1-d40765979bd7

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

StreamMultiDiffusion is a *real-time* *interactive* *multiple*-text-to-image generation from user-assigned *regional* text prompts.

In other words, **you can now draw using brushes that paints *meanings* instead of *colors***!




---

## 🤖 Install

```bash
conda create -n smd python=3.10 && conda activate smd
pip install -r requirements.txt
```

## ⚡ Usage

StreamMultiDiffusion is served in three different forms.

1. The main interactive demo powered by Gradio is available at `src/app.py`. Just type the below line in your command prompt and open `https://localhost:8000` with any web browser will launch the app.

```bash
CUDA_VISIBLE_DEVICES=0 python app.py --model {your stable diffusion 1.5 checkpoint} --height 512 --width 512 --port 8000
```

2. Jupyter Lab demos are available in the `notebooks` directory. Simply type `jupyter lab` in the command prompt will open a Jupyter server.

3. Command line prompts by importing the `model` in `src`. For detailed examples and interfaces, please see the Jupyter demos.


### More Detailed Instruction

Coming soon!


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
