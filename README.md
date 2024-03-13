# StreamMultiDiffusion

Real-time interactive region-based text-to-image generation tool.




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