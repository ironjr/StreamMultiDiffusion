import importlib
import importlib.util
import os
import subprocess
import sys
from typing import Dict, Literal, Optional

import fire
import platform
from packaging.version import Version


python = sys.executable
index_url = os.environ.get("INDEX_URL", "")


def version(package: str) -> Optional[Version]:
    try:
        return Version(importlib.import_module(package).__version__)
    except ModuleNotFoundError:
        return None


def is_installed(package: str) -> bool:
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run_python(command: str, env: Dict[str, str] = None) -> str:
    run_kwargs = {
        "args": f"\"{python}\" {command}",
        "shell": True,
        "env": os.environ if env is None else env,
        "encoding": "utf8",
        "errors": "ignore",
    }

    print(run_kwargs["args"])

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        print(f"Error running command: {command}", file=sys.stderr)
        raise RuntimeError(f"Error running command: {command}")

    return result.stdout or ""


def run_pip(command: str, env: Dict[str, str] = None) -> str:
    return run_python(f"-m pip {command}", env)

def get_cuda_version_from_torch() -> Optional[Literal["11", "12"]]:
    try:
        import torch
    except ImportError:
        return None

    return torch.version.cuda.split(".")[0]


def install(cu: Optional[Literal["11", "12"]] = get_cuda_version_from_torch()):
    if cu is None or cu not in ["11", "12"]:
        print("Could not detect CUDA version. Please specify manually.")
        return
    print("Installing TensorRT requirements...")

    if is_installed("tensorrt"):
        if version("tensorrt") < Version("9.0.0"):
            run_pip("uninstall -y tensorrt")

    cudnn_name = f"nvidia-cudnn-cu{cu}==8.9.4.25"

    if not is_installed("tensorrt"):
        run_pip(f"install {cudnn_name} --no-cache-dir")
        run_pip(
            "install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4 --no-cache-dir"
        )

    if not is_installed("polygraphy"):
        run_pip(
            "install polygraphy==0.47.1 --extra-index-url https://pypi.ngc.nvidia.com"
        )
    if not is_installed("onnx_graphsurgeon"):
        run_pip(
            "install onnx-graphsurgeon==0.3.26 --extra-index-url https://pypi.ngc.nvidia.com"
        )
    if platform.system() == 'Windows' and not is_installed("pywin32"):
        run_pip(
            "install pywin32"
        )

    pass


if __name__ == "__main__":
    fire.Fire(install)
