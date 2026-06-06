from __future__ import annotations

import argparse
import platform
import subprocess

import torch

from src.common import save_json


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture the experiment software/hardware stack.")
    parser.add_argument("--output", default="results/environment.json")
    args = parser.parse_args()
    cuda_available = torch.cuda.is_available()
    save_json(
        {
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "status", "--short"]),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
            "flash_attn": command_output(
                ["python", "-c", "import flash_attn; print(flash_attn.__version__)"]
            ),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
