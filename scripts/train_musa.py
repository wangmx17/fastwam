"""MUSA entry point; accepts the same Hydra overrides as scripts/train.py."""
import os
from pathlib import Path
import runpy

# DeepSpeed must select its backend before it is imported by the trainer.
os.environ.setdefault("DS_ACCELERATOR", "musa")

from fastwam.utils.musa import install


if __name__ == "__main__":
    install()
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")
