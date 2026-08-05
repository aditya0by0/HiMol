#!/usr/bin/env bash
# HiMol environment setup (Linux/macOS). Requires `uv` (https://docs.astral.sh/uv)
# and, for GPU training, an NVIDIA driver new enough for the CUDA 12.1 build
# (>= 525). Run from the repository root: bash setup.sh
set -euo pipefail

uv venv --python 3.12 .venv
source .venv/bin/activate

# PyTorch. The cu121 build works on most recent GPUs; switch the index to
# https://download.pytorch.org/whl/cu118 if your driver is older, and update
# the torch-scatter/torch-sparse find-links URL below to match.
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric companion CUDA kernels, matched to the torch build above.
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Remaining dependencies. rdkit is pinned: 2025+ raises a "Pre-condition
# Violation" in the motif decomposition. numpy is pinned <2 for compatibility.
uv pip install torch-geometric "numpy<2" scipy networkx pandas scikit-learn tqdm wandb rdkit==2024.3.5

python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Setup complete. Activate with: source .venv/bin/activate"
