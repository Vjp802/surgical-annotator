#!/bin/bash
set -e

echo "=========================================="
echo " Surgical Annotator — Setup"
echo "=========================================="

echo "[1/4] Creating conda environment..."
if conda env list | grep -q "^surgical-annotator "; then
    echo "  → Environment already exists. Updating..."
    conda env update -n surgical-annotator -f environment.yml --prune
else
    conda env create -f environment.yml
fi
echo "  ✓ Conda environment ready."

eval "$(conda shell.bash hook)"
conda activate surgical-annotator

echo "[2/4] Installing SAM 3 from source..."
SAM3_DIR="/tmp/sam3"
if [ -d "$SAM3_DIR" ]; then
    echo "  → SAM 3 repo exists. Pulling latest..."
    cd "$SAM3_DIR" && git pull --quiet
else
    git clone --quiet https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
    cd "$SAM3_DIR"
fi
SAM3_BUILD_CUDA=0 pip install -e . --quiet
pip install setuptools==69.5.1 einops psutil --quiet
cd -
echo "  ✓ SAM 3 installed."

echo "[3/4] Pulling Gemma 4 via Ollama..."
if command -v ollama &> /dev/null; then
    ollama pull gemma4:e4b
    echo "  ✓ gemma4:e4b ready."
else
    echo "  ⚠ Ollama not found. Install from https://ollama.com then run: ollama pull gemma4:e4b"
fi

echo "[4/4] Creating directories..."
mkdir -p uploads/images uploads/videos frames exports data static
echo "  ✓ Directories created."

echo ""
echo "=========================================="
echo " Setup complete. Run ./run.sh to start."
echo "=========================================="
