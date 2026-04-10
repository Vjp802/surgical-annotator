#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate surgical-annotator

# Fix for SAM 2 MPS fallback (aten::upsample_bicubic2d)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Load environment variables from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
