# force-model-check-build-2

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN python -m pip install \
    --break-system-packages \
    --no-cache-dir \
    --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt

RUN python -m pip install \
    --break-system-packages \
    --no-cache-dir \
    -r /app/requirements.txt

RUN python -m pip install \
    --break-system-packages \
    --no-cache-dir \
    --ignore-installed \
    cryptography \
    runpod==1.12.0

COPY handler.py /app/handler.py
COPY scripts /app/scripts
COPY models /app/models
COPY api /app/api

# Copy classification checkpoints into their runtime locations
RUN mkdir -p \
    /app/models/unified_classifier \
    /app/models/glioma_grade

COPY best_unified_classifier.pt \
    /app/models/unified_classifier/best_unified_classifier.pt

COPY best_glioma_grade_model.pt \
    /app/models/glioma_grade/best_glioma_grade_model.pt


# ============================================================
# VERIFY REQUIRED MODEL CHECKPOINTS
# ============================================================
RUN python - <<'PY'
from pathlib import Path

required = [
    Path("/app/models/unified_classifier/best_unified_classifier.pt"),
    Path("/app/models/glioma_grade/best_glioma_grade_model.pt"),
]

print("=" * 60)
print("CHECKING SERVERLESS MODEL FILES")
print("=" * 60)

for path in required:
    print(f"Checking: {path}")

    if not path.exists():
        raise RuntimeError(f"MODEL NOT FOUND: {path}")

    if not path.is_file():
        raise RuntimeError(f"MODEL PATH IS NOT A FILE: {path}")

    size = path.stat().st_size

    if size <= 0:
        raise RuntimeError(f"MODEL FILE IS EMPTY: {path}")

    print(f"FOUND: {path}")
    print(f"SIZE: {size / (1024 * 1024):.2f} MB")
    print("-" * 60)

print("MODEL CHECK: PASS")
PY


# ============================================================
# VERIFY PYTHON DEPENDENCIES
# ============================================================
RUN python - <<'PY'
import torch
import torchvision
import monai
import nibabel
import boto3
import PIL
import reportlab
import runpod

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("monai:", monai.__version__)
print("nibabel:", nibabel.__version__)
print("runpod:", runpod.__version__)
print("CUDA available during build:", torch.cuda.is_available())
PY


CMD ["python", "-u", "/app/handler.py"]
