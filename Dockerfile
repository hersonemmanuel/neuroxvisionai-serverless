FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV RUNPOD_VOLUME_PATH=/runpod-volume

# Upgrade packaging tools in RunPod's externally-managed Python environment
RUN python -m pip install \
    --break-system-packages \
    --no-cache-dir \
    --upgrade \
    pip setuptools wheel

# Install MONAI + Nibabel separately
COPY requirements.txt /app/requirements.txt

RUN python -m pip install \
    --break-system-packages \
    --no-cache-dir \
    -r /app/requirements.txt

# RunPod SDK has a known cryptography conflict with the base image,
# so install it separately while ignoring the preinstalled copy.
RUN python -m pip install \
    --break-system-packages \
    --no-cache-dir \
    --ignore-installed cryptography \
    runpod==1.12.0

# Verify the runtime during image build
RUN python - <<'PY'
import torch
import numpy
import monai
import nibabel
import runpod

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("NumPy:", numpy.__version__)
print("MONAI:", monai.__version__)
print("Nibabel:", nibabel.__version__)
print("RunPod imported successfully")
print("BUILD IMPORT CHECK: PASS")
PY

COPY handler.py /app/handler.py
COPY scripts /app/scripts
COPY models /app/models

CMD ["python", "-u", "/app/handler.py"]
