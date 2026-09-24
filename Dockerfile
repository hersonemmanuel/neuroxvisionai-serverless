FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV RUNPOD_VOLUME_PATH=/runpod-volume

RUN python -m pip install \
    --no-cache-dir \
    --upgrade \
    pip setuptools wheel

COPY requirements.txt /app/requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    -r /app/requirements.txt

RUN python - <<'PY'
import torch
import numpy
import runpod
import monai
import nibabel

print("Torch:", torch.__version__)
print("NumPy:", numpy.__version__)
print("MONAI:", monai.__version__)
print("Nibabel:", nibabel.__version__)
print("RunPod SDK imported successfully")
print("BUILD IMPORT CHECK: PASS")
PY

COPY handler.py /app/handler.py
COPY scripts /app/scripts
COPY models /app/models

CMD ["python", "-u", "/app/handler.py"]
