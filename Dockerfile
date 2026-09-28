FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN python -m pip install --break-system-packages --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --break-system-packages --no-cache-dir -r /app/requirements.txt
RUN python -m pip install --break-system-packages --no-cache-dir --ignore-installed cryptography runpod==1.12.0

COPY handler.py /app/handler.py
COPY scripts /app/scripts
COPY models /app/models
COPY api /app/api

RUN echo "========================================" && \
    echo "CHECKING SERVERLESS MODEL FILES" && \
    echo "========================================" && \
    find /app/models -maxdepth 3 -type f -printf "%p  %s bytes\n" && \
    echo "----------------------------------------" && \
    test -s /app/models/unified_classifier/best_unified_classifier.pt && \
    echo "Unified classifier: FOUND" && \
    test -s /app/models/glioma_grade/best_glioma_grade_model.pt && \
    echo "Glioma grade model: FOUND" && \
    echo "MODEL CHECK: PASS"

RUN python - <<'PY'
import torch, torchvision, monai, nibabel, boto3, PIL, reportlab, runpod
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("monai", monai.__version__)
print("nibabel", nibabel.__version__)
print("CUDA available during build:", torch.cuda.is_available())
PY

CMD ["python", "-u", "/app/handler.py"]
