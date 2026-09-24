FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY scripts /app/scripts
COPY models /app/models

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV RUNPOD_VOLUME_PATH=/runpod-volume

CMD ["python", "-u", "/app/handler.py"]
