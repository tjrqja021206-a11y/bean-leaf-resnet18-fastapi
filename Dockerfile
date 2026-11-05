FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# CPU-only PyTorch: avoid large CUDA wheels in a CPU-only evaluation environment.
RUN python -m pip install --no-cache-dir \
      torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN mkdir -p /app/media/jobs /app/artifacts /app/.cache/huggingface
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
