# PhysioTokenizer — HuggingFace Space Dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System dependencies for wfdb, mne
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git \
    libsndfile1 libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub datasets gradio

# Copy project
COPY . .

# Pre-create directories
RUN mkdir -p data datasets checkpoints results figures logs

# Default entrypoint (override in HF Space settings)
CMD ["python", "scripts/run_on_hf.py", "--mode", "all", "--config", "ecg_single_band"]
