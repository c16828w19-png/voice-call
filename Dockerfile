FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8765

WORKDIR /app

# System deps: audio, git-lfs (for model weights), build tools for native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git git-lfs curl wget unzip ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Python deps (cached layer — changes rarely)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy everything from the build context: app.py, core/, templates/,
# models/, .gitattributes, hidden files — Docker COPY . . includes dotfiles.
COPY . .

# If git-lfs was used to store model weights, the COPY above may have copied
# pointer files (~130 bytes) instead of the actual weights. Detect that and pull.
RUN set -e; \
    if [ -d .git ]; then \
        if find models -name "*.safetensors" -size -1k 2>/dev/null | grep -q .; then \
            echo "[lfs] Model files are LFS pointers — running git lfs pull..."; \
            git lfs pull; \
        else \
            echo "[lfs] Model files already materialised, skipping pull."; \
        fi; \
    else \
        echo "[lfs] No .git directory in build context, cannot pull LFS."; \
    fi

# Copy models to the absolute path the app expects (/models/...), then validate
RUN set -eux; \
    mkdir -p /models; \
    if [ -d /app/models ]; then cp -a /app/models/. /models/; fi; \
    echo "--- /models layout ---"; \
    find /models -maxdepth 4 \( -type f -o -type d \) | sort | head -120; \
    echo "--- Sanity checks ---"; \
    [ -f /models/vosk-tts/vosk-model-small-en-us-0.15/am/final.mdl ] || \
        { echo "ERROR: Vosk model missing at /models/vosk-tts/vosk-model-small-en-us-0.15/am/final.mdl"; exit 1; }; \
    [ -f /models/pocket-tts/p_tts.safetensors ] || \
        { echo "ERROR: Pocket TTS weights missing at /models/pocket-tts/p_tts.safetensors"; exit 1; }; \
    [ -f /models/pocket-tts/tokenizer.model ] || \
        echo "WARN: tokenizer.model missing at /models/pocket-tts/tokenizer.model"; \
    [ -f /models/pocket-tts/local.yaml ] || \
        echo "WARN: local.yaml missing at /models/pocket-tts/local.yaml"; \
    ls -lh /models/pocket-tts/ /models/vosk-tts/vosk-model-small-en-us-0.15/

EXPOSE 8765

CMD ["python", "app.py"]