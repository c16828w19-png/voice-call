FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends 
libsndfile1 
ffmpeg 
git 
git-lfs 
build-essential 
&& rm -rf /var/lib/apt/lists/*

RUN git lfs install

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && 
pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /models && 
cp -a /app/models/. /models/

RUN test -d /models/vosk-tts/vosk-model-small-en-us-0.15 && 
test -f /models/pocket-tts/local.yaml && 
test -s /models/pocket-tts/p_tts.safetensors && 
! grep -q "git-lfs.github.com/spec/v1" /models/pocket-tts/p_tts.safetensors

ENV PORT=8765

EXPOSE 8765

CMD ["sh", "-c", "python app.py"]