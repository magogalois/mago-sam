# ==================================================================
# Builder image with python 3.12
# ------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV APT_INSTALL="apt-get install -y --no-install-recommends"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        git
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# ==================================================================
# Runtime image
# ------------------------------------------------------------------
FROM python:3.12-slim AS app-image

ENV DEBIAN_FRONTEND=noninteractive
ENV APT_INSTALL="apt-get install -y --no-install-recommends"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        ffmpeg \
        git \
        libgomp1 \
        libsndfile1
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=builder /venv /venv

ADD mago /app/mago
ADD sam /app/sam
ADD service /app/service
ADD README.md /app/README.md

ENV LC_ALL=C.UTF-8
ENV PATH="/venv/bin:$PATH"
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV SAM_AUDIO_PORT=8080
ENV SAM_AUDIO_DEVICE=cuda:0
ENV SAM_AUDIO_MODEL=/data/models/mstudio/sam-audio/facebook/sam-audio-small
ENV SAM_AUDIO_OUT_DIR=/app/exp/sam-audio

WORKDIR /app

EXPOSE 8080
ENTRYPOINT ["python", "service/app.py"]
