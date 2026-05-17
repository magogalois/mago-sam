# MAGO SAM-Audio

![MAGO](https://img.shields.io/badge/MAGO-Voice%20AI-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0wIDE4Yy00LjQyIDAtOC0zLjU4LTgtOHMzLjU4LTggOC04IDggMy41OCA4IDgtMy41OCA4LTggOHoiLz48L3N2Zz4=)
![SAM-Audio](https://img.shields.io/badge/Model-SAM--Audio-FF6F00?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Version](https://img.shields.io/badge/Version-0.1.0-green?style=for-the-badge)

SAM-Audio separates a target sound from an input audio file using a text prompt. This project wraps `facebookresearch/sam-audio` for local model loading and repeated inference in the **MAGO Voice AI** environment.

## Features

- **Text Prompted Audio Separation**: Separate target audio using descriptions such as `man speaking`, `drums`, or `dog barking`
- **Local Model Loading**: Load models from `/data/models/mstudio/sam-audio` without downloading on every run
- **Reusable Separator**: `SAMAudioSeparator` loads the model once and reuses it for repeated calls
- **GPU/CPU Support**: Select devices with `cpu`, `cuda`, or indexed GPUs such as `cuda:2`
- **Small/Large Model Support**: Use `sam-audio-small` for lower GPU memory or `sam-audio-large` for higher quality
- **CLI Support**: Run separation directly from terminal through `mago/cli.py`
- **Web Upload Test UI**: Upload audio and test separation through a browser on port `8304`
- **Output Artifacts**: Saves `target.wav` and `residual.wav` for each request

## Limitations

- Hugging Face access is required before downloading SAM-Audio checkpoints.
- Reranking is disabled in the current compatibility mode and `reranking_candidates` is forced to `1`.
- `xFormers` may print compatibility warnings if PyTorch, CUDA, Python, and xFormers builds do not match.
- CPU inference is supported but can be very slow.

## Model Directory

Local models are stored below:

```bash
/data/models/mstudio/sam-audio
```

Expected model paths:

```bash
/data/models/mstudio/sam-audio/facebook/sam-audio-small
/data/models/mstudio/sam-audio/facebook/sam-audio-large
/data/models/mstudio/sam-audio/facebook/pe-a-frame-large
/data/models/mstudio/sam-audio/facebook/sam-audio-judge
/data/models/mstudio/sam-audio/t5-base
/data/models/mstudio/sam-audio/roberta-base
```

Default model:

```bash
/data/models/mstudio/sam-audio/facebook/sam-audio-small
```

## Setup

Install Python dependencies:

```bash
./scripts/install.sh
```

If you need to download checkpoints from Hugging Face, request access to the SAM-Audio model first and authenticate:

```bash
huggingface-cli login
```

## Quick Start

### GPU

```bash
python mago/cli.py \
  --audio test/audio/sample_20s.flac \
  --description "man speaking" \
  --device cuda:2
```

### CPU

```bash
python mago/cli.py \
  --audio test/audio/sample_20s.flac \
  --description "man speaking" \
  --device cpu
```

### Large Model

```bash
python mago/cli.py \
  --audio test/audio/sample_20s.flac \
  --description "man speaking" \
  --device cuda:2 \
  --model /data/models/mstudio/sam-audio/facebook/sam-audio-large
```

Output files are written to:

```bash
exp/sam-audio/<content_id>/target.wav
exp/sam-audio/<content_id>/residual.wav
```

## Web Service

Start the upload test service on port `8304`:

```bash
python service/app.py
```

Open the browser:

```bash
http://localhost:8304
```

The service loads `SAMAudioSeparator` once at startup and reuses the loaded model for each uploaded file.

Environment variables:

| Environment Variable | Description | Default |
|---|---|---|
| `SAM_AUDIO_PORT` | Web service port | `8304` |
| `SAM_AUDIO_MODEL` | Local SAM-Audio model path | `/data/models/mstudio/sam-audio/facebook/sam-audio-small` |
| `SAM_AUDIO_DEVICE` | Inference device | `cuda:0` |
| `SAM_AUDIO_OUT_DIR` | Output directory | `exp/sam-audio` |
| `SAM_AUDIO_INPUT_SAMPLE_RATE` | Uploaded/microphone WAV sample rate | `16000` |
| `SAM_AUDIO_INPUT_SAMPLE_WIDTH` | Uploaded/microphone WAV bit depth | `16` |
| `SAM_AUDIO_INPUT_CHANNELS` | Uploaded/microphone WAV channels | `1` |

## Python Usage

Use `SAMAudioSeparator` when the model should be loaded once and reused:

```python
from mago.separator import SAMAudioSeparator

separator = SAMAudioSeparator(
    model_path="/data/models/mstudio/sam-audio/facebook/sam-audio-small",
    device="cuda:2",
    out_dir="exp/sam-audio",
)

result = separator(
    audio="test/audio/sample_20s.flac",
    description="man speaking",
)

print(result["target"])
print(result["residual"])
```

## Configuration

| Argument | Description | Default |
|---|---|---|
| `--audio` | Input audio file path | Required |
| `--description` | Target sound prompt | Required |
| `--out-dir` | Output directory | `exp/sam-audio` |
| `--model` | Local model directory | `/data/models/mstudio/sam-audio/facebook/sam-audio-small` |
| `--device` | Inference device (`cpu`, `cuda`, `cuda:2`) | `cuda:0` if available, otherwise `cpu` |
| `--predict-spans` | Predict target time spans from text prompt | Disabled |
| `--reranking-candidates` | Number of reranking candidates | `1` |

## Supported Formats

SAM-Audio uses `torchaudio` for loading audio files. Common formats such as WAV and FLAC are supported when the local audio backend can decode them.

## Volumes

| Host Path | Description |
|---|---|
| `/data/models/mstudio/sam-audio` | Local SAM-Audio and dependent model checkpoints |
| `exp/sam-audio` | Separation outputs |
| `test/audio` | Local test audio files |

## License

Copyright (c) 2026 MAGO. All rights reserved.
