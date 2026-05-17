#!/usr/bin/env python3
# encoding: utf-8
# Copyright (c) 2026- MAGO
# AUTHORS
# Sukbong Kwon (Galois)

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    # Set command line arguments
    parser = argparse.ArgumentParser(
        description="Separate a target sound from an audio file using SAM-Audio."
    )
    parser.add_argument(
        "--audio",
        default=None,
        type=Path,
        help="Input audio file path. If omitted, interactive mode is used.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help='Target sound prompt, for example "man speaking" or "drums". If omitted, interactive mode is used.',
    )
    parser.add_argument(
        "--out-dir",
        default=Path("exp/sam-audio"),
        type=Path,
        help="Directory where target.wav and residual.wav will be written.",
    )
    parser.add_argument(
        "--model",
        default="/data/models/mstudio/sam-audio/facebook/sam-audio-small",
        help="Hugging Face model id or local checkpoint directory.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face access token. Default uses the token saved by huggingface-cli login.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device used for inference. Examples: cpu, cuda, cuda:2.",
    )
    parser.add_argument(
        "--predict-spans",
        action="store_true",
        help="Predict target time spans from the text prompt. Slower but often better for events.",
    )
    parser.add_argument(
        "--reranking-candidates",
        default=1,
        type=int,
        help="Number of generated candidates to rerank. Higher values need more memory.",
    )
    return parser.parse_args()
