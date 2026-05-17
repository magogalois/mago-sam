#!/usr/bin/env python3
# encoding: utf-8

import argparse
from pathlib import Path

import torch
import torchaudio
from sam_audio import SAMAudio, SAMAudioProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate a target sound from an audio file using SAM-Audio."
    )
    parser.add_argument(
        "--audio",
        required=True,
        type=Path,
        help="Input audio file path.",
    )
    parser.add_argument(
        "--description",
        required=True,
        help='Target sound prompt, for example "man speaking" or "drums".',
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/sam-audio"),
        type=Path,
        help="Directory where target.wav and residual.wav will be written.",
    )
    parser.add_argument(
        "--model",
        default="facebook/sam-audio-large",
        help="Hugging Face model id or local checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device used for inference.",
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


def main() -> None:
    args = parse_args()

    if not args.audio.is_file():
        raise FileNotFoundError(f"Input audio file does not exist: {args.audio}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = SAMAudio.from_pretrained(args.model)
    processor = SAMAudioProcessor.from_pretrained(args.model)
    model = model.eval().to(args.device)

    print(f"Processing audio: {args.audio}")
    batch = processor(
        audios=[str(args.audio)],
        descriptions=[args.description.lower()],
    ).to(args.device)

    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=args.predict_spans,
            reranking_candidates=args.reranking_candidates,
        )

    sample_rate = processor.audio_sampling_rate
    target_path = args.output_dir / "target.wav"
    residual_path = args.output_dir / "residual.wav"

    torchaudio.save(str(target_path), result.target[0].cpu().unsqueeze(0), sample_rate)
    torchaudio.save(
        str(residual_path), result.residual[0].cpu().unsqueeze(0), sample_rate
    )

    print(f"Saved target audio: {target_path}")
    print(f"Saved residual audio: {residual_path}")


if __name__ == "__main__":
    main()
