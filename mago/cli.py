#!/usr/bin/env python3
# encoding: utf-8
# Copyright (c) 2026- MAGO
# AUTHORS
# Sukbong Kwon (Galois)

"""
Command line interface for SAM-Audio separation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mago.parser import parse_args
from mago.separator import SAMAudioSeparator
from sam.utils.logs import get_logger

# Define
logger = get_logger(__name__, level='INFO')


def run_separation(
    separator: SAMAudioSeparator,
    audio: str,
    description: str,
    predict_spans: bool = False,
    reranking_candidates: int = 1,
) -> None:
    """
    Run one SAM-Audio separation request.

    Args:
        separator (SAMAudioSeparator): Loaded SAM-Audio separator.
        audio (str): Input audio file path.
        description (str): Target sound prompt.
        predict_spans (bool): If True, predict target time spans from prompt.
        reranking_candidates (int): Number of candidates. Current compatibility uses 1.
    """
    result = separator(
        audio=audio,
        description=description,
        content_id="",
        predict_spans=predict_spans,
        reranking_candidates=reranking_candidates,
    )

    logger.info(f"Content id: {result['content_id']}")
    logger.info(f"Target audio: {result['target']}")
    logger.info(f"Residual audio: {result['residual']}")


def run_interactive(
    separator: SAMAudioSeparator,
    predict_spans: bool = False,
    reranking_candidates: int = 1,
) -> None:
    """
    Run interactive separation loop.

    Args:
        separator (SAMAudioSeparator): Loaded SAM-Audio separator.
        predict_spans (bool): If True, predict target time spans from prompt.
        reranking_candidates (int): Number of candidates. Current compatibility uses 1.
    """
    logger.info("Interactive mode is ready. Type q, quit, or exit to stop.")

    while True:
        audio = input("audio file path> ").strip()
        if audio.lower() in {"q", "quit", "exit"}:
            logger.info("Stop interactive mode.")
            break
        if not audio:
            logger.warning("Audio file path is empty.")
            continue

        description = input("description> ").strip()
        if description.lower() in {"q", "quit", "exit"}:
            logger.info("Stop interactive mode.")
            break
        if not description:
            logger.warning("Description is empty.")
            continue

        try:
            run_separation(
                separator=separator,
                audio=audio,
                description=description,
                predict_spans=predict_spans,
                reranking_candidates=reranking_candidates,
            )
        except Exception:
            logger.exception("Failed to separate audio")


def main() -> None:
    """
    Run SAM-Audio separation from terminal arguments.
    """
    args = parse_args()

    # Load separator once and run separation.
    separator = SAMAudioSeparator(
        model_path=args.model,
        device=args.device,
        out_dir=str(args.out_dir),
    )

    if args.audio and args.description:
        run_separation(
            separator=separator,
            audio=str(args.audio),
            description=args.description,
            predict_spans=args.predict_spans,
            reranking_candidates=args.reranking_candidates,
        )

    run_interactive(
        separator=separator,
        predict_spans=args.predict_spans,
        reranking_candidates=args.reranking_candidates,
    )


if __name__ == "__main__":
    main()
