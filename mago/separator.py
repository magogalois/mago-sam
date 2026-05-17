#!/usr/bin/env python3
# encoding: utf-8
# Copyright (c) 2026- MAGO
# AUTHORS
# Sukbong Kwon (Galois)

"""
SAM-Audio separator loads a local model once and separates target audio repeatedly.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Dict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torchaudio
from sam_audio import SAMAudio, SAMAudioProcessor

from sam.utils.logs import get_logger

# Define
logger = get_logger(__name__, level='INFO')
MODEL_ROOT = Path("/data/models/mstudio/sam-audio")
T5_MODEL_PATH = MODEL_ROOT / "t5-base"


class SAMAudioSeparator:
    """
    SAM-Audio processing class that separates target sound from audio.

    This class loads SAM-Audio model and processor from a local model directory
    once at initialization time, then reuses them for repeated separation calls.
    """
    def __init__(
        self,
        model_path: str,
        device: str = "",
        out_dir: str = "exp/sam-audio",
    ) -> None:
        """
        Initialize the SAM-Audio separator.

        Args:
            model_path (str): Path to the local SAM-Audio model directory.
            device (str): Device for inference. Examples: cpu, cuda, cuda:2.
            out_dir (str): Directory for output audio files.

        Raises:
            FileNotFoundError: If local model directory doesn't exist.
            RuntimeError: If SAM-Audio model loading fails.
        """
        self.model_path = Path(model_path)
        self.out_dir = out_dir
        self.device = self._get_device(device)

        logger.info(f"Device selected: {self.device}")

        # Retrieve model path from arguments
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found at path: {self.model_path}")

        # Load SAM-Audio model
        try:
            self.model = self._load_model(str(self.model_path))
            self.processor = SAMAudioProcessor.from_pretrained(str(self.model_path))
            self.model = self.model.eval().to(self.device)
            logger.info("SAM-Audio model loaded successfully.")
        except Exception as e:
            logger.exception("Error loading SAM-Audio model")
            raise RuntimeError(f"Failed to load SAM-Audio model: {e}") from e

        logger.info("Ready to separate target audio.")

    @classmethod
    def from_config_yml(
        cls,
        config_yml: str
    ) -> "SAMAudioSeparator":
        """
        Create an SAMAudioSeparator instance from a YAML configuration file.

        Args:
            config_yml (str): Path to YAML with model_path, device, and output options.

        Returns:
            SAMAudioSeparator: A fully initialized separator instance.
        """
        import yaml

        config = yaml.load(Path(config_yml).open("r", encoding="utf-8"), Loader=yaml.FullLoader)
        return cls(**config)

    def _load_model(
        self,
        model_path: str,
    ) -> SAMAudio:
        """
        Load SAM-Audio model from local directory.

        Args:
            model_path (str): Path to local SAM-Audio model directory.

        Returns:
            SAMAudio: Loaded SAM-Audio model.
        """
        text_encoder = self._get_text_encoder_config(model_path)
        return SAMAudio._from_pretrained(
            model_id=model_path,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            local_files_only=True,
            token=None,
            text_encoder=text_encoder,
            text_ranker=None,
            visual_ranker=None,
            span_predictor=None,
        )

    def _get_text_encoder_config(
        self,
        model_path: str,
    ) -> Dict:
        """
        Get text encoder config with local T5 model path.

        Args:
            model_path (str): Path to local SAM-Audio model directory.

        Returns:
            Dict: Text encoder config for local loading.
        """
        config_path = Path(model_path) / "config.json"
        config = json.load(config_path.open("r", encoding="utf-8"))
        text_encoder = config.get("text_encoder", {})

        # Use local T5 model to avoid Hugging Face Hub access.
        if not T5_MODEL_PATH.exists():
            raise FileNotFoundError(f"Local T5 model not found at path: {T5_MODEL_PATH}")
        text_encoder["name"] = str(T5_MODEL_PATH)
        return text_encoder

    def _get_device(
        self,
        device: str,
    ) -> str:
        """
        Get inference device.

        Args:
            device (str): Device for inference. Empty string selects automatically.

        Returns:
            str: Device name.
        """
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            return device
        if device == "cuda":
            device = "cuda:0"
        if not device.startswith("cuda:"):
            raise ValueError(f"Unsupported device: {device}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        device_index = int(device.split(":", maxsplit=1)[1])
        cuda_count = torch.cuda.device_count()
        if device_index >= cuda_count:
            raise ValueError(f"CUDA device {device_index} is not available. Found {cuda_count} GPU(s).")
        return device

    def __call__(
        self,
        audio: str,
        description: str,
        content_id: str = "",
        out_dir: str = "",
        predict_spans: bool = False,
        reranking_candidates: int = 1,
    ) -> Dict:
        """
        Separate target sound from audio.

        Args:
            audio (str): Path to input audio file.
            description (str): Target sound prompt.
            content_id (str): Id for this content; if empty, a UUID is generated.
            out_dir (str): Directory to save separated audio files.
            predict_spans (bool): If True, predict target time spans from text prompt.
            reranking_candidates (int): Number of candidates. Current compatibility uses 1.

        Returns:
            Dict: Result paths and metadata.
        """
        # Use a stable output folder for this request.
        content_id = content_id or str(uuid.uuid4())

        # All separated audio files are written below this folder.
        _out_dir = Path(out_dir or self.out_dir) / content_id
        _out_dir.mkdir(parents=True, exist_ok=True)

        audio_path = Path(audio)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Input audio file does not exist: {audio_path}")

        if reranking_candidates > 1:
            logger.warning("Reranking is disabled for current SAM-Audio compatibility.")
            reranking_candidates = 1

        # Set audio batch with text prompt.
        batch = self.processor(
            audios=[str(audio_path)],
            descriptions=[description.lower()],
        ).to(self.device)

        # Run audio separation.
        with torch.inference_mode():
            result = self.model.separate(
                batch,
                predict_spans=predict_spans,
                reranking_candidates=reranking_candidates,
            )

        detection = self._detect_target_activity(result.target[0])
        sample_rate = self.processor.audio_sampling_rate
        target_path = _out_dir / "target.wav"
        residual_path = _out_dir / "residual.wav"

        # Save separated audio files.
        torchaudio.save(str(target_path), result.target[0].cpu().unsqueeze(0), sample_rate)
        torchaudio.save(
            str(residual_path), result.residual[0].cpu().unsqueeze(0), sample_rate
        )

        return {
            "content_id": content_id,
            "audio": str(audio_path),
            "description": description,
            "target": str(target_path),
            "residual": str(residual_path),
            "sample_rate": sample_rate,
            "detection": detection,
        }

    def _detect_target_activity(
        self,
        waveform: torch.Tensor,
    ) -> Dict:
        """
        Detect whether separated target audio has meaningful activity.

        Args:
            waveform (torch.Tensor): Target waveform.

        Returns:
            Dict: Target activity metrics and detected flag.
        """
        wav = waveform.detach().float().cpu()
        if wav.numel() == 0:
            return {
                "detected": False,
                "label": "No target sound",
                "rms": 0.0,
                "peak": 0.0,
                "active_ratio": 0.0,
            }

        rms = torch.sqrt(torch.mean(wav.pow(2))).item()
        peak = torch.max(torch.abs(wav)).item()
        active_ratio = torch.mean((torch.abs(wav) > 0.01).float()).item()
        detected = rms >= 0.003 or peak >= 0.05 or active_ratio >= 0.01

        return {
            "detected": detected,
            "label": "Target sound detected" if detected else "No clear target sound",
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "active_ratio": round(active_ratio, 6),
        }
