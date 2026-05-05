"""Speech-to-Text via faster-whisper (runs entirely on the CM4)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class WhisperSTT:
    """Thin wrapper around faster-whisper for on-device transcription."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        stt_cfg = cfg.get("stt", {})
        self._model_size: str = stt_cfg.get("model", "tiny")
        self._language: str | None = stt_cfg.get("language", "en") or None
        self._device: str = stt_cfg.get("device", "cpu")
        self._compute_type: str = stt_cfg.get("compute_type", "int8")
        self._beam_size: int = stt_cfg.get("beam_size", 1)
        self._model = None
        self._load()  # pre-load at startup so first utterance isn't delayed

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed.  Run: pip install faster-whisper"
            ) from exc

        logger.info(
            "Loading Whisper model '%s' on %s (%s) …",
            self._model_size, self._device, self._compute_type,
        )
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("Whisper model ready.")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 16 kHz mono audio; return stripped text."""
        self._load()

        # vad_filter disabled – audio is already VAD-trimmed by collect_utterance()
        # enabling it here double-filters and often discards valid speech
        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
        )
        text = " ".join(seg.text for seg in segments).strip()
        logger.debug(
            "STT (lang=%s, prob=%.2f): %r",
            info.language, info.language_probability, text,
        )
        return text
