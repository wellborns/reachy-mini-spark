"""Text-to-Speech – Piper neural TTS with espeak fallback (runs on CM4).

Output: float32 numpy array at the robot's output sample rate, ready to
pass directly to robot.media.push_audio_sample().
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal import resample_poly

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)

_PIPER_MODEL_DIR_DEFAULT = str(Path.home() / ".local/share/piper-voices")


class PiperTTS:
    """Neural TTS via piper-tts CLI or Python binding."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        tts_cfg = cfg.get("tts", {})
        self._voice: str = tts_cfg.get("voice", "en_US-lessac-medium")
        self._speed: float = float(tts_cfg.get("speed", 1.0))
        self._engine: str = tts_cfg.get("engine", "piper")
        self._espeak_voice: str = tts_cfg.get("espeak_voice", "en-us")
        self._espeak_speed: int = int(tts_cfg.get("espeak_speed", 150))
        self._piper_available: bool | None = None  # lazy-check

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str, robot: "ReachyMini") -> np.ndarray:
        """Return float32 audio array at the robot's output sample rate."""
        out_rate: int = robot.media.get_output_audio_samplerate()
        out_channels: int = robot.media.get_output_channels()

        if self._engine == "piper" and self._has_piper():
            pcm, src_rate = self._piper_synthesize(text)
        else:
            pcm, src_rate = self._espeak_synthesize(text)

        audio = _wav_bytes_to_float32(pcm)
        audio = _resample(audio, src_rate, out_rate)

        # Expand to output channel count
        if out_channels > 1:
            audio = np.tile(audio[:, np.newaxis], (1, out_channels))

        return audio

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_piper(self) -> bool:
        if self._piper_available is None:
            self._piper_available = shutil.which("piper") is not None
            if not self._piper_available:
                # Also check the Python binding
                try:
                    import piper  # noqa: F401
                    self._piper_available = True
                except ImportError:
                    logger.warning("piper not found; falling back to espeak.")
        return self._piper_available  # type: ignore[return-value]

    def _piper_synthesize(self, text: str) -> tuple[bytes, int]:
        """Run piper CLI and return raw WAV bytes + sample rate."""
        model_path = self._resolve_model()

        cmd = [
            "piper",
            "--model", model_path,
            "--output-raw",
            "--length-scale", str(1.0 / self._speed),
        ]

        try:
            proc = subprocess.run(
                cmd,
                input=text.encode(),
                capture_output=True,
                check=True,
                timeout=30,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return self._piper_python_synthesize(text)

        # piper --output-raw emits headerless 16-bit PCM at 22050 Hz
        pcm_raw = proc.stdout
        rate = 22_050
        wav_bytes = _raw_pcm_to_wav(pcm_raw, rate, channels=1, sampwidth=2)
        return wav_bytes, rate

    def _piper_python_synthesize(self, text: str) -> tuple[bytes, int]:
        """Use piper Python binding if the CLI is not available."""
        from piper import PiperVoice
        model_path = self._resolve_model()
        voice = PiperVoice.load(model_path)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(voice.config.sample_rate)
            for audio_bytes in voice.synthesize_stream_raw(text):
                wf.writeframes(audio_bytes)

        return buf.getvalue(), voice.config.sample_rate

    def _resolve_model(self) -> str:
        """Return path to the .onnx model file, downloading if needed."""
        import os
        model_dir = os.environ.get("PIPER_MODEL_DIR", _PIPER_MODEL_DIR_DEFAULT)
        onnx = f"{model_dir}/{self._voice}.onnx"
        if not os.path.isfile(onnx):
            _download_piper_model(self._voice, model_dir)
        return onnx

    def _espeak_synthesize(self, text: str) -> tuple[bytes, int]:
        """Fallback: espeak-ng → WAV bytes."""
        rate = 22_050
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        subprocess.run(
            [
                "espeak-ng",
                "-v", self._espeak_voice,
                "-s", str(self._espeak_speed),
                "-w", tmp_path,
                text,
            ],
            check=True,
            timeout=15,
        )
        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
        import os
        os.unlink(tmp_path)

        with wave.open(io.BytesIO(wav_bytes)) as wf:
            rate = wf.getframerate()

        return wav_bytes, rate


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _wav_bytes_to_float32(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
        sw = wf.getsampwidth()

    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2_147_483_648.0
    else:
        pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    return pcm


def _raw_pcm_to_wav(
    raw: bytes, rate: int, channels: int = 1, sampwidth: int = 2
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(raw)
    return buf.getvalue()


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    import math
    g = math.gcd(src, dst)
    return resample_poly(audio, dst // g, src // g).astype(np.float32)


def _download_piper_model(voice: str, model_dir: str) -> None:
    import os
    import urllib.request

    os.makedirs(model_dir, exist_ok=True)
    base = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    parts = voice.split("-")
    lang = parts[0]
    lang_region = f"{parts[0]}_{parts[1]}" if len(parts) > 1 else lang
    path = f"{lang}/{lang_region}/{voice}"

    for ext in (".onnx", ".onnx.json"):
        url = f"{base}/{path}/{voice}{ext}"
        dest = f"{model_dir}/{voice}{ext}"
        if not os.path.isfile(dest):
            logger.info("Downloading piper model: %s", url)
            urllib.request.urlretrieve(url, dest)
    logger.info("Piper model ready: %s", voice)
