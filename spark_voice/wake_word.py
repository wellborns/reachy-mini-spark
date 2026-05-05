"""Wake-word detection using openwakeword (optional; ARM-compatible, no API key).

When wake_word.enabled is false, this module is a no-op and the caller
falls through to always-on listening.

openwakeword ships several built-in models.  Because "hey reachy" isn't a
pre-trained model, we default to "hey_jarvis" which is close enough for
getting started.  You can drop a custom .tflite or .onnx model into
~/.local/share/openwakeword/ and set wake_word.model_path in config.yaml
to point to it.

Audio format expected by openwakeword: 16-bit PCM, 16 kHz, mono (same as
what we already have in the VAD pipeline).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal import resample_poly

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)

# openwakeword processes 80 ms chunks at 16 kHz = 1280 samples
_OWW_RATE = 16_000
_OWW_CHUNK = 1280   # 80 ms


def wait_for_wake_word(robot: "ReachyMini", cfg: dict[str, Any]) -> bool:
    """Block until the wake word is detected or stop is requested.

    Returns True when the wake word fires, False if called with wake word
    disabled (always-on mode – caller should proceed immediately).
    """
    ww_cfg = cfg.get("wake_word", {})
    if not ww_cfg.get("enabled", False):
        return True  # always-on: skip detection, go straight to VAD

    threshold: float = ww_cfg.get("threshold", 0.5)
    model_path: str | None = ww_cfg.get("model_path", None)

    oww = _load_oww(model_path)
    if oww is None:
        logger.warning("openwakeword unavailable – falling back to always-on mode.")
        return True

    logger.info("Wake word active – say '%s' …", ww_cfg.get("phrase", "hey jarvis"))

    hw_rate: int = robot.media.get_input_audio_samplerate()
    hw_channels: int = robot.media.get_input_channels()

    robot.media.start_recording()
    try:
        return _detection_loop(oww, robot, hw_rate, hw_channels, threshold)
    finally:
        robot.media.stop_recording()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_oww(model_path: str | None):
    try:
        import openwakeword
        from openwakeword.model import Model

        # download_models() was added in openwakeword 0.5+; in 0.4.x models are
        # bundled inside the package so we skip the download step if unavailable.
        if hasattr(openwakeword.utils, "download_models"):
            openwakeword.utils.download_models()

        if model_path:
            # Try modern kwarg first, fall back for older versions
            try:
                return Model(wakeword_models=[model_path], inference_framework="onnx")
            except TypeError:
                return Model(wakeword_models=[model_path])
        else:
            try:
                return Model(inference_framework="onnx")
            except TypeError:
                return Model()

    except ImportError:
        logger.debug("openwakeword not installed.")
        return None
    except Exception as exc:
        logger.warning("Could not load openwakeword model: %s", exc)
        return None


def _detection_loop(oww, robot, hw_rate, hw_channels, threshold) -> bool:
    import math
    buf: list[np.ndarray] = []
    buf_samples = 0

    while True:
        raw = robot.media.get_audio_sample()
        if raw is None or len(raw) == 0:
            time.sleep(0.01)
            continue

        # Mono
        if raw.ndim > 1:
            raw = raw[:, 0]

        # Resample to 16 kHz
        if hw_rate != _OWW_RATE:
            g = math.gcd(hw_rate, _OWW_RATE)
            raw = resample_poly(raw, _OWW_RATE // g, hw_rate // g).astype(np.float32)

        buf.append(raw)
        buf_samples += len(raw)

        # Process as many complete 80 ms chunks as we have
        while buf_samples >= _OWW_CHUNK:
            chunk_f32 = np.concatenate(buf)
            frame = chunk_f32[:_OWW_CHUNK]
            remainder = chunk_f32[_OWW_CHUNK:]
            buf = [remainder] if len(remainder) else []
            buf_samples = len(remainder)

            # openwakeword wants int16
            frame_i16 = np.clip(frame, -1.0, 1.0)
            frame_i16 = (frame_i16 * 32767).astype(np.int16)

            prediction = oww.predict(frame_i16)

            for model_name, score in prediction.items():
                if score >= threshold:
                    logger.info(
                        "Wake word detected! model=%s score=%.3f", model_name, score
                    )
                    return True
