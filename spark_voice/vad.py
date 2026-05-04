"""Voice Activity Detection – collects a complete utterance from the mic.

Uses webrtcvad for lightweight, real-time speech/silence discrimination.
Audio from the Reachy Mini SDK is float32 at whatever rate the hardware
reports; we resample to 16 kHz before feeding webrtcvad.
"""

from __future__ import annotations

import collections
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import webrtcvad
from scipy.signal import resample_poly

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)

# webrtcvad only accepts 8 / 16 / 32 kHz
_VAD_RATE = 16_000


def _to_int16(samples: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] to int16 PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    import math
    g = math.gcd(src_rate, dst_rate)
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)


def collect_utterance(
    robot: "ReachyMini",
    cfg: dict,
) -> np.ndarray | None:
    """Record until an utterance is complete; return float32 16 kHz mono array.

    Returns None on timeout or if no speech was detected.
    """
    vad_cfg = cfg.get("vad", {})
    mode: int = vad_cfg.get("mode", 2)
    frame_ms: int = vad_cfg.get("frame_duration_ms", 30)
    silence_ms: int = vad_cfg.get("silence_duration_ms", 1200)
    min_speech_ms: int = vad_cfg.get("min_speech_duration_ms", 400)
    max_s: float = vad_cfg.get("max_utterance_duration_s", 30)

    vad = webrtcvad.Vad(mode)
    frame_samples = (_VAD_RATE * frame_ms) // 1000  # samples per vad frame
    silence_frames_needed = silence_ms // frame_ms
    min_speech_frames = min_speech_ms // frame_ms

    hw_rate: int = robot.media.get_input_audio_samplerate()
    hw_channels: int = robot.media.get_input_channels()
    hw_frame_samples = (hw_rate * frame_ms) // 1000

    logger.debug(
        "VAD init: hw_rate=%d channels=%d vad_frame_samples=%d",
        hw_rate, hw_channels, frame_samples,
    )

    robot.media.start_recording()
    try:
        return _run_vad_loop(
            robot, vad,
            hw_rate, hw_channels, hw_frame_samples,
            frame_samples, silence_frames_needed,
            min_speech_frames, max_s,
        )
    finally:
        robot.media.stop_recording()


def _run_vad_loop(
    robot, vad, hw_rate, hw_channels, hw_frame_samples,
    frame_samples, silence_frames_needed, min_speech_frames, max_s,
) -> np.ndarray | None:
    buffer: list[np.ndarray] = []   # accumulated 16 kHz float32 frames
    ring = collections.deque(maxlen=silence_frames_needed)
    speech_started = False
    speech_frame_count = 0
    silence_frame_count = 0
    deadline = time.monotonic() + max_s

    # Pre-speech padding: keep the last ~300 ms so we don't clip the start
    padding_frames = 10
    pre_speech: collections.deque[np.ndarray] = collections.deque(maxlen=padding_frames)

    while time.monotonic() < deadline:
        raw = robot.media.get_audio_sample()
        if raw is None or len(raw) == 0:
            time.sleep(0.005)
            continue

        # Ensure 1-D mono
        if raw.ndim > 1:
            raw = raw[:, 0]

        # Accumulate enough samples for one VAD frame
        ring.append(raw)
        chunk = np.concatenate(list(ring))
        if len(chunk) < hw_frame_samples:
            continue

        # Take exactly one frame worth from the front
        frame_hw = chunk[:hw_frame_samples]
        # resample to VAD rate
        frame_16k = _resample(frame_hw, hw_rate, _VAD_RATE)

        is_speech = False
        try:
            is_speech = vad.is_speech(_to_int16(frame_16k[:frame_samples]), _VAD_RATE)
        except Exception:
            pass

        if not speech_started:
            pre_speech.append(frame_16k)
            if is_speech:
                speech_started = True
                buffer.extend(list(pre_speech))
                speech_frame_count = 1
                silence_frame_count = 0
                logger.debug("Speech started")
        else:
            buffer.append(frame_16k)
            if is_speech:
                speech_frame_count += 1
                silence_frame_count = 0
            else:
                silence_frame_count += 1
                if silence_frame_count >= silence_frames_needed:
                    logger.debug(
                        "Speech ended: %d speech frames", speech_frame_count
                    )
                    break

    if not buffer or speech_frame_count < min_speech_frames:
        logger.debug("No valid utterance (speech_frames=%d)", speech_frame_count)
        return None

    utterance = np.concatenate(buffer)
    logger.debug("Utterance collected: %.2f s", len(utterance) / _VAD_RATE)
    return utterance
