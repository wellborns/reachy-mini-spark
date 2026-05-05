"""Voice Activity Detection – collects a complete utterance from the mic.

Uses webrtcvad for lightweight, real-time speech/silence discrimination.
Audio from the Reachy Mini SDK is float32 at whatever rate the hardware
reports; we resample to 16 kHz before feeding webrtcvad.
"""

from __future__ import annotations

import collections
import logging
import math
import time
from typing import TYPE_CHECKING

import numpy as np
import webrtcvad
from scipy.signal import resample_poly

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)

_VAD_RATE = 16_000  # webrtcvad only accepts 8 / 16 / 32 kHz


def _to_int16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    g = math.gcd(src_rate, dst_rate)
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)


def collect_utterance(robot: "ReachyMini", cfg: dict) -> np.ndarray | None:
    """Record until an utterance is complete; return float32 16 kHz mono array.

    Returns None on timeout or if no speech was detected.
    """
    vad_cfg = cfg.get("vad", {})
    mode: int = vad_cfg.get("mode", 3)
    frame_ms: int = vad_cfg.get("frame_duration_ms", 30)
    silence_ms: int = vad_cfg.get("silence_duration_ms", 900)
    min_speech_ms: int = vad_cfg.get("min_speech_duration_ms", 600)
    max_s: float = vad_cfg.get("max_utterance_duration_s", 30)
    rms_threshold: float = float(vad_cfg.get("rms_threshold", 0.008))

    vad = webrtcvad.Vad(mode)

    # Number of 16 kHz samples in one VAD frame (must be 10/20/30 ms)
    vad_frame_samples = (_VAD_RATE * frame_ms) // 1000
    silence_frames_needed = silence_ms // frame_ms
    min_speech_frames = min_speech_ms // frame_ms

    hw_rate: int = robot.media.get_input_audio_samplerate()
    hw_channels: int = robot.media.get_input_channels()

    logger.info(
        "VAD ready: hw_rate=%d ch=%d  mode=%d  rms_gate=%.4f  silence_thresh=%dms",
        hw_rate, hw_channels, mode, rms_threshold, silence_ms,
    )

    robot.media.start_recording()
    try:
        return _run_vad_loop(
            robot, vad, hw_rate, hw_channels,
            vad_frame_samples, silence_frames_needed, min_speech_frames, max_s,
            rms_threshold,
        )
    finally:
        robot.media.stop_recording()


def _run_vad_loop(
    robot, vad, hw_rate, hw_channels,
    vad_frame_samples, silence_frames_needed, min_speech_frames, max_s,
    rms_threshold: float,
) -> np.ndarray | None:
    """Core VAD accumulator.

    Maintains a flat float32 sample buffer at _VAD_RATE.  Each iteration
    pulls whatever `get_audio_sample()` returns, resamples, appends to the
    buffer, then slices off as many complete VAD frames as are available.
    This avoids the stale-sample bug where the same prefix was reprocessed
    on every iteration.

    An RMS energy gate is applied before webrtcvad: frames below the threshold
    are treated as silence regardless of webrtcvad's verdict, which filters
    low-level ambient noise that webrtcvad may misclassify as speech.
    """
    pending_arr = np.empty(0, dtype=np.float32)

    utterance_frames: list[np.ndarray] = []
    pre_speech: collections.deque[np.ndarray] = collections.deque(maxlen=10)  # ~300 ms

    speech_started = False
    speech_frame_count = 0
    silence_frame_count = 0
    deadline = time.monotonic() + max_s
    logged_waiting = False

    while time.monotonic() < deadline:
        raw = robot.media.get_audio_sample()
        if raw is None or len(raw) == 0:
            time.sleep(0.005)
            continue

        if not logged_waiting:
            logger.debug("Audio flowing from mic.")
            logged_waiting = True

        # Ensure 1-D mono float32
        chunk = np.asarray(raw, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = chunk[:, 0]

        # Resample to VAD rate and append to pending buffer
        chunk_16k = _resample(chunk, hw_rate, _VAD_RATE)
        pending_arr = np.concatenate([pending_arr, chunk_16k])

        # Drain as many complete VAD frames as we have
        while len(pending_arr) >= vad_frame_samples:
            frame = pending_arr[:vad_frame_samples]
            pending_arr = pending_arr[vad_frame_samples:]

            # RMS energy gate: quiet frames can't be speech
            rms = float(np.sqrt(np.mean(frame ** 2)))
            is_speech = False
            if rms >= rms_threshold:
                try:
                    is_speech = vad.is_speech(_to_int16(frame), _VAD_RATE)
                except Exception:
                    pass

            if not speech_started:
                pre_speech.append(frame)
                if is_speech:
                    speech_started = True
                    utterance_frames.extend(list(pre_speech))
                    speech_frame_count = 1
                    silence_frame_count = 0
                    logger.info(
                        "Speech detected (rms=%.4f) – recording utterance …", rms
                    )
            else:
                utterance_frames.append(frame)
                if is_speech:
                    speech_frame_count += 1
                    silence_frame_count = 0
                else:
                    silence_frame_count += 1
                    if silence_frame_count >= silence_frames_needed:
                        if speech_frame_count < min_speech_frames:
                            logger.debug(
                                "Utterance too short (%d frames, need %d) – discarding",
                                speech_frame_count, min_speech_frames,
                            )
                            return None
                        logger.debug(
                            "End of utterance: %d speech frames, %d silence frames",
                            speech_frame_count, silence_frame_count,
                        )
                        keep = speech_frame_count + min(silence_frame_count, 5)
                        return np.concatenate(utterance_frames[-keep:]) \
                            if keep < len(utterance_frames) \
                            else np.concatenate(utterance_frames)

    if not utterance_frames or speech_frame_count < min_speech_frames:
        logger.debug("No valid utterance (speech_frames=%d)", speech_frame_count)
        return None

    result = np.concatenate(utterance_frames)
    logger.debug("Utterance collected: %.2f s", len(result) / _VAD_RATE)
    return result
