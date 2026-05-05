"""Main conversation loop – ties together VAD, STT, LLM, TTS, and robot behaviour.

Implements ReachyMiniApp so it can be launched via the standard SDK CLI:
    reachy-mini-app spark_voice.conversation:SparkVoiceApp

or run directly:
    python -m spark_voice.conversation
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from reachy_mini import ReachyMini, ReachyMiniApp

from .config import load as load_cfg
from .hass import HASSClient
from .llm import SparkLLM
from . import robot as robot_behavior
from .stt import WhisperSTT
from .tts import PiperTTS
from .vad import collect_utterance
from .wake_word import wait_for_wake_word

logger = logging.getLogger(__name__)


class SparkVoiceApp(ReachyMiniApp):
    """Continuous voice assistant that uses the Spark for LLM inference."""

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__()
        self._cfg: dict[str, Any] = load_cfg(config_path)
        self._stt = WhisperSTT(self._cfg)   # pre-loads Whisper model at startup
        self._tts = PiperTTS(self._cfg)
        self._llm = SparkLLM(self._cfg)
        self._hass = HASSClient(self._cfg)

    # ------------------------------------------------------------------
    # ReachyMiniApp interface
    # ------------------------------------------------------------------

    def run(self, robot: ReachyMini, stop_event: threading.Event) -> None:  # type: ignore[override]
        logger.info("Spark Voice Assistant started.")
        self._check_spark()
        robot.wake_up()
        robot_behavior.go_idle(robot, self._cfg)

        while not stop_event.is_set():
            try:
                self._conversation_turn(robot, stop_event)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.error("Error in conversation turn: %s", exc, exc_info=True)
                robot_behavior.on_error(robot, self._cfg)
                time.sleep(1.0)

        robot.goto_sleep()
        logger.info("Spark Voice Assistant stopped.")

    # ------------------------------------------------------------------
    # Startup diagnostics
    # ------------------------------------------------------------------

    def _check_spark(self) -> None:
        """Ping the Spark LLM endpoint and log reachability."""
        import urllib.request
        import urllib.error
        spark_cfg = self._cfg.get("spark", {})
        api_base = spark_cfg.get("api_base", "")
        models_url = api_base.rstrip("/").replace("/v1", "") + "/api/tags"
        try:
            with urllib.request.urlopen(models_url, timeout=5) as r:
                logger.info("Spark reachable at %s (HTTP %d)", api_base, r.status)
        except urllib.error.URLError as exc:
            logger.warning(
                "Cannot reach Spark at %s: %s  "
                "– LLM calls will fail until it is reachable.",
                api_base, exc,
            )
        except Exception as exc:
            logger.warning("Spark connectivity check failed: %s", exc)

    # ------------------------------------------------------------------
    # Single conversation turn
    # ------------------------------------------------------------------

    def _conversation_turn(
        self, robot: ReachyMini, stop_event: threading.Event
    ) -> None:
        # 1. Wait for wake word (no-op when wake_word.enabled is false)
        if stop_event.is_set():
            return
        detected = wait_for_wake_word(robot, self._cfg)
        if not detected or stop_event.is_set():
            return

        robot_behavior.on_listen_start(robot, self._cfg)

        # Speak a short ready cue so the user knows to ask, and so the
        # wake-word audio tail (still in the mic buffer) has time to clear.
        ww_cfg = self._cfg.get("wake_word", {})
        ready_phrase = ww_cfg.get("ready_phrase", "Yes?")
        if ready_phrase:
            self._speak_quick(ready_phrase, robot)
        else:
            time.sleep(0.4)

        logger.info("Listening …")
        utterance = collect_utterance(robot, self._cfg)

        if utterance is None or stop_event.is_set():
            robot_behavior.go_idle(robot, self._cfg)
            return

        logger.info("Utterance collected: %.2f s – transcribing …", len(utterance) / 16_000)
        robot_behavior.on_listen_end(robot, self._cfg)

        # 2. Transcribe
        robot_behavior.on_thinking(robot, self._cfg)
        try:
            text = self._stt.transcribe(utterance)
        except Exception as exc:
            logger.error("STT failed: %s", exc)
            robot_behavior.on_error(robot, self._cfg)
            return

        if not text:
            logger.info("Empty transcription – ignoring utterance.")
            robot_behavior.go_idle(robot, self._cfg)
            return

        logger.info("Heard: %r", text)

        # 3. Build HASS context and query LLM
        hass_context = self._hass.build_context() if self._hass.enabled else ""
        logger.info("Querying Spark LLM …")

        try:
            spoken, hass_action = self._llm.chat(text, context_prefix=hass_context)
        except Exception as exc:
            logger.error("LLM failed: %s", exc)
            robot_behavior.on_error(robot, self._cfg)
            return

        logger.info("Reachy: %r", spoken)

        # 4. Execute HASS action (if any) in background while TTS runs
        if hass_action:
            threading.Thread(
                target=self._hass.call_action,
                args=(hass_action,),
                daemon=True,
            ).start()

        # 5. Speak
        if spoken:
            self._speak(spoken, robot)

        robot_behavior.go_idle(robot, self._cfg)

    # ------------------------------------------------------------------
    # TTS + playback
    # ------------------------------------------------------------------

    def _speak_quick(self, text: str, robot: ReachyMini) -> None:
        """Short acknowledgment phrase via espeak (always available, instant).

        Used for wake-word ready cues where we can't afford piper latency.
        Falls back silently to a 400 ms sleep if espeak fails.
        """
        import subprocess
        import io
        import wave
        import numpy as np
        from scipy.signal import resample_poly
        import math

        try:
            tts_cfg = self._cfg.get("tts", {})
            espeak_voice = tts_cfg.get("espeak_voice", "en-us")
            espeak_speed = int(tts_cfg.get("espeak_speed", 150))

            proc = subprocess.run(
                ["espeak-ng", "-v", espeak_voice, "-s", str(espeak_speed), "--stdout", text],
                capture_output=True, timeout=5,
            )
            if proc.returncode != 0 or not proc.stdout:
                time.sleep(0.4)
                return

            # Parse WAV from espeak stdout
            with wave.open(io.BytesIO(proc.stdout)) as wf:
                n = wf.getnframes()
                raw = wf.readframes(n)
                src_rate = wf.getframerate()

            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            out_rate = robot.media.get_output_audio_samplerate()
            if out_rate > 0 and out_rate != src_rate:
                g = math.gcd(src_rate, out_rate)
                audio = resample_poly(audio, out_rate // g, src_rate // g).astype(np.float32)
            else:
                out_rate = src_rate
            out_channels = robot.media.get_output_channels()
            if out_channels > 1:
                audio = np.tile(audio[:, np.newaxis], (1, out_channels))

            duration_s = audio.shape[0] / out_rate
            robot.media.start_playing()
            robot.media.push_audio_sample(audio)
            time.sleep(duration_s + 0.2)
            robot.media.stop_playing()
        except Exception as exc:
            logger.debug("Ready cue failed (%s) – using pause instead.", exc)
            time.sleep(0.4)

    def _speak(self, text: str, robot: ReachyMini) -> None:
        robot_behavior.on_speaking_start(robot, self._cfg)
        try:
            audio = self._tts.synthesize(text, robot)
            if audio is None or len(audio) == 0:
                logger.warning("TTS returned empty audio – nothing to play.")
                return
            out_rate = robot.media.get_output_audio_samplerate()
            # push_audio_sample enqueues asynchronously; sleep for playback duration
            # before stopping so the audio isn't cut off
            duration_s = audio.shape[0] / out_rate if out_rate > 0 else 2.0
            logger.debug("Speaking %.2f s of audio …", duration_s)
            robot.media.start_playing()
            robot.media.push_audio_sample(audio)
            time.sleep(duration_s + 0.3)  # +0.3 s buffer for stream flush
            robot.media.stop_playing()
        except Exception as exc:
            logger.error("TTS/playback failed: %s", exc, exc_info=True)
            robot_behavior.on_error(robot, self._cfg)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Reachy Mini Spark Voice Assistant")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--reset-history", action="store_true",
                        help="Clear LLM conversation history and exit")
    args = parser.parse_args()

    # force=True overrides any handler the SDK already installed on the root logger.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        force=True,
    )
    # Suppress the websocket frame-level chatter regardless of our level.
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("websockets.client").setLevel(logging.WARNING)

    app = SparkVoiceApp(config_path=args.config)
    app.wrapped_run()


if __name__ == "__main__":
    main()
