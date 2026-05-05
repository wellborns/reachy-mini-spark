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
        self._stt = WhisperSTT(self._cfg)
        self._tts = PiperTTS(self._cfg)
        self._llm = SparkLLM(self._cfg)
        self._hass = HASSClient(self._cfg)

    # ------------------------------------------------------------------
    # ReachyMiniApp interface
    # ------------------------------------------------------------------

    def run(self, robot: ReachyMini, stop_event: threading.Event) -> None:  # type: ignore[override]
        logger.info("Spark Voice Assistant started.")
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

        # Signal readiness: brief antenna perk + log
        logger.info("Listening …")
        robot_behavior.on_listen_start(robot, self._cfg)

        utterance = collect_utterance(robot, self._cfg)

        if utterance is None or stop_event.is_set():
            robot_behavior.go_idle(robot, self._cfg)
            return

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
            logger.debug("Empty transcription – ignoring.")
            robot_behavior.go_idle(robot, self._cfg)
            return

        logger.info("Heard: %r", text)

        # 3. Build HASS context and query LLM
        hass_context = self._hass.build_context() if self._hass.enabled else ""

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

    def _speak(self, text: str, robot: ReachyMini) -> None:
        robot_behavior.on_speaking_start(robot, self._cfg)
        try:
            audio = self._tts.synthesize(text, robot)
            robot.media.start_playing()
            robot.media.push_audio_sample(audio)
            robot.media.stop_playing()
        except Exception as exc:
            logger.error("TTS/playback failed: %s", exc)
            robot_behavior.on_error(robot, self._cfg)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Reachy Mini Spark Voice Assistant")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--reset-history", action="store_true",
                        help="Clear LLM conversation history and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    app = SparkVoiceApp(config_path=args.config)
    app.wrapped_run()


if __name__ == "__main__":
    main()
