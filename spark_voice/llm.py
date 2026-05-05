"""Spark LLM client – OpenAI-compatible API (Ollama, LM Studio, etc.)

Maintains a rolling conversation history and returns (spoken_text, hass_action).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# [HASS:domain/service:entity_id:json_data]
_HASS_TAG_RE = re.compile(
    r"\[HASS:(?P<domain>[^/]+)/(?P<service>[^:]+):(?P<entity>[^:\]]*):(?P<data>[^\]]*)\]"
)


@dataclass
class HASSAction:
    domain: str
    service: str
    entity_id: str
    data: dict[str, Any]


class SparkLLM:
    def __init__(self, cfg: dict[str, Any]) -> None:
        spark = cfg.get("spark", {})
        self._api_base: str = spark.get("api_base", "http://spark.local:11434/v1")
        self._api_key: str = spark.get("api_key", "ollama")
        self._model: str = spark.get("model", "llama3.2:3b")
        self._max_tokens: int = spark.get("max_tokens", 256)
        self._temperature: float = spark.get("temperature", 0.7)
        self._timeout: int = spark.get("timeout", 30)
        self._system_prompt: str = spark.get("system_prompt", "You are Reachy, a helpful robot assistant.")
        self._history: list[dict] = []
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai package not installed.  Run: pip install openai"
                ) from exc
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._api_base,
                timeout=self._timeout,
            )
        return self._client

    def chat(
        self,
        user_text: str,
        context_prefix: str = "",
    ) -> tuple[str, HASSAction | None]:
        """Send user_text; return (spoken_reply, optional_hass_action).

        context_prefix is prepended to the user message to inject HASS state.
        """
        client = self._get_client()

        user_content = f"{context_prefix}{user_text}" if context_prefix else user_text
        self._history.append({"role": "user", "content": user_content})

        messages = [{"role": "system", "content": self._system_prompt}] + self._history

        try:
            resp = client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception as exc:
            logger.error("LLM request failed: %s", exc)
            self._history.pop()  # don't pollute history on error
            raise

        raw = resp.choices[0].message.content or ""
        logger.debug("LLM raw response: %r", raw)

        hass_action = _extract_hass_action(raw)
        spoken = _HASS_TAG_RE.sub("", raw).strip()
        # Strip null-action markers the model emits instead of omitting the tag
        spoken = re.sub(r"\[\s*[Nn]one\s*\]", "", spoken).strip()

        self._history.append({"role": "assistant", "content": raw})
        self._trim_history()

        return spoken, hass_action

    def reset(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    def _trim_history(self, max_turns: int = 10) -> None:
        """Keep only the last max_turns of back-and-forth."""
        if len(self._history) > max_turns * 2:
            self._history = self._history[-(max_turns * 2):]


def _extract_hass_action(text: str) -> HASSAction | None:
    m = _HASS_TAG_RE.search(text)
    if not m:
        return None

    import json
    data_str = m.group("data").strip()
    try:
        data = json.loads(data_str) if data_str else {}
    except json.JSONDecodeError:
        logger.warning("Could not parse HASS data JSON: %r", data_str)
        data = {}

    return HASSAction(
        domain=m.group("domain"),
        service=m.group("service"),
        entity_id=m.group("entity"),
        data=data,
    )
