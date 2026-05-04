"""Home Assistant REST API client.

Provides:
  - Entity state querying (to inject context into the LLM prompt)
  - Service calls (to execute actions the LLM decides on)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .llm import HASSAction

logger = logging.getLogger(__name__)


class HASSClient:
    def __init__(self, cfg: dict[str, Any]) -> None:
        hass_cfg = cfg.get("hass", {})
        self._enabled: bool = hass_cfg.get("enabled", False)
        self._url: str = hass_cfg.get("url", "http://homeassistant.local:8123").rstrip("/")
        self._token: str = hass_cfg.get("token", "")
        self._exposed: list[str] = hass_cfg.get("exposed_entities", []) or []
        self._timeout: int = hass_cfg.get("request_timeout", 5)
        self._session = None

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self._token)

    # ------------------------------------------------------------------
    # Context for LLM
    # ------------------------------------------------------------------

    def build_context(self) -> str:
        """Return a short description of exposed entity states for the LLM."""
        if not self.enabled or not self._exposed:
            return ""

        lines: list[str] = []
        for entity_id in self._exposed:
            try:
                state = self.get_state(entity_id)
                friendly = state.get("attributes", {}).get("friendly_name", entity_id)
                val = state.get("state", "unknown")
                attrs = _format_attrs(state.get("attributes", {}))
                lines.append(f"  {friendly} ({entity_id}): {val}{attrs}")
            except Exception as exc:
                logger.debug("Could not fetch state for %s: %s", entity_id, exc)
                lines.append(f"  {entity_id}: unavailable")

        if not lines:
            return ""

        return "Current home state:\n" + "\n".join(lines) + "\n\n"

    # ------------------------------------------------------------------
    # Entity state
    # ------------------------------------------------------------------

    def get_state(self, entity_id: str) -> dict[str, Any]:
        resp = self._request("GET", f"/api/states/{entity_id}")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Service calls
    # ------------------------------------------------------------------

    def call_action(self, action: "HASSAction") -> bool:
        """Execute a HASSAction returned by the LLM. Returns True on success."""
        if not self.enabled:
            logger.info("HASS disabled – skipping action: %s/%s", action.domain, action.service)
            return False

        payload: dict[str, Any] = {}
        if action.entity_id:
            payload["entity_id"] = action.entity_id
        payload.update(action.data or {})

        try:
            resp = self._request(
                "POST",
                f"/api/services/{action.domain}/{action.service}",
                json=payload,
            )
            resp.raise_for_status()
            logger.info(
                "HASS action executed: %s/%s %s %s",
                action.domain, action.service, action.entity_id, action.data,
            )
            return True
        except Exception as exc:
            logger.error("HASS action failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs):
        import requests
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            })
        return self._session.request(
            method,
            self._url + path,
            timeout=self._timeout,
            **kwargs,
        )


def _format_attrs(attrs: dict) -> str:
    interesting = {
        k: v for k, v in attrs.items()
        if k in ("brightness", "color_temp", "temperature", "humidity",
                  "current_temperature", "hvac_mode", "volume_level", "media_title")
    }
    if not interesting:
        return ""
    parts = [f"{k}={v}" for k, v in interesting.items()]
    return f" ({', '.join(parts)})"
