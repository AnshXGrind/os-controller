"""ai_pc_agent/ai/ollama_client.py

Full-featured Ollama HTTP client.
"""

from __future__ import annotations
import json
import re
import time
from typing import Any, Generator, Iterator

import requests

from ai_pc_agent.utils import config
from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.ollama")


class OllamaClient:
    """Thin wrapper around the Ollama REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        self.base_url = (base_url or config.get("OLLAMA_BASE_URL")).rstrip("/")
        self.model    = model or config.get("OLLAMA_MODEL")
        self.timeout  = timeout or int(config.get("OLLAMA_TIMEOUT", 60))
        self._available: bool | None = None

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if not models:
                logger.warning("Ollama reachable but no models found.")
                self._available = False
                return False
            if self.model not in models:
                logger.warning(
                    "Model '%s' not found. Available: %s. Switching to '%s'.",
                    self.model, models, models[0],
                )
                self.model = models[0]
            self._available = True
            logger.info("Ollama available. Model: %s", self.model)
            return True
        except Exception as exc:
            logger.warning("Ollama not reachable: %s", exc)
            self._available = False
            return False

    def reset_availability(self):
        self._available = None

    def list_models(self) -> list[str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    # ── Core generation ───────────────────────────────────────────────────────

    def ask(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        """Single-shot text generation. Returns response string."""
        payload: dict[str, Any] = {
            "model":  model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or float(config.get("LLM_TEMPERATURE", 0.2)),
                "num_predict": max_tokens or int(config.get("LLM_MAX_TOKENS", 512)),
            },
        }
        if system:
            payload["system"] = system
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as exc:
            logger.error("Ollama ask() error: %s", exc)
            return ""

    def ask_json(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
    ) -> dict | None:
        """Ask and parse first JSON object from response."""
        raw = self.ask(prompt, system=system, model=model, temperature=0.0)
        if not raw:
            return None
        # Strip markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
        # Extract first {...} block
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None

    def ask_stream(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
    ) -> Generator[str, None, None]:
        """Streaming text generation; yields partial tokens."""
        payload: dict[str, Any] = {
            "model":  model or self.model,
            "prompt": prompt,
            "stream": True,
        }
        if system:
            payload["system"] = system
        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        yield data.get("response", "")
                        if data.get("done"):
                            break
        except Exception as exc:
            logger.error("Ollama stream error: %s", exc)

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Multi-turn chat endpoint."""
        payload: dict[str, Any] = {
            "model":    model or self.model,
            "messages": messages,
            "stream":   False,
            "options":  {
                "temperature": temperature or float(config.get("LLM_TEMPERATURE", 0.2)),
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()
        except Exception as exc:
            logger.error("Ollama chat() error: %s", exc)
            return ""
