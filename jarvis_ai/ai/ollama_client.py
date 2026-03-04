"""
ai/ollama_client.py

HTTP client for the locally-running Ollama inference server.

Ollama exposes a REST API at http://localhost:11434.
This module wraps the /api/generate and /api/chat endpoints so the
rest of the codebase never has to deal with raw HTTP.

Quick-start:
    client = OllamaClient()
    response = client.ask("What is the capital of France?")
    print(response)   # "Paris"

Streaming is supported via ask_stream() for long responses that
should be displayed word-by-word.
"""

import json
import logging
import time
from typing import Generator, Optional

import requests

from jarvis_ai.utils import config

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Thin wrapper around the Ollama HTTP API.

    Args:
        base_url: Ollama server base URL  (default from config).
        model:    Model name to use       (default from config).
        timeout:  HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str   = None,
        model:    str   = None,
        timeout:  int   = None,
    ):
        self.base_url = (base_url or config.get("OLLAMA_BASE_URL")).rstrip("/")
        self.model    = model   or config.get("OLLAMA_MODEL")
        self.timeout  = timeout or config.get("OLLAMA_TIMEOUT", 30)

        self._generate_url = f"{self.base_url}/api/generate"
        self._chat_url     = f"{self.base_url}/api/chat"
        self._tags_url     = f"{self.base_url}/api/tags"

        logger.info(
            f"OllamaClient ready → {self.base_url}  model={self.model}"
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """
        Return True if the Ollama server is reachable and the configured
        model is available locally.
        """
        try:
            resp = requests.get(self._tags_url, timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model not in models:
                logger.warning(
                    f"Model '{self.model}' not found locally. "
                    f"Available: {models}. "
                    f"Run: ollama pull {self.model}"
                )
                # Use the first available model as a fallback
                if models:
                    logger.warning(f"Falling back to first available model: '{models[0]}'")
                    self.model = models[0]
                    return True
                return False
            return True
        except requests.exceptions.ConnectionError:
            logger.error(
                "Cannot reach Ollama. Make sure the server is running: "
                "ollama serve"
            )
            return False
        except Exception as exc:
            logger.error(f"OllamaClient.is_available: {exc}")
            return False

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def ask(
        self,
        prompt:      str,
        system:      str   = "",
        temperature: float = 0.1,
        max_tokens:  int   = 512,
    ) -> Optional[str]:
        """
        Send a single prompt and return the complete response string.

        Args:
            prompt:      The user message / instruction.
            system:      Optional system prompt prepended to the conversation.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens:  Maximum tokens in the response.

        Returns:
            Generated text string, or None on error.
        """
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        try:
            t0   = time.monotonic()
            resp = requests.post(
                self._generate_url,
                json    = payload,
                timeout = self.timeout,
            )
            resp.raise_for_status()
            data     = resp.json()
            elapsed  = time.monotonic() - t0
            text     = data.get("response", "").strip()
            logger.debug(
                f"OllamaClient: {len(text)} chars in {elapsed:.2f}s  "
                f"[{data.get('eval_count', '?')} tokens]"
            )
            return text

        except requests.exceptions.Timeout:
            logger.error(
                f"Ollama request timed out after {self.timeout}s. "
                "Try a smaller model or increase OLLAMA_TIMEOUT in config."
            )
        except requests.exceptions.ConnectionError:
            logger.error("Ollama server is not running. Start it with: ollama serve")
        except Exception as exc:
            logger.error(f"OllamaClient.ask error: {exc}")

        return None

    def ask_stream(
        self,
        prompt:      str,
        system:      str   = "",
        temperature: float = 0.1,
    ) -> Generator[str, None, None]:
        """
        Stream the response token-by-token.

        Yields:
            Individual text chunks as they arrive from Ollama.

        Usage:
            for chunk in client.ask_stream("Tell me a joke"):
                print(chunk, end="", flush=True)
        """
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        try:
            with requests.post(
                self._generate_url,
                json    = payload,
                timeout = self.timeout,
                stream  = True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line).get("response", "")
                    if chunk:
                        yield chunk

        except requests.exceptions.ConnectionError:
            logger.error("Ollama server not running.")
        except Exception as exc:
            logger.error(f"OllamaClient.ask_stream error: {exc}")

    # ------------------------------------------------------------------
    # Chat (multi-turn) helper
    # ------------------------------------------------------------------

    def chat(
        self,
        messages:    list[dict],
        temperature: float = 0.1,
        max_tokens:  int   = 512,
    ) -> Optional[str]:
        """
        Multi-turn chat using the /api/chat endpoint.

        Args:
            messages: List of {"role": "user"|"assistant"|"system",
                               "content": str} dicts.
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens in response.

        Returns:
            Assistant reply string, or None on error.
        """
        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            "options":  {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            resp = requests.post(
                self._chat_url,
                json    = payload,
                timeout = self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        except Exception as exc:
            logger.error(f"OllamaClient.chat error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Convenience: JSON extraction
    # ------------------------------------------------------------------

    def ask_json(
        self,
        prompt: str,
        system: str = "",
    ) -> Optional[dict]:
        """
        Ask the LLM and attempt to parse the response as JSON.

        Strips markdown code fences if present.

        Returns:
            Parsed dict, or None on error.
        """
        raw = self.ask(prompt, system=system, temperature=0.0, max_tokens=256)
        if not raw:
            return None

        # Strip ```json … ``` fences
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text  = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        # Find the first { … } block
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.debug(f"OllamaClient: no JSON object in response: {text!r}")
            return None

        json_str = text[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.debug(f"OllamaClient: JSON parse error – {exc}  raw={json_str!r}")
            return None
