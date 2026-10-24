"""Ollama backend restricted to a loopback endpoint.

Only a localhost Ollama endpoint is accepted; this prevents a private prompt or image from
being sent to an arbitrary remote server by misconfiguration. The local Ollama daemon itself
can still have external model-pull behavior, so pre-pull models and block daemon egress for a
strict offline deployment.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class OllamaLoopbackBackend:
    """Local Ollama chat API adapter supporting text and local images."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 180.0,
        max_image_bytes: int = 20_000_000,
        keep_alive: str | int = "5m",
    ) -> None:
        if not model.strip():
            raise ValueError("Ollama model name must not be empty")
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("OllamaLoopbackBackend only permits http://localhost or loopback IPs")
        if parsed.username or parsed.password:
            raise ValueError("credentials must not be embedded in the Ollama base URL")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an origin only, e.g. http://127.0.0.1:11434")
        if timeout_seconds <= 0 or max_image_bytes < 1:
            raise ValueError("timeout_seconds and max_image_bytes must be positive")
        if isinstance(keep_alive, int):
            if keep_alive < 0:
                raise ValueError("keep_alive integer must be >= 0")
        elif not isinstance(keep_alive, str) or not keep_alive.strip():
            raise ValueError("keep_alive must be a non-empty duration string or non-negative integer")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_image_bytes = max_image_bytes
        self.keep_alive = keep_alive
        self._model_may_be_loaded = False

    @staticmethod
    def _local_image_b64(path: str | Path, max_bytes: int) -> str:
        raw_path = str(path)
        if raw_path.startswith(("http://", "https://")):
            raise ValueError("remote image URLs are not accepted by the local-only backend")
        image_path = Path(path).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        size = image_path.stat().st_size
        if size > max_bytes:
            raise ValueError(f"image is {size} bytes; configured limit is {max_bytes}")
        return base64.b64encode(image_path.read_bytes()).decode("ascii")

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        image_paths: list[str | Path] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> str:
        """Call /api/chat and return message content. No data is sent to non-loopback hosts."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if max_new_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("invalid generation settings")
        images = [
            self._local_image_b64(path, self.max_image_bytes) for path in (image_paths or [])
        ]
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        user_message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user_message["images"] = images
        messages.append(user_message)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        request = urllib.request.Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        # A timed-out request may still have loaded weights in the daemon; remember that the
        # external model could be resident, and attempt explicit unload before switching models.
        self._model_may_be_loaded = True
        # Do not inherit HTTP_PROXY/HTTPS_PROXY environment variables for loopback inference.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(4_000_001)
                if len(raw) > 4_000_000:
                    raise RuntimeError("Ollama response exceeds the 4 MB safety limit")
                result = json.loads(raw.decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach local Ollama at {self.base_url}; check that the daemon is running"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON") from exc

        message = result.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise RuntimeError("Ollama response lacks message.content")
        self.last_call_stats = {
            "model": self.model,
            "elapsed_seconds": time.monotonic() - started,
            "prompt_eval_count": result.get("prompt_eval_count"),
            "eval_count": result.get("eval_count"),
            "done_reason": result.get("done_reason"),
        }
        return content.strip()

    def unload(self) -> None:
        """Ask the local Ollama daemon to release this model before another route is loaded."""
        if not self._model_may_be_loaded:
            return
        payload = {
            "model": self.model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
        request = urllib.request.Request(
            self.base_url + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(1_000_001)
                if len(raw) > 1_000_000:
                    raise RuntimeError("Ollama unload response exceeds the 1 MB safety limit")
                json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            # Let the router fail closed rather than load another heavy model on uncertain VRAM.
            raise RuntimeError(
                f"could not confirm unload of Ollama model {self.model!r}"
            ) from exc
        self._model_may_be_loaded = False
