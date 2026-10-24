"""OpenAI-compatible llama.cpp server backend, restricted to loopback endpoints."""
from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class LlamaCppLoopbackBackend:
    """Call llama-server /v1/chat/completions using local text and optional local images."""

    def __init__(
        self,
        model: str = "local-vlm",
        *,
        base_url: str = "http://127.0.0.1:8080/v1",
        timeout_seconds: float = 180.0,
        max_image_bytes: int = 10_000_000,
        max_images: int = 4,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("llama.cpp endpoint must be on localhost/loopback over HTTP")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query, or fragment")
        if not parsed.path.rstrip("/").endswith("/v1"):
            raise ValueError("base_url must include the /v1 path")
        if not model.strip() or timeout_seconds <= 0 or max_image_bytes < 1 or max_images < 1:
            raise ValueError("model, timeout, image size and image count must be valid")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_image_bytes = max_image_bytes
        self.max_images = max_images
        self.last_call_stats: dict[str, Any] = {}

    def _image_part(self, image_path: str | Path) -> dict[str, Any]:
        if str(image_path).startswith(("http://", "https://")):
            raise ValueError("remote image URLs are blocked; provide local image files")
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size > self.max_image_bytes:
            raise ValueError(f"image exceeds the {self.max_image_bytes}-byte limit")
        media_type = mimetypes.guess_type(path.name)[0]
        if media_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("only local JPEG, PNG, and WebP images are accepted")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{encoded}"},
        }

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
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if max_new_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("invalid generation settings")
        image_paths = image_paths or []
        if len(image_paths) > self.max_images:
            raise ValueError(f"at most {self.max_images} images are permitted")

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(self._image_part(path) for path in image_paths)
        messages.append({"role": "user", "content": content if image_paths else prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Disable environment proxy inheritance so localhost inference cannot be routed outward.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        started = time.monotonic()
        try:
            with opener.open(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach llama.cpp at {self.base_url}; start llama-server first"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("llama.cpp returned invalid JSON") from exc

        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llama.cpp response has no choices")
        message = choices[0].get("message", {})
        content_value = message.get("content")
        if not isinstance(content_value, str):
            raise RuntimeError("llama.cpp response lacks text message content")
        usage = result.get("usage", {})
        self.last_call_stats = {
            "model": self.model,
            "elapsed_seconds": time.monotonic() - started,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
        return content_value.strip()
