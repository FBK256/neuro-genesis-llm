import json
from unittest.mock import patch
import urllib.request

import pytest

from mylm_omni.ollama_backend import OllamaLoopbackBackend


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        raw = json.dumps(self.payload).encode("utf-8")
        return raw if size < 0 else raw[:size]


class FakeOpener:
    def __init__(self):
        self.request = None
        self.timeout = None

    def open(self, request, timeout):
        self.request = request
        self.timeout = timeout
        return FakeResponse({
            "message": {"content": " local answer "},
            "prompt_eval_count": 11,
            "eval_count": 4,
            "done_reason": "stop",
        })


def test_ollama_backend_rejects_non_loopback_urls():
    with pytest.raises(ValueError, match="loopback"):
        OllamaLoopbackBackend("model", base_url="http://example.com:11434")
    with pytest.raises(ValueError, match="origin only"):
        OllamaLoopbackBackend("model", base_url="http://127.0.0.1:11434/api")


def test_ollama_backend_disables_proxies_and_records_token_stats(tmp_path):
    image = tmp_path / "screen.png"
    image.write_bytes(b"image-bytes")
    opener = FakeOpener()
    with patch(
        "mylm_omni.ollama_backend.urllib.request.build_opener",
        return_value=opener,
    ) as build_opener:
        backend = OllamaLoopbackBackend("local-model")
        answer = backend.complete(
            "What is in this picture?",
            system="Describe accurately.",
            image_paths=[image],
            max_new_tokens=77,
        )

    assert answer == "local answer"
    proxy_handler = build_opener.call_args.args[0]
    assert isinstance(proxy_handler, urllib.request.ProxyHandler)
    assert proxy_handler.proxies == {}
    request_payload = json.loads(opener.request.data.decode("utf-8"))
    assert request_payload["stream"] is False
    assert request_payload["options"]["num_predict"] == 77
    assert request_payload["messages"][-1]["images"]
    assert backend.last_call_stats["eval_count"] == 4


def test_ollama_backend_rejects_remote_images():
    backend = OllamaLoopbackBackend("local-model")
    with pytest.raises(ValueError, match="remote image"):
        backend.complete("summarize", image_paths=["https://example.org/img.png"])



def test_ollama_backend_explicitly_unloads_model_when_router_switches(tmp_path):
    opener = FakeOpener()
    backend = OllamaLoopbackBackend("local-model", keep_alive="5m")
    with patch(
        "mylm_omni.ollama_backend.urllib.request.build_opener",
        return_value=opener,
    ):
        assert backend.complete("hello") == "local answer"
        assert backend._model_may_be_loaded is True
        backend.unload()
    assert backend._model_may_be_loaded is False
    assert opener.request.full_url.endswith("/api/generate")
    payload = json.loads(opener.request.data.decode("utf-8"))
    assert payload["model"] == "local-model"
    assert payload["keep_alive"] == 0
