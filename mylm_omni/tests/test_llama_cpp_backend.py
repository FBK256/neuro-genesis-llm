import base64
import json
from unittest.mock import patch

import pytest

from mylm_omni.llama_cpp_backend import LlamaCppLoopbackBackend


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeOpener:
    def __init__(self, payload):
        self.payload = payload
        self.request = None
        self.timeout = None

    def open(self, request, timeout):
        self.request = request
        self.timeout = timeout
        return FakeResponse(self.payload)


def test_backend_rejects_non_loopback_and_invalid_endpoint_paths():
    with pytest.raises(ValueError, match="loopback"):
        LlamaCppLoopbackBackend(base_url="http://example.com:8080/v1")
    with pytest.raises(ValueError, match="/v1"):
        LlamaCppLoopbackBackend(base_url="http://127.0.0.1:8080")


def test_backend_sends_text_and_local_image_to_loopback_api(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"local-image-bytes")
    opener = FakeOpener({
        "choices": [{"message": {"content": '{"type":"final","content":"ok"}'}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    })
    backend = LlamaCppLoopbackBackend(model="test-model")
    with patch("mylm_omni.llama_cpp_backend.urllib.request.build_opener", return_value=opener):
        answer = backend.complete(
            "Describe the image",
            system="Return JSON",
            image_paths=[image],
            max_new_tokens=100,
            temperature=0.0,
        )
    assert answer == '{"type":"final","content":"ok"}'
    request = json.loads(opener.request.data.decode("utf-8"))
    assert request["model"] == "test-model"
    assert request["max_tokens"] == 100
    image_part = request["messages"][-1]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].endswith(base64.b64encode(b"local-image-bytes").decode())
    assert backend.last_call_stats["completion_tokens"] == 7


def test_backend_rejects_remote_image_urls_and_too_many_images(tmp_path):
    backend = LlamaCppLoopbackBackend(max_images=1)
    with pytest.raises(ValueError, match="remote image"):
        backend.complete("inspect", image_paths=["https://example.com/image.png"])
    image = tmp_path / "one.png"
    image.write_bytes(b"x")
    with pytest.raises(ValueError, match="at most"):
        backend.complete("inspect", image_paths=[image, image])
