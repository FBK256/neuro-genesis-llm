from mylm_omni.backends import LocalHFBackend


class FakeProcessor:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "formatted-qwen-prompt"


def test_qwen3vl_template_receives_actual_pil_images():
    backend = LocalHFBackend.__new__(LocalHFBackend)
    backend.model_type = "qwen3-vl"
    backend.processor = FakeProcessor()
    image = object()

    text = backend._format_messages("what is this?", "be accurate", [image])

    assert text == "formatted-qwen-prompt"
    user_content = backend.processor.messages[-1]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[0]["image"] is image
    assert user_content[1] == {"type": "text", "text": "what is this?"}
