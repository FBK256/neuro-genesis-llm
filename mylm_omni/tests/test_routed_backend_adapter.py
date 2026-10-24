from mylm_omni.model_router import RoutedBackendAdapter, infer_routing_task


class FakeRouter:
    def __init__(self):
        self.kwargs = None

    def complete(self, **kwargs):
        self.kwargs = kwargs
        return "routed local answer"


def test_routed_backend_marks_image_calls_as_secret_and_image_modality():
    router = FakeRouter()
    backend = RoutedBackendAdapter(router, task="visual_qa", available_vram_gb=8.0)
    answer = backend.complete(
        "Describe the chart",
        system="Be accurate",
        image_paths=["/workspace/chart.png"],
        max_new_tokens=100,
        temperature=0.0,
        top_p=0.9,
    )
    assert answer == "routed local answer"
    assert router.kwargs["task"] == "visual_qa"
    assert router.kwargs["modality"] == "image"
    assert router.kwargs["privacy"] == "secret"
    assert router.kwargs["available_vram_gb"] == 8.0
    assert router.kwargs["image_paths"] == ["/workspace/chart.png"]


def test_routed_backend_marks_text_calls_as_secret_and_text_modality():
    router = FakeRouter()
    backend = RoutedBackendAdapter(router, task="tool_planning")
    backend.complete("Inspect files", image_paths=[])
    assert router.kwargs["modality"] == "text"
    assert router.kwargs["privacy"] == "secret"


def test_task_category_inference_is_transparent_and_multimodal_aware():
    assert infer_routing_task("Read the tiny text on this chart.", has_images=True) == "ocr"
    assert infer_routing_task("Describe this picture.", has_images=True) == "visual_qa"
    assert infer_routing_task("Debug this Python stack trace.", has_images=False) == "code"
    assert infer_routing_task("Translate this sentence.", has_images=False) == "general"
    assert infer_routing_task("Inspect files and create a report.", has_images=False) == "tool_planning"
