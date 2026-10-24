import pytest

from mylm_omni.model_router import LocalModelProfile, LocalModelRouter


class FakeBackend:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        return self.answer


def profile(model_id, *, score, latency, modalities=None, local=True, task="general"):
    return LocalModelProfile(
        model_id=model_id,
        backend_type="hf-qwen3-vl" if "image" in (modalities or ["text"]) else "hf-text",
        model_ref=f"/local/{model_id}",
        modalities=modalities or ["text"],
        task_quality={task: score},
        task_latency_ms={task: latency},
        estimated_vram_gb=8.0,
        max_context_tokens=1024,
        local_only=local,
        eval_set_version="frozen-v1",
        model_revision="immutable-test-revision",
        parameter_count=1_000_000_000,
    )


def test_select_uses_calibrated_quality_and_latency_not_name(tmp_path):
    router = LocalModelRouter(audit_path=tmp_path / "routing.jsonl", latency_penalty=0.5)
    slow = profile("slow-but-good", score=0.95, latency=2000)
    fast = profile("fast-good", score=0.90, latency=5)
    router.register(slow, FakeBackend("slow"))
    router.register(fast, FakeBackend("fast"))

    selected, backend, score = router.select(
        task="general", modality="text", privacy="confidential", available_vram_gb=12,
    )
    assert selected.model_id == "fast-good"
    assert backend.complete("x") == "fast"
    assert score > 0
    assert '"prompt"' not in (tmp_path / "routing.jsonl").read_text(encoding="utf-8")


def test_route_rejects_unmeasured_task_and_insufficient_vram(tmp_path):
    router = LocalModelRouter(audit_path=tmp_path / "routing.jsonl")
    router.register(profile("local", score=0.8, latency=10), FakeBackend("answer"))
    with pytest.raises(LookupError, match="no eligible calibrated"):
        router.select(task="code", modality="text")
    with pytest.raises(LookupError, match="no eligible calibrated"):
        router.select(task="general", modality="text", available_vram_gb=1.0)


def test_strict_local_rejects_nonlocal_backend_profile():
    router = LocalModelRouter(strict_local=True)
    with pytest.raises(PermissionError, match="non-local"):
        router.register(profile("remote", score=0.9, latency=50, local=False), FakeBackend("x"))


def test_image_request_requires_vision_capability():
    router = LocalModelRouter()
    router.register(profile("text-only", score=0.9, latency=5), FakeBackend("text"))
    with pytest.raises(LookupError, match="no eligible calibrated"):
        router.select(task="general", modality="image", privacy="secret")



def test_router_rejects_example_scores_without_real_eval_provenance():
    router = LocalModelRouter()
    uncalibrated = profile("example", score=0.99, latency=1)
    uncalibrated.eval_set_version = "replace-with-frozen-eval-version"
    router.register(uncalibrated, FakeBackend("unvalidated answer"))
    with pytest.raises(LookupError, match="no eligible calibrated"):
        router.select(task="general", modality="text", privacy="confidential")
