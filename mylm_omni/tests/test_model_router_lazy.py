from mylm_omni.model_router import LocalModelProfile, LocalModelRouter


class FakeBackend:
    def __init__(self, answer, on_unload=None):
        self.answer = answer
        self.on_unload = on_unload

    def complete(self, prompt, **kwargs):
        return self.answer

    def unload(self):
        if self.on_unload is not None:
            self.on_unload()


def profile(name, task_quality):
    return LocalModelProfile(
        model_id=name,
        backend_type="hf-text",
        model_ref=f"/local/{name}",
        modalities=["text"],
        task_quality=task_quality,
        task_latency_ms={task: 1 for task in task_quality},
        estimated_vram_gb=4,
        max_context_tokens=2048,
        local_only=True,
        eval_set_version="frozen-v1",
        model_revision="checksum-test",
        parameter_count=1_000_000_000,
    )


def test_lazy_factory_is_not_called_until_model_is_selected_and_used():
    router = LocalModelRouter()
    calls = []
    router.register_lazy(profile("first", {"general": 0.9}), lambda: calls.append("first") or FakeBackend("one"))
    assert calls == []
    _, backend, _ = router.select(task="general", modality="text")
    assert calls == []
    assert backend.complete("hello") == "one"
    assert calls == ["first"]


def test_lazy_pool_unloads_previous_model_before_loading_next():
    router = LocalModelRouter()
    calls = []
    router.register_lazy(profile("first", {"general": 0.9}), lambda: calls.append("first") or FakeBackend("one"))
    router.register_lazy(profile("second", {"code": 0.95}), lambda: calls.append("second") or FakeBackend("two"))

    _, first_proxy, _ = router.select(task="general", modality="text")
    assert first_proxy.complete("one") == "one"
    assert router._backend_pool.active_model_id == "first"

    _, second_proxy, _ = router.select(task="code", modality="text")
    assert second_proxy.complete("two") == "two"
    assert router._backend_pool.active_model_id == "second"
    assert calls == ["first", "second"]



def test_lazy_pool_calls_backend_unload_hook_before_loading_next_model():
    router = LocalModelRouter()
    loaded = []
    unloaded = []
    router.register_lazy(
        profile("first", {"general": 0.9}),
        lambda: loaded.append("first") or FakeBackend("first", on_unload=lambda: unloaded.append("first")),
    )
    router.register_lazy(
        profile("second", {"code": 0.95}),
        lambda: loaded.append("second") or FakeBackend("second", on_unload=lambda: unloaded.append("second")),
    )

    _, first, _ = router.select(task="general", modality="text")
    assert first.complete("prompt") == "first"
    assert unloaded == []
    _, second, _ = router.select(task="code", modality="text")
    assert second.complete("prompt") == "second"
    assert unloaded == ["first"]
    assert loaded == ["first", "second"]



def test_vram_router_accounts_for_memory_released_by_an_in_process_model_switch():
    router = LocalModelRouter()
    router.register_lazy(
        profile("first", {"general": 0.9}),
        lambda: FakeBackend("first"),
    )
    router.register_lazy(
        profile("second", {"code": 0.95}),
        lambda: FakeBackend("second"),
    )
    _, first, _ = router.select(task="general", modality="text", available_vram_gb=8.0)
    assert first.complete("prompt") == "first"

    # Current free VRAM is only 1 GB, but the active in-process model is estimated at 4 GB.
    # The new 4 GB model can fit after the old one has been released, so it should be eligible.
    selected, _, _ = router.select(
        task="code", modality="text", available_vram_gb=1.0
    )
    assert selected.model_id == "second"


def test_active_model_is_not_rejected_for_the_vram_it_already_uses():
    router = LocalModelRouter()
    router.register_lazy(
        profile("first", {"general": 0.9}),
        lambda: FakeBackend("first"),
    )
    _, backend, _ = router.select(task="general", modality="text", available_vram_gb=4.0)
    assert backend.complete("prompt") == "first"
    selected, _, _ = router.select(
        task="general", modality="text", available_vram_gb=0.1
    )
    assert selected.model_id == "first"
