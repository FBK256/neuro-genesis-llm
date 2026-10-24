"""Task-aware router for local LLMs using measured benchmark results.

The router does not assume one model is best for every task. Selection requires an explicit
per-task quality score from a frozen evaluation set, filters by modality/privacy/VRAM, and logs
hashes and runtime metadata rather than raw prompts or completions.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PRIVACY_RANK = {"public": 0, "internal": 1, "confidential": 2, "secret": 3}


class _SingleModelBackendPool:
    """Keep at most one heavyweight model backend loaded in the local process."""

    def __init__(self) -> None:
        self.factories: dict[str, Any] = {}
        self.active_model_id: str | None = None
        self.active_backend: Any = None
        # One lock covers model switch and inference; two concurrent requests must not load
        # different large models into the same small GPU at the same time.
        self.lock = threading.RLock()

    def register_factory(self, model_id: str, factory) -> None:
        if model_id in self.factories:
            raise ValueError(f"model factory already registered: {model_id}")
        if not callable(factory):
            raise TypeError("model factory must be callable")
        self.factories[model_id] = factory

    def get(self, model_id: str) -> Any:
        if model_id not in self.factories:
            raise KeyError(model_id)
        if self.active_model_id != model_id or self.active_backend is None:
            self.unload()
            # Instantiate only after the router has selected this model for the task.
            backend = self.factories[model_id]()
            self.active_backend = backend
            self.active_model_id = model_id
        return self.active_backend

    def unload(self) -> None:
        """Release the active backend before loading a different local model.

        Daemon-backed backends may implement unload(). An unload failure is propagated so the
        router does not load a second heavy model while the previous one may still occupy VRAM.
        """
        backend = self.active_backend
        if backend is not None:
            unload = getattr(backend, "unload", None)
            if callable(unload):
                unload()
        self.active_backend = None
        self.active_model_id = None
        # Drop our local strong reference before collecting and clearing the CUDA allocator cache.
        del backend
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class _LazyBackendProxy:
    """Present the backend protocol while loading weights only when complete() is called."""

    def __init__(self, pool: _SingleModelBackendPool, model_id: str) -> None:
        self.pool = pool
        self.model_id = model_id

    def complete(self, prompt: str, **kwargs) -> str:
        with self.pool.lock:
            return self.pool.get(self.model_id).complete(prompt, **kwargs)



def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class LocalModelProfile:
    """Deploy-time properties and measured task scores for one local model."""

    model_id: str
    backend_type: str
    model_ref: str
    modalities: list[str]
    task_quality: dict[str, float]
    task_latency_ms: dict[str, float] = field(default_factory=dict)
    estimated_vram_gb: float | None = None
    max_context_tokens: int = 4096
    local_only: bool = True
    enabled: bool = True
    eval_set_version: str = ""
    model_revision: str = ""
    parameter_count: int | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.model_ref.strip():
            raise ValueError("model_id and model_ref must not be empty")
        if self.backend_type not in {"hf-text", "hf-qwen3-vl", "ollama", "llama-cpp"}:
            raise ValueError("unsupported local backend_type")
        if not self.modalities or any(m not in {"text", "image", "audio", "video"} for m in self.modalities):
            raise ValueError("modalities must include at least one supported modality")
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        if self.estimated_vram_gb is not None and self.estimated_vram_gb < 0:
            raise ValueError("estimated_vram_gb must be non-negative")
        if self.parameter_count is not None and self.parameter_count < 1:
            raise ValueError("parameter_count must be positive")
        if any(not 0.0 <= float(score) <= 1.0 for score in self.task_quality.values()):
            raise ValueError("task quality scores must be in [0, 1]")
        if any(not math.isfinite(float(ms)) or ms < 0 for ms in self.task_latency_ms.values()):
            raise ValueError("task latency values must be finite and non-negative")
        if self.backend_type == "hf-text" and "image" in self.modalities:
            raise ValueError("hf-text cannot claim image support")
        if self.backend_type in {"hf-qwen3-vl", "llama-cpp"} and "image" not in self.modalities:
            raise ValueError(f"{self.backend_type} profile must claim image support")
        if self.backend_type in {"ollama", "llama-cpp"} and not self.local_only:
            raise ValueError("loopback inference backends must be marked local_only")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LocalModelProfile":
        return cls(**raw)


class LocalModelRouter:
    """Select a model by task score with hard privacy, modality, and memory constraints."""

    def __init__(
        self,
        *,
        strict_local: bool = True,
        latency_penalty: float = 0.05,
        audit_path: str | Path | None = None,
    ) -> None:
        if latency_penalty < 0:
            raise ValueError("latency_penalty must be non-negative")
        self.strict_local = strict_local
        self.latency_penalty = latency_penalty
        self.backends: dict[str, Any] = {}
        self.profiles: dict[str, LocalModelProfile] = {}
        self._backend_pool = _SingleModelBackendPool()
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def register(self, profile: LocalModelProfile, backend: Any) -> None:
        if profile.model_id in self.profiles:
            raise ValueError(f"model already registered: {profile.model_id}")
        if not profile.local_only and self.strict_local:
            raise PermissionError("strict-local routing rejects non-local models")
        if not callable(getattr(backend, "complete", None)):
            raise TypeError("backend must provide complete(...)")
        if profile.backend_type in {"ollama", "llama-cpp"}:
            from urllib.parse import urlparse
            base = getattr(backend, "base_url", "")
            host = urlparse(base).hostname
            if host not in {"127.0.0.1", "localhost", "::1"}:
                raise PermissionError(f"{profile.backend_type} endpoint is not loopback")
        self.profiles[profile.model_id] = profile
        self.backends[profile.model_id] = backend

    def register_lazy(self, profile: LocalModelProfile, factory) -> None:
        """Register a factory so a heavyweight model is loaded only after routing chooses it."""
        if profile.model_id in self.profiles:
            raise ValueError(f"model already registered: {profile.model_id}")
        if not profile.local_only and self.strict_local:
            raise PermissionError("strict-local routing rejects non-local models")
        self._backend_pool.register_factory(profile.model_id, factory)
        self.profiles[profile.model_id] = profile
        self.backends[profile.model_id] = _LazyBackendProxy(self._backend_pool, profile.model_id)

    def unload_active_backend(self) -> None:
        """Unload the current backend and release CUDA cache before the next task."""
        with self._backend_pool.lock:
            self._backend_pool.unload()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        audit_path: str | Path | None = None,
        strict_local: bool = True,
        latency_penalty: float = 0.05,
    ) -> "LocalModelRouter":
        """Register local model definitions lazily; only the selected model is loaded."""
        from urllib.parse import urlparse

        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        router = cls(
            strict_local=strict_local,
            latency_penalty=latency_penalty,
            audit_path=audit_path,
        )
        for entry in config.get("models", []):
            profile = LocalModelProfile.from_dict(entry["profile"])
            backend_args = dict(entry.get("backend_args", {}))
            if profile.backend_type in {"ollama", "llama-cpp"}:
                default_url = (
                    "http://127.0.0.1:11434" if profile.backend_type == "ollama"
                    else "http://127.0.0.1:8080/v1"
                )
                parsed = urlparse(backend_args.get("base_url", default_url))
                if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
                    raise PermissionError(f"{profile.backend_type} endpoint must be loopback HTTP")

            def factory(selected_profile=profile, selected_args=backend_args):
                if selected_profile.backend_type in {"hf-text", "hf-qwen3-vl"}:
                    from mylm_omni.backends import LocalHFBackend
                    model_type = (
                        "qwen3-vl" if selected_profile.backend_type == "hf-qwen3-vl" else "text"
                    )
                    return LocalHFBackend(
                        selected_profile.model_ref, model_type=model_type, **selected_args
                    )
                if selected_profile.backend_type == "ollama":
                    from mylm_omni.ollama_backend import OllamaLoopbackBackend
                    return OllamaLoopbackBackend(selected_profile.model_ref, **selected_args)
                from mylm_omni.llama_cpp_backend import LlamaCppLoopbackBackend
                return LlamaCppLoopbackBackend(selected_profile.model_ref, **selected_args)

            router.register_lazy(profile, factory)
        return router

    def _audit(self, record: dict[str, Any]) -> None:
        if self.audit_path is None:
            return
        row = {"at": _now(), **record}
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())

    def select(
        self,
        *,
        task: str,
        modality: str = "text",
        privacy: str = "confidential",
        available_vram_gb: float | None = None,
    ) -> tuple[LocalModelProfile, Any, float]:
        """Select only a calibrated backend that meets all hard constraints."""
        if not task.strip():
            raise ValueError("task must not be empty")
        if modality not in {"text", "image", "audio", "video"}:
            raise ValueError("unsupported modality")
        if privacy not in PRIVACY_RANK:
            raise ValueError("privacy must be public, internal, confidential, or secret")
        candidates: list[tuple[float, LocalModelProfile]] = []
        rejections: dict[str, str] = {}
        for profile in self.profiles.values():
            if not profile.enabled:
                rejections[profile.model_id] = "disabled"
                continue
            if self.strict_local and not profile.local_only:
                rejections[profile.model_id] = "not_local"
                continue
            if PRIVACY_RANK[privacy] >= PRIVACY_RANK["confidential"] and not profile.local_only:
                rejections[profile.model_id] = "privacy_policy"
                continue
            if modality not in profile.modalities:
                rejections[profile.model_id] = "modality_unsupported"
                continue
            # Placeholder numbers must never cause an unbenchmarked model to run by default.
            provenance = (profile.eval_set_version + " " + profile.model_revision).casefold()
            if (
                not profile.eval_set_version.strip()
                or not profile.model_revision.strip()
                or "replace-with" in provenance
                or "placeholder" in provenance
            ):
                rejections[profile.model_id] = "missing_eval_provenance"
                continue
            if task not in profile.task_quality:
                rejections[profile.model_id] = "no_task_calibration"
                continue
            if available_vram_gb is not None and profile.estimated_vram_gb is not None:
                active_id = self._backend_pool.active_model_id
                effective_available_vram = available_vram_gb
                if active_id == profile.model_id:
                    # This model is already resident; do not reject it based on the remaining
                    # free memory it has already consumed.
                    effective_available_vram = float("inf")
                elif active_id is not None:
                    active_profile = self.profiles.get(active_id)
                    active_backend = self._backend_pool.active_backend
                    active_has_unload_hook = callable(getattr(active_backend, "unload", None))
                    active_is_process_managed = active_profile is not None and (
                        active_profile.backend_type in {"hf-text", "hf-qwen3-vl"}
                        or active_has_unload_hook
                    )
                    if (
                        active_is_process_managed
                        and active_profile is not None
                        and active_profile.estimated_vram_gb is not None
                    ):
                        # Estimate the space available *after* the pool releases the active model.
                        effective_available_vram += active_profile.estimated_vram_gb
                if profile.estimated_vram_gb > effective_available_vram:
                    rejections[profile.model_id] = "estimated_vram_exceeded"
                    continue
            quality = float(profile.task_quality[task])
            if not math.isfinite(quality) or not 0 <= quality <= 1:
                rejections[profile.model_id] = "invalid_quality_score"
                continue
            latency = float(profile.task_latency_ms.get(task, 0.0))
            score = quality - self.latency_penalty * math.log1p(latency) / 20.0
            candidates.append((score, profile))
        if not candidates:
            self._audit({
                "event": "model_route_failed",
                "task": task,
                "modality": modality,
                "privacy": privacy,
                "rejections": rejections,
            })
            raise LookupError(
                "no eligible calibrated local model; add task scores from a frozen eval set "
                "or register a model with the required modality"
            )
        score, selected = max(candidates, key=lambda entry: (entry[0], entry[1].model_id))
        self._audit({
            "event": "model_selected",
            "task": task,
            "modality": modality,
            "privacy": privacy,
            "model_id": selected.model_id,
            "routing_score": score,
            "benchmark_quality": selected.task_quality[task],
            "eval_set_version": selected.eval_set_version,
            "rejected_models": rejections,
        })
        return selected, self.backends[selected.model_id], score

    def complete(
        self,
        task: str,
        prompt: str,
        *,
        modality: str = "text",
        privacy: str = "confidential",
        image_paths: list[str | Path] | None = None,
        available_vram_gb: float | None = None,
        system: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> str:
        """Route one request and record hashes/metrics, never the private input/output text."""
        profile, backend, routing_score = self.select(
            task=task,
            modality=modality,
            privacy=privacy,
            available_vram_gb=available_vram_gb,
        )
        request_id = str(uuid.uuid4())
        started = time.monotonic()
        try:
            response = backend.complete(
                prompt,
                system=system,
                image_paths=image_paths,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            elapsed = time.monotonic() - started
            self._audit({
                "event": "model_call",
                "request_id": request_id,
                "model_id": profile.model_id,
                "task": task,
                "privacy": privacy,
                "prompt_sha256": _sha(prompt),
                "response_sha256": _sha(response),
                "response_characters": len(response),
                "elapsed_seconds": elapsed,
                "routing_score": routing_score,
                "success": True,
            })
            return response
        except Exception as exc:
            self._audit({
                "event": "model_call",
                "request_id": request_id,
                "model_id": profile.model_id,
                "task": task,
                "privacy": privacy,
                "prompt_sha256": _sha(prompt),
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(exc).__name__,
                "success": False,
            })
            raise



class RoutedBackendAdapter:
    """Adapt the task-aware model router to the planner's backend.complete(...) protocol.

    Secret-classified tasks are routed only to local backends. The task name must correspond to
    a measured score in a frozen evaluation set; no unbenchmarked fallback is attempted.
    """

    def __init__(
        self,
        router: LocalModelRouter,
        *,
        task: str,
        available_vram_gb: float | None = None,
        safety_fraction: float = 0.80,
    ) -> None:
        if not task.strip():
            raise ValueError("task must not be empty")
        if available_vram_gb is not None and available_vram_gb < 0:
            raise ValueError("available_vram_gb must be non-negative")
        if not 0.1 <= safety_fraction <= 1.0:
            raise ValueError("safety_fraction must be in [0.1, 1.0]")
        self.router = router
        self.task = task
        self.available_vram_gb = available_vram_gb
        self.safety_fraction = safety_fraction

    def _available_vram(self) -> float | None:
        """Use a conservative free-VRAM estimate when the caller did not specify a budget."""
        if self.available_vram_gb is not None:
            return self.available_vram_gb
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            free_bytes, _ = torch.cuda.mem_get_info()
            return (free_bytes / (1024 ** 3)) * self.safety_fraction
        except (ImportError, RuntimeError, AttributeError):
            return None

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
        paths = image_paths or []
        return self.router.complete(
            task=self.task,
            prompt=prompt,
            modality="image" if paths else "text",
            privacy="secret",
            available_vram_gb=self._available_vram(),
            image_paths=paths,
            system=system,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )


def infer_routing_task(goal: str, *, has_images: bool) -> str:
    """Map a user task to a benchmark category; tune categories to the local frozen eval suite."""
    text_value = goal.casefold()
    if has_images:
        ocr_cues = ("ocr", "read text", "read the text", "文字", "読み取", "文字起こし")
        if any(term in text_value for term in ocr_cues) or (
            "read" in text_value and "text" in text_value
        ):
            return "ocr"
        return "visual_qa"
    code_cues = (
        "debug", "code", "python", "program", "stack trace",
        "デバッグ", "コード", "プログラム", "例外", "スタックトレース",
    )
    if any(term in text_value for term in code_cues):
        return "code"
    simple_cues = ("what is", "define", "translate", "summarize", "意味", "翻訳", "要約")
    file_cues = ("file", "folder", "read", "write", "save", "create", "ファイル", "フォルダ", "保存", "作成")
    if any(term in text_value for term in simple_cues) and not any(term in text_value for term in file_cues):
        return "general"
    return "tool_planning"
