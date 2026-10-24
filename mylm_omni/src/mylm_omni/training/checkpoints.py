"""Resumable checkpoints using safetensors for weights and an auditable manifest."""
from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_model, save_model


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Prefer same-volume hard links to avoid duplicate multi-gigabyte checkpoint files."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class CheckpointManager:
    """Save every N steps, keep a latest mirror, and retain top-N evaluation snapshots.

    Optimizer/scheduler/scaler states are in trainer_state.pt (a local trusted pickle).
    Model weights are stored separately in model.safetensors. Never load trainer_state.pt
    from untrusted sources.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        checkpoint_steps: int = 100,
        keep_steps: int = 3,
        keep_best: int = 3,
        best_metric: str = "eval_loss",
        lower_is_better: bool = True,
    ) -> None:
        if checkpoint_steps < 1 or keep_steps < 1 or keep_best < 1:
            raise ValueError("checkpoint_steps, keep_steps, and keep_best must be positive")
        self.root = Path(root)
        self.steps_dir = self.root / "steps"
        self.latest_dir = self.root / "latest"
        self.best_dir = self.root / "best"
        self.root.mkdir(parents=True, exist_ok=True)
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_steps = checkpoint_steps
        self.keep_steps = keep_steps
        self.keep_best = keep_best
        self.best_metric = best_metric
        self.lower_is_better = lower_is_better
        self._recover_latest()

    def _recover_latest(self) -> None:
        previous = self.root / "latest.prev"
        if not self.latest_dir.exists() and previous.exists():
            os.replace(previous, self.latest_dir)
        elif previous.exists():
            shutil.rmtree(previous, ignore_errors=True)

    @staticmethod
    def capture_rng_state() -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
        }
        try:
            import numpy as np
            state["numpy"] = np.random.get_state()
        except ImportError:
            pass
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def restore_rng_state(state: dict[str, Any]) -> None:
        if "python" in state:
            random.setstate(state["python"])
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if "numpy" in state:
            try:
                import numpy as np
                np.random.set_state(state["numpy"])
            except ImportError:
                pass
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def save(
        self,
        *,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        training_state: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Write a complete step checkpoint first, then update the latest pointer/mirror."""
        if step < 0:
            raise ValueError("step must be non-negative")
        final_dir = self.steps_dir / f"step-{step:06d}"
        temp_dir = Path(tempfile.mkdtemp(prefix=f".step-{step:06d}-", dir=self.steps_dir))
        try:
            save_model(model, str(temp_dir / "model.safetensors"))
            state = {
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "training_state": training_state or {},
                "rng_state": self.capture_rng_state(),
            }
            torch.save(state, temp_dir / "trainer_state.pt")
            config = getattr(model, "config", None)
            config_dict = config.to_dict() if hasattr(config, "to_dict") else None
            manifest = {
                "step": step,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics or {},
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "config": config_dict,
                "weights_file": "model.safetensors",
                "trainer_state_file": "trainer_state.pt",
                "format": "mylm-omni-checkpoint-v1",
            }
            _atomic_json(temp_dir / "manifest.json", manifest)

            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(temp_dir, final_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        self._refresh_latest(final_dir)
        self._update_best(final_dir, manifest)
        self._prune_steps()
        _atomic_json(self.root / "manifest.json", {
            "latest_step": step,
            "latest_path": str(Path("steps") / final_dir.name),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        return final_dir

    def _refresh_latest(self, source_dir: Path) -> None:
        temporary = self.root / "latest.tmp"
        previous = self.root / "latest.prev"
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        try:
            for src in source_dir.iterdir():
                _link_or_copy(src, temporary / src.name)
            if previous.exists():
                shutil.rmtree(previous, ignore_errors=True)
            if self.latest_dir.exists():
                os.replace(self.latest_dir, previous)
            try:
                os.replace(temporary, self.latest_dir)
            except Exception:
                if previous.exists() and not self.latest_dir.exists():
                    os.replace(previous, self.latest_dir)
                raise
            shutil.rmtree(previous, ignore_errors=True)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def _update_best(self, source_dir: Path, manifest: dict[str, Any]) -> None:
        metrics = manifest.get("metrics", {})
        if self.best_metric not in metrics:
            return
        try:
            score = float(metrics[self.best_metric])
        except (TypeError, ValueError):
            return
        index_path = self.best_dir / "index.json"
        old_entries = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
        entries = [entry for entry in old_entries if entry["step"] != manifest["step"]]
        entries.append({"step": manifest["step"], "score": score})
        entries.sort(key=lambda item: item["score"], reverse=not self.lower_is_better)
        entries = entries[: self.keep_best]

        for rank in range(1, self.keep_best + 1):
            shutil.rmtree(self.best_dir / f"rank-{rank}", ignore_errors=True)
        for rank, entry in enumerate(entries, start=1):
            src_dir = self.steps_dir / f"step-{int(entry['step']):06d}"
            if not src_dir.exists():
                continue
            dst_dir = self.best_dir / f"rank-{rank}"
            dst_dir.mkdir(parents=True)
            # Best inference snapshots need weights and the config/metrics manifest, not optimizer state.
            _link_or_copy(src_dir / "model.safetensors", dst_dir / "model.safetensors")
            _link_or_copy(src_dir / "manifest.json", dst_dir / "manifest.json")
        _atomic_json(index_path, entries)

    def _prune_steps(self) -> None:
        step_dirs = sorted(
            (p for p in self.steps_dir.glob("step-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[-1]),
            reverse=True,
        )
        protected = set()
        for path in step_dirs[: self.keep_steps]:
            protected.add(path)
        # A top-N best snapshot can reference a step older than the rolling window.
        index_path = self.best_dir / "index.json"
        if index_path.exists():
            for item in json.loads(index_path.read_text(encoding="utf-8")):
                protected.add(self.steps_dir / f"step-{int(item['step']):06d}")
        for path in step_dirs:
            if path not in protected:
                shutil.rmtree(path, ignore_errors=True)

    def load_latest(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any] | None:
        """Restore latest weights and training state; return metadata or None if absent."""
        if not self.latest_dir.is_dir():
            return None
        manifest_path = self.latest_dir / "manifest.json"
        weights_path = self.latest_dir / "model.safetensors"
        state_path = self.latest_dir / "trainer_state.pt"
        if not (manifest_path.is_file() and weights_path.is_file() and state_path.is_file()):
            return None

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        load_model(model, str(weights_path), strict=True)
        # Checkpoints are local trusted files. Explicit weights_only=False is needed for RNG state.
        state = torch.load(state_path, map_location=map_location, weights_only=False)
        if optimizer is not None and state["optimizer"] is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state["scaler"] is not None:
            scaler.load_state_dict(state["scaler"])
        self.restore_rng_state(state.get("rng_state", {}))
        manifest["training_state"] = state.get("training_state", {})
        return manifest
