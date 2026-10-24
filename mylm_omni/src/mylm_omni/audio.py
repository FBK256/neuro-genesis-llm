"""Local-only speech-to-text adapter using a pre-downloaded faster-whisper model."""
from __future__ import annotations

from pathlib import Path
from typing import Any


_ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}


class LocalWhisperTranscriber:
    """Transcribe a local audio file using a model directory already present on disk.

    Passing a local directory instead of a model-size string prevents an implicit model download.
    For full offline operation, download the CTranslate2 model beforehand, verify its provenance,
    and block network egress at the OS/container level.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        max_audio_bytes: int = 20_000_000,
        cpu_threads: int = 4,
    ) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(
                f"Whisper model directory must already exist locally: {path}"
            )
        if max_audio_bytes < 1 or cpu_threads < 1:
            raise ValueError("max_audio_bytes and cpu_threads must be positive")
        self.model_path = path
        self.device = device
        self.compute_type = compute_type
        self.max_audio_bytes = max_audio_bytes
        self.cpu_threads = cpu_threads
        self._model: Any = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "Install optional audio dependencies with pip install -e '.[audio]'"
                ) from exc
            self._model = WhisperModel(
                str(self.model_path),
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        beam_size: int = 3,
        vad_filter: bool = True,
    ) -> dict[str, Any]:
        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in _ALLOWED_AUDIO_SUFFIXES:
            raise ValueError(f"unsupported audio file extension: {path.suffix}")
        size = path.stat().st_size
        if size > self.max_audio_bytes:
            raise ValueError(f"audio exceeds the {self.max_audio_bytes}-byte limit")
        if beam_size < 1 or beam_size > 10:
            raise ValueError("beam_size must be in [1, 10]")

        segments, info = self._load().transcribe(
            str(path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        # faster-whisper returns a lazy generator; consume it before returning.
        segment_rows = [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "avg_logprob": float(segment.avg_logprob),
                "no_speech_prob": float(segment.no_speech_prob),
            }
            for segment in segments
        ]
        return {
            "path": str(path),
            "language": getattr(info, "language", language),
            "language_probability": float(getattr(info, "language_probability", 0.0)),
            "text": " ".join(row["text"] for row in segment_rows).strip(),
            "segments": segment_rows,
            "model_path": str(self.model_path),
        }
