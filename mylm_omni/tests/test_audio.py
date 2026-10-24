import pytest

from mylm_omni.audio import LocalWhisperTranscriber


def test_whisper_transcriber_requires_existing_local_model_directory(tmp_path):
    with pytest.raises(NotADirectoryError, match="already exist"):
        LocalWhisperTranscriber(tmp_path / "missing-model")


def test_whisper_transcriber_rejects_unsupported_or_oversized_audio(tmp_path):
    model_dir = tmp_path / "whisper-tiny-ct2"
    model_dir.mkdir()
    transcriber = LocalWhisperTranscriber(model_dir, max_audio_bytes=4)
    unsupported = tmp_path / "prompt.txt"
    unsupported.write_text("not audio", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported audio"):
        transcriber.transcribe(unsupported)

    audio = tmp_path / "large.wav"
    audio.write_bytes(b"x" * 5)
    with pytest.raises(ValueError, match="exceeds"):
        transcriber.transcribe(audio)
