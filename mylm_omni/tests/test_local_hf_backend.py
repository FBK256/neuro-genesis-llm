import pytest

from mylm_omni.backends import LocalHFBackend


def test_local_hf_backend_requires_a_real_local_directory(tmp_path):
    with pytest.raises(NotADirectoryError, match="pre-existing local model directory"):
        LocalHFBackend(tmp_path / "model-not-downloaded")
