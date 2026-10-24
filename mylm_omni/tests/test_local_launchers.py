import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_posix_llama_server_launcher_parses_and_binds_loopback():
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is unavailable on this platform")
    script = SCRIPTS / "start_llama_server.sh"
    result = subprocess.run(
        [bash, "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    content = script.read_text(encoding="utf-8")
    assert "--host 127.0.0.1" in content
    assert not re.search(r"(?m)^\s*--tools(?:\s|$)", content)


def test_windows_llama_server_launcher_binds_loopback_and_does_not_enable_tools():
    script = SCRIPTS / "start_llama_server.ps1"
    content = script.read_text(encoding="utf-8")
    assert '"--host", "127.0.0.1"' in content
    assert not re.search(r'(?m)^\s*"--tools"\s*,?', content)


def test_smoke_test_uses_loopback_backend_and_has_no_cloud_fallback():
    script = (SCRIPTS / "smoke_test_local_vlm.py").read_text(encoding="utf-8")
    assert "LlamaCppLoopbackBackend" in script
    assert "api.openai.com" not in script
    assert "huggingface.co" not in script
    assert "LOCAL_VLM_OK" in script
