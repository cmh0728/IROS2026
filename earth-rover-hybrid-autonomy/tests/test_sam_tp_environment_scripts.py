from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_setup_falls_back_to_an_ignored_independent_venv() -> None:
    script = (ROOT / "scripts/setup_sam_tp_reproduction.sh").read_text(encoding="utf-8")

    assert 'ENV_BACKEND="${ENV_BACKEND:-auto}"' in script
    assert 'ENV_BACKEND="venv"' in script
    assert 'external/venvs/$ENV_NAME' in script
    assert '"$PYTHON_BOOTSTRAP" -m venv "$VENV_PATH"' in script
    assert "env_python -m pip --version" in script
    assert "env_python -m ensurepip --upgrade" in script
    assert "sudo apt install -y python3.10-venv" in script
    assert 'pip install -r "$UPSTREAM_ROOT/requirements.txt"' in script
    assert "conda is required" not in script


def test_runner_auto_detects_conda_or_venv_without_system_install() -> None:
    script = (ROOT / "scripts/run_sam_tp_reproduction.sh").read_text(encoding="utf-8")

    assert 'ENV_BACKEND="${ENV_BACKEND:-auto}"' in script
    assert 'elif [[ -x "$VENV_PATH/bin/python" ]]' in script
    assert '"$VENV_PATH/bin/python" "$@"' in script
    assert 'export SAM_TP_ENVIRONMENT_BACKEND="$ENV_BACKEND"' in script
