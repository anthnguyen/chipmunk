from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_gates_training_health_not_two_noisy_minibatches():
    source = (ROOT / "scripts" / "smoke.py").read_text()
    assert 'losses[-1] < losses[0]' not in source
    assert "np.all(np.isfinite(losses))" in source
    assert 'log["final_step"] == cfg.max_steps' in source


def test_pod_python_output_is_unbuffered():
    source = (ROOT / "scripts" / "pod.sh").read_text()
    assert "PY=(.venv/bin/python -u -I)" in source


def test_pod_has_explicit_validated_smoke_resume_path():
    source = (ROOT / "scripts" / "pod.sh").read_text()
    assert 'CHIPMUNK_SMOKE_VALIDATED:-0' in source
    assert 'scripts/run_gate0.py $MODELS' in source
