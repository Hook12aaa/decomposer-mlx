from decomposer.config import Settings


def test_settings_defaults_are_sensible():
    s = Settings()
    assert s.hf_repo == "Qwen/Qwen-Image-Layered"
    assert s.max_zip_bytes > 0
    assert s.inference_timeout_seconds == 600.0


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("DECOMPOSER_INFERENCE_TIMEOUT_SECONDS", "30.0")
    monkeypatch.setenv("DECOMPOSER_HF_REPO", "test/repo")
    s = Settings()
    assert s.inference_timeout_seconds == 30.0
    assert s.hf_repo == "test/repo"


def test_settings_hf_token_does_not_leak_in_repr(monkeypatch):
    monkeypatch.setenv("DECOMPOSER_HF_TOKEN", "hf_secret_xyz")
    s = Settings()
    assert "hf_secret_xyz" not in repr(s)
    assert s.hf_token is not None
    assert s.hf_token.get_secret_value() == "hf_secret_xyz"
