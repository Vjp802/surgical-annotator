import pytest
from app.config import Settings

def test_settings_defaults():
    settings = Settings()
    # Check simple defaults
    assert settings.SAM_IMAGE_MODEL_ID == "facebook/sam2.1-hiera-small"
    assert settings.VLM_BACKEND == "ollama"
    assert settings.IS_HF_SPACES is False
    assert settings.DEVICE in ["cuda", "mps", "cpu"]
    assert "liver" in settings.ABDOMINAL_ORGANS

def test_settings_override(monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "robotics_er")
    monkeypatch.setenv("ENABLE_DEPTH_ESTIMATION", "true")
    
    settings = Settings()
    assert settings.VLM_BACKEND == "robotics_er"
    assert settings.ENABLE_DEPTH_ESTIMATION is True
