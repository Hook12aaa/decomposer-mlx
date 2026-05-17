from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Decomposer configuration. Set via environment variables (DECOMPOSER_*) or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DECOMPOSER_",
        case_sensitive=False,
        extra="ignore",
    )

    hf_repo: str = "Qwen/Qwen-Image-Layered"
    text_encoder_repo: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    gguf_repo: str = "unsloth/Qwen-Image-Layered-GGUF"
    gguf_file: str = "qwen-image-layered-Q8_0.gguf"
    gguf_sha256: str = "0d9f3cc357f8ecaf9547908fc24ac9cf05d14f659115ac0aa24c7fb5e86aa92c"

    runs_dir: Path = Path("runs")
    max_zip_bytes: int = 500 * 1024 * 1024
    inference_timeout_seconds: float = 600.0
    job_ttl_seconds: float = 3600.0
    job_store_db_path: Path | None = None

    default_resolution: int = 640
    default_steps: int = 8
    default_layers: int = 6

    lightning_lora_repo: str | None = None
    lightning_lora_filename: str | None = None
    lightning_lora_scale: float = 1.0

    backend: str = "mps"
    mlx_weights_dir: Path = Path("mlx-weights-8bit")

    use_fake_backend: bool = False

    hf_token: SecretStr | None = Field(default=None, repr=False)


def get_settings() -> Settings:
    """Lazy accessor; can be overridden in tests via dependency injection."""
    return Settings()
