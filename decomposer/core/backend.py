import time
from typing import Protocol

from PIL import Image

from decomposer.core.xray import Tracer


class InferenceBackend(Protocol):
    def decompose(
        self,
        image: Image.Image,
        layers: int,
        resolution: int = 640,
        steps: int = 8,
        seed: int | None = None,
        tracer: Tracer | None = None,
    ) -> list[Image.Image]: ...


class FakeBackend:
    def __init__(self, latency_ms: int = 100) -> None:
        self._latency_s = latency_ms / 1000.0

    def decompose(
        self,
        image: Image.Image,
        layers: int,
        resolution: int = 640,
        steps: int = 8,
        seed: int | None = None,
        tracer: Tracer | None = None,
    ) -> list[Image.Image]:
        t = tracer or Tracer(run_id="fake")
        with t.stage("load_text_encoder"):
            time.sleep(self._latency_s)
        with t.stage("encode_prompt"):
            time.sleep(self._latency_s)
        with t.stage("free_text_encoder"):
            pass
        with t.stage("load_dit"):
            time.sleep(self._latency_s)
        with t.stage("denoise_loop", steps=steps):
            for i in range(steps):
                with t.step("denoise_step", i=i):
                    time.sleep(self._latency_s / steps)
        with t.stage("free_dit"):
            pass
        with t.stage("load_vae"):
            time.sleep(self._latency_s)
        with t.stage("decode_layers", n=layers):
            time.sleep(self._latency_s)
        with t.stage("free_vae"):
            pass
        return [Image.new("RGBA", (resolution, resolution), (i * 30, 0, 0, 255))
                for i in range(layers)]


def get_backend(name: str) -> InferenceBackend:
    if name == "fake":
        return FakeBackend(latency_ms=50)
    if name == "mlx":
        from decomposer.config import get_settings
        from decomposer.core.mlx_backend import MlxBackend
        return MlxBackend(settings=get_settings())
    from decomposer.core.mps_backend import MpsBackend
    return MpsBackend()
