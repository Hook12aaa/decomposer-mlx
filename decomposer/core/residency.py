import gc
from typing import Callable, Literal

import torch


ALLOWED = {"text", "dit", "vae"}


class ResidencyManager:
    def __init__(self, device: str = "mps") -> None:
        self.device = device
        self._current: object | None = None
        self.current_name: str | None = None

    def load(self, name: Literal["text", "dit", "vae"], factory: Callable[[], object]) -> object:
        if name not in ALLOWED:
            raise ValueError(f"unknown module {name!r}; allowed={sorted(ALLOWED)}")
        if self._current is not None:
            self.free()
        module = factory()
        if not hasattr(module, "to"):
            raise TypeError(
                f"factory for {name!r} returned {type(module).__name__} without .to(); "
                f"expected an nn.Module"
            )
        module = module.to(self.device)
        self._current = module
        self.current_name = name
        return module

    def free(self) -> None:
        if self._current is None:
            return
        self._current = None
        self.current_name = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
