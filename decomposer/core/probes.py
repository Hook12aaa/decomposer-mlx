import os

import psutil
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def mps_alloc_mb() -> float:
    if not torch.backends.mps.is_available():
        return 0.0
    return torch.mps.driver_allocated_memory() / (1024 * 1024)


class FallbackCounter:
    def __init__(self) -> None:
        self.count = 0

    def note(self, message: str) -> None:
        if "fell back to CPU" in message or ("MPS: " in message and "fallback" in message):
            self.count += 1
