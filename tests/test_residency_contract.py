import gc
import weakref

import torch.nn as nn

from decomposer.core.residency import ResidencyManager


def _make_tiny_module(name: str) -> nn.Module:
    return nn.Sequential(
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
    )


def test_loading_second_module_releases_the_first():
    rm = ResidencyManager(device="cpu")
    first = rm.load("text", lambda: _make_tiny_module("text"))
    first_ref = weakref.ref(first)

    second = rm.load("dit", lambda: _make_tiny_module("dit"))

    del first
    gc.collect()

    # If ResidencyManager still held `first`, this weakref would resolve, proving it
    # would be retaining two modules and violating the at-most-one invariant.
    assert first_ref() is None, (
        "ResidencyManager.load() did not release the previous module, "
        "at-most-one invariant violated"
    )
    assert rm.current_name == "dit"
    assert second is not None


def test_free_releases_module():
    rm = ResidencyManager(device="cpu")
    module = rm.load("vae", lambda: _make_tiny_module("vae"))
    module_ref = weakref.ref(module)

    del module
    rm.free()
    gc.collect()

    assert module_ref() is None, "ResidencyManager.free() did not release the module"
    assert rm.current_name is None
