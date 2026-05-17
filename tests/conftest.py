import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.backends.mps.is_available():
        return
    skip_mps = pytest.mark.skip(reason="requires Apple Silicon MPS")
    for item in items:
        if "mps_required" in item.keywords:
            item.add_marker(skip_mps)
