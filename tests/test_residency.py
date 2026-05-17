import pytest
from decomposer.core.residency import ResidencyManager


class FakeModule:
    def __init__(self, name: str) -> None:
        self.name = name
        self.on_device = False

    def to(self, device: str) -> "FakeModule":
        self.on_device = (device != "cpu")
        return self


def test_residency_loads_module_to_mps():
    rm = ResidencyManager(device="cpu")
    loaded = rm.load("text", lambda: FakeModule("text"))
    assert rm.current_name == "text"
    assert loaded.name == "text"
    assert loaded.on_device is False


def test_residency_moves_module_to_configured_device():
    rm = ResidencyManager(device="mps")
    loaded = rm.load("dit", lambda: FakeModule("dit"))
    assert loaded.on_device is True


def test_residency_rejects_factory_returning_non_module():
    rm = ResidencyManager(device="cpu")
    with pytest.raises(TypeError, match="without .to"):
        rm.load("text", lambda: object())


def test_residency_frees_previous_before_loading_next():
    rm = ResidencyManager(device="cpu")
    loaded_order: list[str] = []
    rm.load("text", lambda: (loaded_order.append("load-text"), FakeModule("text"))[1])
    rm.load("dit", lambda: (loaded_order.append("load-dit"), FakeModule("dit"))[1])
    assert loaded_order == ["load-text", "load-dit"]
    assert rm.current_name == "dit"


def test_residency_free_is_idempotent():
    rm = ResidencyManager(device="cpu")
    rm.load("text", lambda: FakeModule("text"))
    rm.free()
    rm.free()
    assert rm.current_name is None


def test_residency_rejects_unknown_module_name():
    rm = ResidencyManager(device="cpu")
    with pytest.raises(ValueError, match="unknown module"):
        rm.load("not_a_module", lambda: FakeModule("x"))
