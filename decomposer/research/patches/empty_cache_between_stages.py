from pathlib import Path


def apply(worktree_path: Path) -> None:
    backend_file = worktree_path / "decomposer" / "core" / "mps_backend.py"
    if not backend_file.exists():
        raise RuntimeError(f"mps_backend.py not found at {backend_file}")
    content = backend_file.read_text()
    new_content = content.replace(
        "self.residency.free()\n            gc.collect()",
        "self.residency.free()\n            gc.collect()\n            "
        "torch.mps.empty_cache() if torch.backends.mps.is_available() else None",
    )
    if new_content == content:
        raise RuntimeError(f"no residency.free()+gc.collect() pattern found in {backend_file}")
    backend_file.write_text(new_content)
