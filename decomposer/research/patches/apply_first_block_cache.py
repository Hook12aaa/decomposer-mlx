from pathlib import Path


THRESHOLD = 0.08
ANCHOR = "pipe = self._build_pipeline(dit, vae=None)"


def apply(worktree_path: Path) -> None:
    backend_file = worktree_path / "decomposer" / "core" / "mps_backend.py"
    if not backend_file.exists():
        raise RuntimeError(f"mps_backend.py not found at {backend_file}")
    content = backend_file.read_text()
    if ANCHOR not in content:
        raise RuntimeError(f"anchor line not found in {backend_file}: {ANCHOR!r}")
    injection = (
        f"        from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig\n"
        f"        apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold={THRESHOLD}))\n"
    )
    new_content = content.replace(ANCHOR + "\n", ANCHOR + "\n" + injection)
    backend_file.write_text(new_content)
