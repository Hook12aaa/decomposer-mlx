from pathlib import Path


ANCHOR = "latents = pipe.denoise_only("


def apply(worktree_path: Path) -> None:
    backend_file = worktree_path / "decomposer" / "core" / "mps_backend.py"
    if not backend_file.exists():
        raise RuntimeError(f"mps_backend.py not found at {backend_file}")
    content = backend_file.read_text()
    if ANCHOR not in content:
        raise RuntimeError(f"anchor line not found in {backend_file}: {ANCHOR!r}")

    indent = "            "
    old_block = f"{indent}latents = pipe.denoise_only("
    new_block = (
        f"{indent}with torch.autocast(device_type='mps', dtype=torch.float16):\n"
        f"{indent}    latents = pipe.denoise_only("
    )
    new_content = content.replace(old_block, new_block)
    if new_content == content:
        raise RuntimeError("replacement failed, indent mismatch?")
    backend_file.write_text(new_content)
