from pathlib import Path


def apply(worktree_path: Path) -> None:
    backend_file = worktree_path / "decomposer" / "core" / "mps_backend.py"
    if not backend_file.exists():
        raise RuntimeError(f"mps_backend.py not found at {backend_file}")
    content = backend_file.read_text()
    out_lines: list[str] = []
    replaced = 0
    for line in content.splitlines():
        if line.lstrip().startswith("self.residency.free()"):
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}# keep-warm: self.residency.free()")
            replaced += 1
        else:
            out_lines.append(line)
    if replaced == 0:
        raise RuntimeError(
            f"no self.residency.free() lines found in {backend_file}, nothing to disable"
        )
    backend_file.write_text("\n".join(out_lines) + ("\n" if content.endswith("\n") else ""))
