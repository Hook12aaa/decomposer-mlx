from pathlib import Path


GGUF_FILE = "qwen-image-layered-Q5_K_M.gguf"

CONFIG_FILE = "decomposer/config.py"
OLD_GGUF_LINE = 'gguf_file: str = "qwen-image-layered-Q8_0.gguf"'
NEW_GGUF_LINE = f'gguf_file: str = "{GGUF_FILE}"'

OLD_SHA_PREFIX = '    gguf_sha256: str = "'
NEW_SHA_LINE = '    gguf_sha256: str = ""'


def apply(worktree_path: Path) -> None:
    config = worktree_path / CONFIG_FILE
    if not config.exists():
        raise RuntimeError(f"config.py not found at {config}")
    content = config.read_text()

    if OLD_GGUF_LINE not in content:
        raise RuntimeError(
            f"Expected gguf_file line not found in {config}; "
            f"was the default already changed?"
        )
    content = content.replace(OLD_GGUF_LINE, NEW_GGUF_LINE)

    lines = content.splitlines(keepends=True)
    new_lines = []
    for line in lines:
        if line.strip().startswith("gguf_sha256"):
            new_lines.append(NEW_SHA_LINE + "\n")
        else:
            new_lines.append(line)
    config.write_text("".join(new_lines))
