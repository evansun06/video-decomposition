from __future__ import annotations

from pathlib import Path


def setup_work_folder(work_dir: str | Path) -> dict[str, Path]:
    """Create the expected per-video work folders."""

    root = Path(work_dir)
    paths = {
        "root": root,
        "audio": root / "audio_temp",
        "image": root / "image_temp",
        "result": root / "result_temp",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths

