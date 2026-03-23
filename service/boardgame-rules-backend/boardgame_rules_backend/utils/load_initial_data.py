import tempfile
import zipfile
from pathlib import Path


def extract_or_use_path(path: Path) -> tuple[Path, bool]:
    """Extract zip to temp dir or return path as-is. Returns (base_path, was_extracted)."""
    if path.suffix.lower() == ".zip":
        tmpdir = tempfile.mkdtemp(prefix="load_initial_")
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmpdir)
        return Path(tmpdir), True
    return path, False
