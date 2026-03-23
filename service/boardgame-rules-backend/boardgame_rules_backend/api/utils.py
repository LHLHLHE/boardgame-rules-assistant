import io
import zipfile
from pathlib import Path


def extract_zip_safely(archive_content: bytes, destination: Path) -> None:
    """Extract ZIP into destination and block path traversal entries."""
    with zipfile.ZipFile(io.BytesIO(archive_content), "r") as zf:
        base = destination.resolve()
        for member in zf.infolist():
            target = (base / member.filename).resolve()
            try:
                target.relative_to(base)
            except ValueError as exc:
                raise ValueError(
                    f"Archive entry escapes target dir: {member.filename}"
                ) from exc
        zf.extractall(base)
