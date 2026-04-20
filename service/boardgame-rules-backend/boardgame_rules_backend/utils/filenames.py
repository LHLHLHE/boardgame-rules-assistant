from slugify import slugify


def build_rules_source_filename(
    game_title: str,
    extension: str,
    game_id: int | None = None,
) -> str:
    """Build normalized rules source filename: <slugified_game_title>_rules.<ext>."""
    ext = extension.lower().lstrip(".") or "bin"
    slug = slugify(game_title, separator="_", lowercase=True)
    base = slug or (f"game_{game_id}" if game_id is not None else "game")
    suffix = f"_rules.{ext}"
    max_base_len = max(1, 255 - len(suffix))
    base = base[:max_base_len]
    return f"{base}{suffix}"
