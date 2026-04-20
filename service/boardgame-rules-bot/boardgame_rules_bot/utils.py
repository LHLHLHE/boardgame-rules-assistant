from boardgame_rules_bot.config import settings


def clip_text(value: str, limit: int) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[обрезано]"


def build_history_text(qa_history: list[dict[str, str]]) -> str | None:
    turns = qa_history[-settings.max_history_turns :]
    blocks: list[str] = []
    total_chars = 0
    for idx, item in enumerate(turns, start=1):
        q = clip_text(item.get("q", ""), settings.max_history_chars_per_item)
        a = clip_text(item.get("a", ""), settings.max_history_chars_per_item)
        block = f"QUESTION{idx}: {q}\nANSWER{idx}: {a}"
        if total_chars + len(block) > settings.max_history_chars_total:
            break
        blocks.append(block)
        total_chars += len(block)

    if not blocks:
        return None

    history = "\n\n".join(blocks)
    return "Контекст предыдущих вопросов по этой же игре:\n" + history
