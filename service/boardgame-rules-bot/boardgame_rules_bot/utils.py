from boardgame_rules_bot.constants import (MAX_HISTORY_CHARS_PER_ITEM, MAX_HISTORY_CHARS_TOTAL,
                                           MAX_HISTORY_TURNS)


def clip_text(value: str, limit: int) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[обрезано]"


def build_history_text(qa_history: list[dict[str, str]]) -> str | None:
    turns = qa_history[-MAX_HISTORY_TURNS:]
    blocks: list[str] = []
    total_chars = 0
    for idx, item in enumerate(turns, start=1):
        q = clip_text(item.get("q", ""), MAX_HISTORY_CHARS_PER_ITEM)
        a = clip_text(item.get("a", ""), MAX_HISTORY_CHARS_PER_ITEM)
        block = f"QUESTION{idx}: {q}\nANSWER{idx}: {a}"
        if total_chars + len(block) > MAX_HISTORY_CHARS_TOTAL:
            break
        blocks.append(block)
        total_chars += len(block)

    if not blocks:
        return None

    history = "\n\n".join(blocks)
    return "Контекст предыдущих вопросов по этой же игре:\n" + history
