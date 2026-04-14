from __future__ import annotations

import re
import unicodedata


def _is_punctuation_or_symbol_only(value: str) -> bool:
    if not value:
        return False
    return all(unicodedata.category(ch).startswith(("P", "S")) for ch in value)


def clean_text_for_tts(text: str) -> str:
    if not text:
        return ""

    cleaned = text

    # Normalize long dashes to commas for smoother TTS pauses
    cleaned = cleaned.replace("—", ",").replace("–", ",")

    # Remove common narration markers if present
    cleaned = re.sub(
        r"\[(whisper|softly|chuckle|pause|gently)\]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()

        if not stripped:
            lines.append("")
            continue

        # Drop isolated punctuation/symbol lines, including orphan apostrophes/quotes.
        # Python's stdlib re does not support \p{...}, so use Unicode categories.
        if _is_punctuation_or_symbol_only(stripped):
            continue

        # Drop isolated single letters
        if len(stripped) == 1 and stripped.isalpha():
            continue

        lines.append(line.rstrip())

    # Collapse excessive blank lines
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)

    return cleaned.strip()


def apply_pronunciation(text: str, child_name: str | None, pronunciation: str | None) -> str:
    if not text or not child_name or not pronunciation:
        return text

    escaped_name = re.escape(child_name.strip())
    if not escaped_name:
        return text

    # Replace the child's visible name with a phonetic pronunciation in the TTS copy only
    pattern = rf"\b{escaped_name}\b"
    return re.sub(pattern, pronunciation.strip(), text, flags=re.IGNORECASE)