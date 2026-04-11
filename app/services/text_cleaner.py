import re

def clean_text_for_tts(text: str) -> str:
    text = text.replace("—", ",").replace("–", ",")

    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()
        if len(stripped) <= 1:
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def apply_pronunciation(text: str, name: str, pronunciation: str) -> str:
    if not pronunciation:
        return text

    pattern = rf"\b{name}\b"
    return re.sub(pattern, pronunciation, text, flags=re.IGNORECASE)