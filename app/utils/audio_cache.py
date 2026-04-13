import hashlib


def generate_audio_hash(text: str, narrator: str, lang: str, pronunciation: str | None = None) -> str:
    safe_text = (text or "").strip().lower()
    safe_narrator = (narrator or "").strip().lower()
    safe_lang = (lang or "").strip().lower()
    safe_pronunciation = (pronunciation or "").strip().lower()

    key = f"{safe_text}|{safe_narrator}|{safe_lang}|{safe_pronunciation}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()
