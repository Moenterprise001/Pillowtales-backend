import hashlib

def generate_audio_hash(text: str, narrator: str, lang: str, pronunciation: str = "") -> str:
    key = f"{text.strip().lower()}|{narrator}|{lang}|{pronunciation}"
    return hashlib.md5(key.encode()).hexdigest()