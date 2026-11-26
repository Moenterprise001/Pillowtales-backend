"""Shared standard-voice narration for Seasonal stories (26 Nov 2026).

Seasonal stories are pre-written, so every family that plays the same page with
the same narrator sends OpenAI exactly the same text. Instead of paying for that
audio again, the first generation is also saved under a shared path, and later
plays copy it into the family's own folder.

Safety rules:
- Only Seasonal stories (theme "seasonal:<celebration>:<story>"), only standard
  OpenAI narrators, and only when no translation is needed. Never Parent Voice,
  Bedtime or Story Worlds (Story Worlds are personalised per child).
- The shared file is keyed on a fingerprint of the exact text and the exact voice
  directions sent to OpenAI, plus the voice, language and audio cache version.
  Identical input gives identical audio, so a different edition, age band,
  page, wording, pronunciation or voice direction can never reuse the wrong file.
- Every storage step is best-effort: any failure falls back to normal generation.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

SHARED_BUCKET = "story-audio"
SHARED_ROOT = "shared/seasonal"
_SAFE = re.compile(r"[^a-z0-9_-]+")


def _safe_segment(value: str, fallback: str) -> str:
    cleaned = _SAFE.sub("-", str(value or "").strip().lower()).strip("-")
    return cleaned[:80] or fallback


def parse_seasonal_theme(theme: Optional[str]) -> Optional[tuple[str, str]]:
    """Return (celebration, story) for a theme "seasonal:<celebration>:<story>", else None."""
    parts = str(theme or "").strip().split(":")
    if len(parts) != 3 or parts[0].strip().lower() != "seasonal":
        return None
    celebration, story = parts[1].strip(), parts[2].strip()
    if not celebration or not story:
        return None
    return celebration, story


def is_shareable(
    *,
    story_theme: Optional[str],
    voice: str,
    voice_mode: str,
    story_language: str,
    narration_language: str,
) -> bool:
    """True only for Seasonal + standard narrator + same-language narration."""
    if voice_mode != "standard" or voice == "parent_voice":
        return False
    if parse_seasonal_theme(story_theme) is None:
        return False
    return bool(story_language) and story_language == narration_language


def audio_fingerprint(*, tts_text: str, instructions: str) -> str:
    """Fingerprint of exactly what OpenAI receives for one page."""
    payload = f"{tts_text}\n\x1e\n{instructions}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def shared_audio_path(
    *,
    story_theme: str,
    voice: str,
    language_code: str,
    cache_version: str,
    tts_text: str,
    instructions: str,
) -> Optional[str]:
    parsed = parse_seasonal_theme(story_theme)
    if parsed is None or not tts_text:
        return None
    celebration, story = parsed
    folder = "_".join(
        [
            _safe_segment(voice, "voice"),
            _safe_segment(language_code, "lang"),
            _safe_segment(cache_version, "v"),
        ]
    )
    fingerprint = audio_fingerprint(tts_text=tts_text, instructions=instructions)
    return (
        f"{SHARED_ROOT}/{_safe_segment(celebration, 'celebration')}/"
        f"{_safe_segment(story, 'story')}/{folder}/{fingerprint}.mp3"
    )


def download_shared_audio(storage_client, path: str) -> Optional[bytes]:
    """Return the shared MP3, or None if it doesn't exist or storage fails."""
    try:
        data = storage_client.from_(SHARED_BUCKET).download(path)
    except Exception:
        return None
    if isinstance(data, (bytes, bytearray)) and len(data) > 0:
        return bytes(data)
    return None


def upload_shared_audio(storage_client, path: str, audio: bytes) -> bool:
    """Save the MP3 for later families. Never raises."""
    try:
        storage_client.from_(SHARED_BUCKET).upload(
            path,
            audio,
            {"content-type": "audio/mpeg", "upsert": "true"},
        )
        return True
    except Exception as exc:
        print(f"[NARRATION] Shared seasonal audio save skipped path={path}: {exc!r}")
        return False
