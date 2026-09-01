from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from app.repositories.story_world_repository import StoryWorldRepository


VERIFIED_STATUSES = {"native_speaker_verified", "provider_verified", "reference_verified"}


@dataclass(frozen=True)
class PronunciationRule:
    display_text: str
    replacement_text: str
    normalized_key: str
    version: int


class StoryWorldPronunciationService:
    """Prepare narration-only text for Story World names and cultural terms.

    This service never changes stored or displayed story text. It returns a
    narration copy that can later be passed to a TTS provider.

    This service is called by NarrationService during page-audio generation.
    It prepares narration-only text and never changes stored/displayed story text.
    """

    def __init__(
        self,
        repository: StoryWorldRepository,
        *,
        cache_ttl_seconds: int = 300,
        allow_unverified: bool = False,
    ) -> None:
        self.repository = repository
        self.cache_ttl_seconds = max(0, int(cache_ttl_seconds))
        self.allow_unverified = bool(allow_unverified)
        self._cache: dict[tuple[str, str], tuple[float, list[dict[str, Any]]]] = {}

    @staticmethod
    def _base_language(language_code: Optional[str]) -> str:
        raw = str(language_code or "en").strip().lower().replace("_", "-")
        return raw.split("-", 1)[0] or "en"

    @staticmethod
    def _normalise_provider(provider: Optional[str]) -> str:
        raw = str(provider or "").strip().lower()
        aliases = {
            "openai_tts": "openai",
            "gpt-4o-mini-tts": "openai",
            "eleven_labs": "elevenlabs",
            "parent": "elevenlabs",
            "parent_voice": "elevenlabs",
        }
        return aliases.get(raw, raw)

    def resolve_world_id(self, slug: Optional[str]) -> Optional[str]:
        """Resolve a published Story World slug to its database id."""
        world_slug = str(slug or "").strip().lower()
        if not world_slug:
            return None

        world = self.repository.get_published_world(world_slug)
        if not world:
            return None

        world_id = str(world.get("id") or "").strip()
        return world_id or None

    def clear_cache(self, *, world_id: Optional[str] = None) -> None:
        if world_id is None:
            self._cache.clear()
            return

        world_key = str(world_id)
        for key in list(self._cache):
            if key[0] == world_key:
                self._cache.pop(key, None)

    def _load_global_rows(self, language_code: str) -> list[dict[str, Any]]:
        """Load global pronunciation rows with canonical-English fallback.

        Global rows are available to ordinary personalised bedtime stories and
        Story Worlds. A language-specific global set wins when present; otherwise
        the canonical English set can still protect proper-name pronunciation.
        """
        base_language = self._base_language(language_code)
        cache_key = ("__global__", base_language)
        now = time.monotonic()

        cached = self._cache.get(cache_key)
        if cached is not None:
            cached_at, rows = cached
            if self.cache_ttl_seconds == 0 or now - cached_at <= self.cache_ttl_seconds:
                return rows

        rows = self.repository.get_global_pronunciations(
            language_code=base_language,
        )

        if not rows and base_language != "en":
            rows = self.repository.get_global_pronunciations(
                language_code="en",
            )

        self._cache[cache_key] = (now, rows)
        return rows

    def _load_rows(self, world_id: str, language_code: str) -> list[dict[str, Any]]:
        """Load language-specific pronunciation rows with canonical fallback.

        Story World pronunciation is cultural metadata, not narration-language
        metadata. If a language-specific override exists, use it. Otherwise fall
        back to the canonical English pronunciation set currently used by the
        existing Ireland/Japan data. This keeps names protected when the same
        story is narrated in Spanish, French, German, Italian, or future
        supported languages without duplicating every pronunciation row.
        """
        base_language = self._base_language(language_code)
        cache_key = (str(world_id), base_language)
        now = time.monotonic()

        cached = self._cache.get(cache_key)
        if cached is not None:
            cached_at, rows = cached
            if self.cache_ttl_seconds == 0 or now - cached_at <= self.cache_ttl_seconds:
                return rows

        rows = self.repository.get_pronunciations(
            world_id=str(world_id),
            language_code=base_language,
        )

        # Global Story Worlds fallback: existing canonical pronunciation sets are
        # stored under English. A narration-language-specific set, when present,
        # takes precedence; otherwise reuse the canonical set. English itself
        # naturally stops here and is never queried twice.
        if not rows and base_language != "en":
            rows = self.repository.get_pronunciations(
                world_id=str(world_id),
                language_code="en",
            )

        self._cache[cache_key] = (now, rows)
        return rows

    @staticmethod
    def _provider_replacement(
        row: dict[str, Any],
        provider: str,
        voice: Optional[str],
    ) -> Optional[str]:
        overrides = row.get("provider_overrides") or {}
        if not isinstance(overrides, dict):
            overrides = {}

        provider_config = overrides.get(provider)
        replacement: Optional[str] = None

        if isinstance(provider_config, str):
            replacement = provider_config
        elif isinstance(provider_config, dict):
            voice_key = str(voice or "").strip()
            voice_overrides = provider_config.get("voices")
            if voice_key and isinstance(voice_overrides, dict):
                voice_value = voice_overrides.get(voice_key)
                if isinstance(voice_value, str):
                    replacement = voice_value
                elif isinstance(voice_value, dict):
                    replacement = (
                        voice_value.get("audio_text")
                        or voice_value.get("pronunciation_text")
                        or voice_value.get("text")
                    )

            if not replacement:
                replacement = (
                    provider_config.get("audio_text")
                    or provider_config.get("pronunciation_text")
                    or provider_config.get("text")
                )

        # A phonetic hint is a safe provider-independent fallback only when the
        # record has been explicitly populated and approved.
        if not replacement:
            replacement = row.get("phonetic_hint")

        replacement = str(replacement or "").strip()
        return replacement or None

    def _build_rules(
        self,
        rows: list[dict[str, Any]],
        *,
        provider: str,
        voice: Optional[str],
    ) -> list[PronunciationRule]:
        rules: list[PronunciationRule] = []

        for row in rows:
            if not row.get("active", False):
                continue

            verification_status = str(row.get("verification_status") or "unverified").strip().lower()
            if not self.allow_unverified and verification_status not in VERIFIED_STATUSES:
                continue

            display_text = str(row.get("display_text") or "").strip()
            if not display_text:
                continue

            replacement = self._provider_replacement(row, provider, voice)
            if not replacement or replacement.casefold() == display_text.casefold():
                continue

            rules.append(
                PronunciationRule(
                    display_text=display_text,
                    replacement_text=replacement,
                    normalized_key=str(row.get("normalized_key") or display_text.casefold()).strip(),
                    version=int(row.get("version") or 1),
                )
            )

        # Longest phrase first prevents "Fionn" from partially replacing
        # "Fionn mac Cumhaill" before the full name can be matched.
        rules.sort(key=lambda rule: (-len(rule.display_text), rule.normalized_key, -rule.version))
        return rules

    @staticmethod
    def _replace_phrase(text: str, source: str, target: str) -> str:
        escaped = re.escape(source)
        pattern = rf"(?i)(?<!\w){escaped}(?!\w)"
        return re.sub(pattern, lambda _match: target, text)

    def apply(
        self,
        *,
        text: str,
        world_id: Optional[str],
        language_code: str,
        provider: str,
        voice: Optional[str] = None,
    ) -> str:
        """Return narration-only text with shared verified pronunciation rules.

        Resolution order:
        1. Load global PillowTales pronunciation rows for every narration.
        2. If a Story World is present, layer its rows over the global rows.
        3. Story World rows win for the same normalized_key.
        4. Provider/voice-specific overrides and phonetic hints continue to be
           resolved by the existing rule builder.

        The caller applies the parent's explicit child-name pronunciation before
        this service, so an explicit parent override remains highest priority.
        Stored/displayed story text is never changed.
        """
        if not text:
            return text

        normalised_provider = self._normalise_provider(provider)

        global_rows = self._load_global_rows(language_code)
        merged_rows: dict[str, dict[str, Any]] = {}

        for row in global_rows:
            key = str(
                row.get("normalized_key")
                or row.get("display_text")
                or ""
            ).strip().casefold()
            if key:
                merged_rows[key] = row

        if world_id:
            world_rows = self._load_rows(str(world_id), language_code)
            for row in world_rows:
                key = str(
                    row.get("normalized_key")
                    or row.get("display_text")
                    or ""
                ).strip().casefold()
                if key:
                    # World-specific metadata intentionally overrides the global
                    # row, including the decision to leave a pronunciation alone.
                    merged_rows[key] = row

        rules = self._build_rules(
            list(merged_rows.values()),
            provider=normalised_provider,
            voice=voice,
        )

        narration_text = text
        for rule in rules:
            narration_text = self._replace_phrase(
                narration_text,
                rule.display_text,
                rule.replacement_text,
            )

        return narration_text
