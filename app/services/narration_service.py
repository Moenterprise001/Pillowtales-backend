from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import BackgroundTasks, HTTPException

from app.domain.constants import SUPPORTED_LANGUAGES, VOICE_PRESETS
from app.models.narration import NarrationRequest, NarrationResponse, PageStatusResponse
from app.models.subscription import SubscriptionResponse
from app.repositories.story_repository import StoryRepository
from app.repositories.user_repository import UserRepository
from app.services.subscription_service import SubscriptionService
from app.services.story_world_pronunciation_service import StoryWorldPronunciationService
from app.services.text_cleaner import clean_text_for_tts, apply_pronunciation

# In-memory job state for active chunked narration generation.
# Good enough for a single Render instance launch setup.
_chunked_jobs: dict[str, dict] = {}

# Cache versioning. Keep non-English/other narrator caches stable for launch,
# but isolate Wise Owl English audio so old mixed chunks cannot be reused.
DEFAULT_AUDIO_CACHE_VERSION = "v5"
WISE_OWL_AUDIO_CACHE_VERSION = "v8"
# Bump standard non-English narration caches so improved bedtime/accent shaping
# is generated fresh. Parent Voice cache paths remain untouched.
STANDARD_LANGUAGE_AUDIO_CACHE_VERSION = {
    # Separate English locale caches so US Night Owl and UK Wise Owl never
    # replay audio generated through the previous single-English narrator path.
    "en-US": "v8",
    "en-GB": "v8",
    "es": "v10",
    "fr": "v10",
    "de": "v10",
    "it": "v10",
    "ja": "v1",
    "ar": "v1",
}

# In-memory abuse/rate limiting state.
# Also single-instance only, but enough for launch while on one Render instance.
_parent_voice_ip_log: dict[str, list[float]] = {}
_parent_voice_user_log: dict[str, list[float]] = {}

ENGLISH_US_CODES = {"en", "en-us", "en_us"}
ENGLISH_UK_CODES = {"en-gb", "en_uk", "en-uk", "en_gb"}

def normalize_language_code(language_code: Optional[str], *, preserve_english_locale: bool = True) -> str:
    raw = str(language_code or "en-US").strip().lower().replace("_", "-")
    if raw in ENGLISH_UK_CODES:
        return "en-GB" if preserve_english_locale else "en"
    if raw in ENGLISH_US_CODES:
        return "en-US" if preserve_english_locale else "en"
    return raw[:2] or "en"

def base_language_code(language_code: Optional[str]) -> str:
    return normalize_language_code(language_code, preserve_english_locale=False)

def prepare_narration_text(text: str) -> str:
    if not text:
        return text

    text = text.strip()

    # Add natural pacing using punctuation spacing.
    # This avoids SSML tags so it stays compatible with OpenAI TTS and ElevenLabs.
    # Keep this conservative: it must never add spoken instructions or change story meaning.
    text = re.sub(r"\s+", " ", text)
    text = text.replace("...", "…")
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = text.replace(" — ", ".  ")
    text = text.replace(" – ", ".  ")
    text = re.sub(r"([.!?…])\s+", r"\1  ", text)
    text = re.sub(r"([,;:])\s+", r"\1 ", text)

    return text.strip()




def prepare_parent_voice_text(text: str, language_code: str) -> str:
    """Prepare Parent Voice text for natural ElevenLabs delivery.

    Parent Voice should rely on the cloned speaker's natural rhythm rather than
    injected SSML-style break tags. Some cloned voices vocalise values from
    tags such as <break time="0.18s" /> (for example, as "eighteens"), which
    adds words that are absent from the story and breaks narration sync.

    This remains a Parent Voice-only text cleanup. It does not change credits,
    cache ownership, chunk generation, polling, playback, sync, or stored text.
    """
    if not text:
        return text

    cleaned = str(text).strip()
    if not cleaned:
        return cleaned

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("...", "…")

    # Defensive cleanup in case break markup arrives from another text layer.
    # ElevenLabs receives plain story text only.
    cleaned = re.sub(
        r'<break\\b[^>]*?/?>',
        ' ',
        cleaned,
        flags=re.IGNORECASE,
    )

    # Preserve genuine paragraph boundaries so the voice can pause naturally.
    cleaned = re.sub(r"\n[ \t]*\n+", "\n\n", cleaned)
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)

    # Clean spacing without inserting anything that could be spoken aloud.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:…])", r"\1", cleaned)
    cleaned = re.sub(r"([,.!?;:…])(?=[^\s\n])", r"\1 ", cleaned)

    # Keep normal sentence flow. Only genuine story paragraph boundaries should
    # become paragraph pauses; turning every sentence into a paragraph makes
    # delivery clipped and is interpreted inconsistently across page requests.
    cleaned = re.sub(r",\s+", ", ", cleaned)
    cleaned = re.sub(r" *\n\n *", "\n\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()

def add_soft_chunk_leadin(text: str) -> str:
    """Normalize later page starts without adding spoken punctuation.

    Older builds prefixed pages 2+ with punctuation to soften hard starts. That
    removed some hard attacks but could create an audible gulp/breath in
    Spanish and French. We now keep the spoken text clean and rely on the TTS
    performance instructions to maintain seamless page-to-page continuity.

    This is TTS-input polish only: it must not affect narration ownership,
    chunking, page-status polling, playback, sync, Parent Voice replay, or
    story text stored in the database.
    """
    if not text:
        return text

    cleaned = text.strip()
    if not cleaned:
        return cleaned

    return cleaned


def _replace_phrases(text: str, replacements: dict[str, str]) -> str:
    """Case-sensitive phrase replacement used by narration polish only."""
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def adapt_spanish_castellano(text: str) -> str:
    if not text:
        return text

    # Spain-Spanish / Castellano biasing for narration.
    # This cannot fully force the provider's accent, but it removes common LATAM
    # wording and pushes the spoken text toward a Spain bedtime register.
    replacements = {
        "ustedes": "vosotros",
        "Ustedes": "Vosotros",
        "computadora": "ordenador",
        "Computadora": "Ordenador",
        "celular": "móvil",
        "Celular": "Móvil",
        "carro": "coche",
        "Carro": "Coche",
        "auto": "coche",
        "Auto": "Coche",
        "manejar": "conducir",
        "Manejar": "Conducir",
        "platicar": "charlar",
        "Platicar": "Charlar",
        "enojado": "enfadado",
        "Enojado": "Enfadado",
        "enojada": "enfadada",
        "Enojada": "Enfadada",
        "sándwich": "bocadillo",
        "Sándwich": "Bocadillo",
        "emparedado": "bocadillo",
        "Emparedado": "Bocadillo",
        "muy lindo": "muy bonito",
        "Muy lindo": "Muy bonito",
        "muy linda": "muy bonita",
        "Muy linda": "Muy bonita",
        "lindo": "bonito",
        "Lindo": "Bonito",
        "linda": "bonita",
        "Linda": "Bonita",
        "chiquito": "pequeño",
        "Chiquito": "Pequeño",
        "chiquita": "pequeña",
        "Chiquita": "Pequeña",
        "calientito": "calentito",
        "Calientito": "Calentito",
        "calientita": "calentita",
        "Calientita": "Calentita",
        "lucecita": "luz suave",
        "Lucecita": "Luz suave",
        "dragoncito": "pequeño dragón",
        "Dragoncito": "Pequeño dragón",
        "estrellitas": "pequeñas estrellas",
        "Estrellitas": "Pequeñas estrellas",
        "chispitas": "destellos suaves",
        "Chispitas": "Destellos suaves",
        # Stronger neutral-dub / Latin-American diminutive cleanup.
        # Keep these phrase-based so we do not damage normal Spanish words like "bonito".
        "poquito": "poco",
        "Poquito": "Poco",
        "un poquito": "un poco",
        "Un poquito": "Un poco",
        "despacito": "despacio",
        "Despacito": "Despacio",
        "suavecito": "suave",
        "Suavecito": "Suave",
        "suavecita": "suave",
        "Suavecita": "Suave",
        "pequeñito": "pequeño",
        "Pequeñito": "Pequeño",
        "pequeñita": "pequeña",
        "Pequeñita": "Pequeña",
        "ratito": "rato",
        "Ratito": "Rato",
        "momentito": "momento",
        "Momentito": "Momento",
        "amiguito": "amigo",
        "Amiguito": "Amigo",
        "amiguita": "amiga",
        "Amiguita": "Amiga",
        "osito": "oso",
        "Osito": "Oso",
        "zorrito": "zorro",
        "Zorrito": "Zorro",
        "gatito": "gato",
        "Gatito": "Gato",
        "gatita": "gata",
        "Gatita": "Gata",
        "perrito": "perro",
        "Perrito": "Perro",
        "perrita": "perra",
        "Perrita": "Perra",
        "conejito": "conejo",
        "Conejito": "Conejo",
        "pajarito": "pájaro",
        "Pajarito": "Pájaro",
        "florecita": "flor",
        "Florecita": "Flor",
        "nubecita": "nube suave",
        "Nubecita": "Nube suave",
        "casita": "casa",
        "Casita": "Casa",
        "caminito": "camino",
        "Caminito": "Camino",
        "brillito": "brillo suave",
        "Brillito": "Brillo suave",
        "besito": "beso",
        "Besito": "Beso",
        "abracito": "abrazo",
        "Abracito": "Abrazo",
    }

    text = _replace_phrases(text, replacements)

    # Softer Spain bedtime rhythm, without adding narration instructions.
    phrase_replacements = [
        (r"\bde pronto\b", "entonces"),
        (r"\bDe pronto\b", "Entonces"),
        (r"\baventura muy especial\b", "camino tranquilo"),
        (r"\bAventura muy especial\b", "Camino tranquilo"),
        (r"\bpura curiosidad\b", "curiosidad tranquila"),
        (r"\bPura curiosidad\b", "Curiosidad tranquila"),
        (r"\bla noche quería contarle un secreto\b", "la noche parecía guardar algo especial"),
        (r"\bLa noche quería contarle un secreto\b", "La noche parecía guardar algo especial"),
        (r"\bcomenzó una aventura\b", "empezó un camino"),
        (r"\bComenzó una aventura\b", "Empezó un camino"),
        (r"\buna aventura mágica\b", "un camino lleno de magia"),
        (r"\bUna aventura mágica\b", "Un camino lleno de magia"),
        (r"\bmuy emocionado\b", "muy ilusionado"),
        (r"\bMuy emocionado\b", "Muy ilusionado"),
        (r"\bmuy emocionada\b", "muy ilusionada"),
        (r"\bMuy emocionada\b", "Muy ilusionada"),
        (r"\bse sintió emocionado\b", "se sintió ilusionado"),
        (r"\bSe sintió emocionado\b", "Se sintió ilusionado"),
        (r"\bse sintió emocionada\b", "se sintió ilusionada"),
        (r"\bSe sintió emocionada\b", "Se sintió ilusionada"),
    ]
    for pattern, replacement in phrase_replacements:
        text = re.sub(pattern, replacement, text)

    return text


def soften_bedtime_narration_text(text: str, language_code: str) -> str:
    """Light language-specific spoken-flow shaping before TTS.

    This must stay a text-only polish layer. It must not affect narration job
    ownership, chunking, playback, page status, or cache logic.
    """
    if not text:
        return text

    lang = (language_code or "en").lower()[:2]
    text = text.strip()

    # Normalize long dashes before language-specific shaping.
    text = text.replace(" — ", ". ").replace(" – ", ". ")

    if lang == "es":
        text = adapt_spanish_castellano(text)
        # Spain bedtime speech benefits from short, clear breaths. Keep commas
        # light and sentence pauses soft; avoid adding words or instructions.
        text = re.sub(r"([.!?])\s+", r"\1  ", text)
        text = re.sub(r"(;|:)\s+", r".  ", text)
        text = re.sub(r",\s+(y|pero|porque|cuando|mientras)\s+", r", \1 ", text, flags=re.IGNORECASE)
        return text.strip()

    if lang == "fr":
        # Encourage softer, less robotic French delivery through breathing rhythm,
        # without altering meaning or adding unspoken directions.
        text = re.sub(r"([.!?])\s+", r"\1  ", text)
        text = re.sub(r"(;|:)\s+", r".  ", text)
        text = re.sub(r"\btrès très\b", "très", text, flags=re.IGNORECASE)
        text = text.replace(" tout doucement ", " doucement ")
        text = text.replace(" petite petite ", " petite ")
        text = text.replace(" petit petit ", " petit ")
        text = text.replace("d’un seul coup", "doucement")
        text = text.replace("tout à coup", "alors")
        return text.strip()

    if lang == "de":
        text = re.sub(r"([.!?])\s+", r"\1  ", text)
        text = re.sub(r"(;|:)\s+", r".  ", text)
        text = text.replace("ganz ganz", "ganz")
        text = text.replace("plötzlich", "leise")
        text = text.replace("auf einmal", "dann")
        return text.strip()

    if lang == "it":
        text = re.sub(r"([.!?])\s+", r"\1  ", text)
        text = re.sub(r"(;|:)\s+", r".  ", text)
        text = re.sub(r"\bmolto molto\b", "molto", text, flags=re.IGNORECASE)
        text = text.replace("all’improvviso", "piano piano")
        text = text.replace("tutto a un tratto", "poi")
        return text.strip()

    if lang == "ja":
        # Keep Japanese wording intact. Only normalise whitespace so TTS receives
        # clean native text without introducing English-style punctuation.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*([。！？])\s*", r"\1", text)
        return text.strip()

    if lang == "ar":
        # Keep Arabic wording intact; use punctuation/spacing cleanup only.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s+([،؛؟.!])", r"\1", text)
        return text.strip()

    return text

class NarrationService:
    def __init__(
        self,
        story_repo: StoryRepository,
        user_repo: UserRepository,
        subscription_service: SubscriptionService,
        story_world_pronunciation_service: Optional[StoryWorldPronunciationService] = None,
    ):
        self.story_repo = story_repo
        self.user_repo = user_repo
        self.subscription_service = subscription_service
        self.story_world_pronunciation_service = story_world_pronunciation_service

    def get_narration_usage(self, user_id: str) -> dict:
        """
        Minimal usage payload for the current frontend.
        Safe stub to unblock narration testing.
        """
        return {
            "plan": "premium",
            "narrations_remaining": None,
            "daily_narrations_used": 0,
            "daily_limit": None,
            "can_narrate": True,
        }

    def resolve_language(self, story: dict, requested_language: Optional[str]) -> str:
        language_code = (
            requested_language
            or story.get("narration_language_code")
            or story.get("story_language_code")
            or story.get("language")
            or "en"
        )
        language_code = normalize_language_code(language_code, preserve_english_locale=True)
        base_lang = base_language_code(language_code)
        if base_lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail="Unsupported narration language")
        return language_code

    def default_voice_for_language(self, language_code: str) -> str:
        language_code = normalize_language_code(language_code, preserve_english_locale=True)

        # Product rule for English locales:
        # - UK English keeps the original Wise Owl brand and uses the UK/British voice path.
        # - US English uses Night Owl English and the American voice path.
        # This must happen before collapsing to the base language, otherwise en-US
        # and en-GB both become "en" and the app can route both to the same narrator.
        if language_code == "en-GB":
            return "wise_owl"
        if language_code == "en-US":
            return "night_owl_english"

        base_lang = base_language_code(language_code)
        return {
            "en": "night_owl_english",
            "es": "night_owl_spanish",
            "de": "night_owl_german",
            "fr": "night_owl_french",
            "it": "night_owl_italian",
            "ja": "night_owl_japanese",
            "ar": "night_owl_arabic",
        }.get(base_lang, "night_owl_english")

    def _preset_language_for_voice(self, voice: str) -> str:
        preset = VOICE_PRESETS.get(voice, {}) or {}
        preset_lang = (
            preset.get("language_code")
            or preset.get("language")
            or "all"
        )
        preset_lang = str(preset_lang).strip().lower()
        return preset_lang[:2] if preset_lang != "all" else "all"

    def _enforce_standard_voice_language(self, voice: str, language_code: str) -> str:
        """
        Runtime safety guard: standard narrator presets must always use their
        configured language. This prevents stale frontend params or saved story
        metadata from pairing Wise Owl with non-English narration. Parent Voice
        remains multilingual and is intentionally excluded.
        """
        if voice == "parent_voice":
            return language_code

        language_code = normalize_language_code(language_code, preserve_english_locale=True)
        requested_base_lang = base_language_code(language_code)
        preset_lang = self._preset_language_for_voice(voice)
        if preset_lang != "all" and preset_lang in SUPPORTED_LANGUAGES:
            if requested_base_lang != preset_lang:
                print(
                    f"[NARRATION] Enforcing narrator language voice={voice} "
                    f"requested_language={language_code} enforced_language={preset_lang}"
                )
                return preset_lang
            # Preserve en-US/en-GB for English so cache paths and TTS instructions
            # can distinguish American and British narration while using the same narrator preset.
            return language_code

        return language_code

    def _audio_cache_version(self, voice: str, language_code: str) -> str:
        # Keep Parent Voice cache-first and untouched. Parent Voice replay must
        # remain free and stable. Standard narrator cache bumps are only used
        # to avoid replaying older generated chunks.
        language_code = normalize_language_code(language_code, preserve_english_locale=True)
        if voice == "parent_voice":
            return DEFAULT_AUDIO_CACHE_VERSION

        # Locale-specific English caches are required because en-US and en-GB can
        # now use different narrator presets and OpenAI voices.
        standard_version = STANDARD_LANGUAGE_AUDIO_CACHE_VERSION.get(language_code) or STANDARD_LANGUAGE_AUDIO_CACHE_VERSION.get(base_language_code(language_code))
        if standard_version:
            return standard_version

        if voice == "wise_owl" and language_code in {"en-US", "en"}:
            return WISE_OWL_AUDIO_CACHE_VERSION
        return DEFAULT_AUDIO_CACHE_VERSION

    def resolve_voice(self, requested_voice: Optional[str], language_code: str) -> str:
        # Product rule: default narrator must remain Wise Owl / standard narrator family,
        # never Parent Voice unless explicitly selected.
        language_code = normalize_language_code(language_code, preserve_english_locale=True)
        default_voice = self.default_voice_for_language(language_code)

        if not requested_voice:
            return default_voice

        if requested_voice not in VOICE_PRESETS:
            raise HTTPException(status_code=400, detail="Unsupported narrator")

        # Parent Voice is allowed explicitly and handles multilingual separately.
        if requested_voice == "parent_voice":
            return requested_voice

        # Frontend can still send the previously selected English narrator while
        # the language changes. Correct English locale mismatches here so backend
        # storage, page-status, and generated audio use the intended narrator even
        # before the frontend selector state catches up.
        if language_code == "en-US" and requested_voice == "wise_owl":
            print("[NARRATION] Remapping en-US Wise Owl request to Night Owl English")
            return "night_owl_english"
        if language_code == "en-GB" and requested_voice == "night_owl_english":
            print("[NARRATION] Remapping en-GB Night Owl English request to Wise Owl")
            return "wise_owl"

        preset_lang = self._preset_language_for_voice(requested_voice)

        # Only allow exact-language, same base-language locale, or universal narrators.
        if preset_lang == "all" or preset_lang == language_code or preset_lang == base_language_code(language_code):
            return requested_voice

        # Frontend can momentarily send a stale narrator during language switches.
        # Fall back to the correct default narrator for the requested language.
        return default_voice

    def _cache_key(
        self,
        user_id: str,
        story_id: str,
        voice: str,
        language_code: str,
        pronunciation: Optional[str] = None,
    ) -> str:
        safe_pronunciation = (pronunciation or "").strip().lower()
        version = self._audio_cache_version(voice, language_code)
        return f"{user_id}:{story_id}:{voice}:{language_code}:{safe_pronunciation}:{version}"

    def _storage_prefix(self, user_id: str, story_id: str, voice: str, language_code: str) -> str:
        version = self._audio_cache_version(voice, language_code)
        return f"{user_id}/{story_id}/chunked/{voice}_{language_code}_{version}"

    def _storage_path(self, user_id: str, story_id: str, voice: str, language_code: str, page: int) -> str:
        return f"{self._storage_prefix(user_id, story_id, voice, language_code)}/page_{page}.mp3"


    def _list_story_chunk_folders(self, user_id: str, story_id: str) -> list[str]:
        prefix = f"{user_id}/{story_id}/chunked"
        try:
            items = self.story_repo.client.storage.from_("story-audio").list(prefix)
        except Exception:
            return []

        names: list[str] = []
        for item in items or []:
            name = item.get("name") or ""
            if name:
                names.append(name)
        return names

    def _list_existing_parent_voice_languages(self, user_id: str, story_id: str) -> list[str]:
        languages: set[str] = set()
        for name in self._list_story_chunk_folders(user_id, story_id):
            # Parent Voice cache folders are versioned, for example:
            # parent_voice_en_v5, parent_voice_es_v5.
            # Recognise both old unversioned folders and current versioned folders
            # so a story cannot be re-generated in a second Parent Voice language.
            match = re.match(r"parent_voice_([A-Za-z]{2}(?:[-_][A-Za-z]{2})?)(?:_v\d+)?$", name.strip())
            if match:
                languages.add(normalize_language_code(match.group(1), preserve_english_locale=True))
        return sorted(languages)

    def _signed_url(self, storage_path: str, expires_in: int = 3600) -> Optional[str]:
        try:
            result = self.story_repo.client.storage.from_("story-audio").create_signed_url(storage_path, expires_in)
            return result.get("signedURL") or result.get("signedUrl")
        except Exception:
            return None

    def _list_ready_pages(self, user_id: str, story_id: str, voice: str, language_code: str) -> list[int]:
        prefix = self._storage_prefix(user_id, story_id, voice, language_code)
        try:
            items = self.story_repo.client.storage.from_("story-audio").list(prefix)
        except Exception:
            return []

        ready: list[int] = []
        for item in items or []:
            name = item.get("name") or ""
            match = re.match(r"page_(\d+)\.mp3$", name)
            if match:
                ready.append(int(match.group(1)))
        return sorted(set(ready))


    def _spawn_chunked_worker(
        self,
        *,
        background_tasks: Optional[BackgroundTasks],
        job_id: str,
        user_id: str,
        story: dict,
        voice: str,
        language_code: str,
        parent_voice_id: Optional[str],
        start_page: int,
    ) -> None:
        """Schedule the deterministic narration worker through FastAPI BackgroundTasks.

        Do not use raw asyncio.create_task() from the request/service layer here.
        On Render/Uvicorn that detached task path can fail silently after the
        response returns, leaving the frontend polling 0/N forever. BackgroundTasks
        gives the request route explicit ownership of the worker launch while
        page-status remains a passive observer.
        """
        if background_tasks is None:
            job = _chunked_jobs.get(job_id)
            if job is not None:
                job["pages_generating"] = []
                job["status"] = "failed"
                job["last_error"] = "Narration worker was not scheduled: missing FastAPI BackgroundTasks"
                if 1 not in job.get("pages_ready", []):
                    job["pages_failed"] = sorted(set([*job.get("pages_failed", []), 1]))
            print(f"[NARRATION] Worker NOT scheduled job_id={job_id}: missing BackgroundTasks")
            return

        print(
            f"[NARRATION] Scheduling chunked worker via BackgroundTasks "
            f"job_id={job_id} story_id={story.get('id')} start_page={start_page}"
        )
        background_tasks.add_task(
            self._process_chunked_job,
            job_id=job_id,
            user_id=user_id,
            story=story,
            voice=voice,
            language_code=language_code,
            parent_voice_id=parent_voice_id,
            start_page=start_page,
        )

    def _clean_page_text(self, page_text: str) -> str:
        text = page_text or ""
        for marker in (
            "[NARRATION_START]",
            "[NARRATION_END]",
            "[whisper]",
            "[softly]",
            "[chuckle]",
            "[pause]",
            "[gently]",
        ):
            text = text.replace(marker, "")
        return text.strip()

    def _parse_csv_env(self, env_name: str) -> set[str]:
        raw = os.getenv(env_name, "")
        return {item.strip() for item in raw.split(",") if item.strip()}

    def _normalize_client_ip(self, client_ip: Optional[str]) -> str:
        if not client_ip:
            return "unknown"
        client_ip = client_ip.strip()
        if "," in client_ip:
            client_ip = client_ip.split(",")[0].strip()
        return client_ip or "unknown"

    def _record_event_with_window(
        self,
        bucket: dict[str, list[float]],
        key: str,
        *,
        now_ts: float,
        window_seconds: int,
    ) -> int:
        entries = bucket.get(key, [])
        cutoff = now_ts - window_seconds
        entries = [ts for ts in entries if ts >= cutoff]
        entries.append(now_ts)
        bucket[key] = entries
        return len(entries)

    def _enforce_parent_voice_security(self, *, user_id: str, client_ip: Optional[str]) -> None:
        ip = self._normalize_client_ip(client_ip)
        blocked_ips = self._parse_csv_env("PARENT_VOICE_BLOCKED_IPS")
        allowed_ips = self._parse_csv_env("PARENT_VOICE_ALLOWED_IPS")

        if allowed_ips and ip not in allowed_ips:
            print(f"[SECURITY] Parent Voice denied: IP not allowlisted ip={ip} user_id={user_id}")
            raise HTTPException(status_code=403, detail="Parent Voice is temporarily unavailable.")

        if ip in blocked_ips:
            print(f"[SECURITY] Parent Voice denied: blocked IP ip={ip} user_id={user_id}")
            raise HTTPException(status_code=403, detail="Parent Voice is temporarily unavailable.")

        if ip == "unknown":
            # Log but do not block unknown IPs; Render/proxy headers can vary.
            print(f"[SECURITY] Parent Voice request with unknown IP user_id={user_id}")
            return

        window_seconds = int(os.getenv("PARENT_VOICE_RATE_LIMIT_WINDOW_SECONDS", "3600"))
        max_per_ip = int(os.getenv("PARENT_VOICE_MAX_REQUESTS_PER_IP_PER_WINDOW", "5"))
        max_per_user = int(os.getenv("PARENT_VOICE_MAX_REQUESTS_PER_USER_PER_WINDOW", "5"))

        now_ts = datetime.now(timezone.utc).timestamp()
        ip_count = self._record_event_with_window(
            _parent_voice_ip_log,
            ip,
            now_ts=now_ts,
            window_seconds=window_seconds,
        )
        user_count = self._record_event_with_window(
            _parent_voice_user_log,
            user_id,
            now_ts=now_ts,
            window_seconds=window_seconds,
        )

        print(
            f"[SECURITY] Parent Voice request ip={ip} user_id={user_id} "
            f"ip_count={ip_count}/{max_per_ip} user_count={user_count}/{max_per_user}"
        )

        if ip_count > max_per_ip:
            raise HTTPException(
                status_code=429,
                detail="Too many Parent Voice requests from this network. Please try again later.",
            )

        if user_count > max_per_user:
            raise HTTPException(
                status_code=429,
                detail="Too many Parent Voice requests on this account. Please try again later.",
            )

    def _openai_tts_instructions(self, voice: str, language_code: Optional[str] = None) -> str:
        """Performance direction for OpenAI standard narrators.

        Keep this as a TTS-only quality layer. It must not affect narration
        ownership, chunking, page-status polling, playback, Parent Voice, or
        cache-first replay rules.
        """
        preset = VOICE_PRESETS.get(voice, {}) or {}
        requested_lang = normalize_language_code(language_code or preset.get("language_code") or "en-US", preserve_english_locale=True)
        lang = base_language_code(requested_lang)

        if requested_lang == "en-GB":
            return (
                "Read as a gentle British mother reading a continuous bedtime story."
		"Use a soft, warm British accent."
		"Treat every page as part of the exact same bedtime reading session."
		"Narrator identity, accent, pacing, energy level, warmth and emotional tone must remain identical throughout the story."
		"Never reinterpret the narrator."
		"Never change voice character."
		"Never change energy level."
		"Never sound like a different speaker."
		"Never switch between storyteller styles."
		"Speak softly and naturally as though reading to one sleepy child."
		"Avoid theatrical, audiobook, presenter, commercial, stage-performance or dramatic delivery."
    	    )  

        if requested_lang == "en-US":
            return (
                "Read as the same calm American parent telling one continuous bedtime story in natural US English. "
                "Use a warm, gentle, reassuring American accent and bedtime-soft pacing. "
                "Treat every page as a continuation of the exact same recording session. Narrator identity, age, accent, speed, energy level, emotional tone, warmth, and pacing must remain identical across all pages. Consistency is more important than expressive variation. Never reinterpret the narrator between pages, and never make later pages sound slower, older, more serious, more distant, or like a different speaker. "
                "Start cleanly and gently on the first word, without an audible breath, gulp, mouth sound, or hard consonant attack. For every page after page one, begin immediately in the same American accent, voice character, rhythm, warmth, and pacing as the rest of the passage, as though continuing one uninterrupted bedtime reading. Do not make the first sentence or first paragraph sound like a new recording, warm-up, reset, different accent, slower introduction, or separate narration take. When continuing later story sections, flow naturally from the previous page and do not add an audible inhale, gulp, mouth sound, or restart effect between pages. "
                "Keep the delivery comforting, intimate, unhurried, and emotionally safe, as if helping a child settle peacefully for sleep. Avoid robotic, theatrical, commercial, audiobook-announcer, cartoon-granny, or overly energetic delivery."
            )

        if lang == "es":
            return (
                "Read as the same calm parent from Madrid, Spain telling one continuous bedtime story in Castilian Spanish. "
                "Use a clearly peninsular Spanish accent from Spain, never a Latin American or neutral-dub accent. "
                "Pronounce the letters 'z' and soft 'c' before e or i with the traditional Castilian Spain pronunciation. "
                "Use natural Spain-Spanish rhythm, intonation, vocabulary, and bedtime pacing. "
                "Keep the delivery warm, soft, intimate, sleepy, and consistent, with gentle natural pauses. "
                "Treat every page as a continuation of the exact same recording session. Narrator identity, age, accent, speed, energy level, emotional tone, warmth, and pacing must remain identical across all pages. Consistency is more important than expressive variation. Never reinterpret the narrator between pages, and never make later pages sound slower, older, more serious, more distant, or like a different speaker. "
                "Start cleanly and gently on the first word, without an audible breath, gulp, mouth sound, or hard consonant attack. For every page after page one, begin immediately in the same accent, voice character, rhythm, warmth, and pacing as the rest of the passage, as though continuing one uninterrupted bedtime reading. Do not make the first sentence or first paragraph sound like a new recording, warm-up, reset, different accent, slower introduction, or separate narration take. When continuing later story sections, flow naturally from the previous page and do not add an audible inhale, gulp, mouth sound, or restart effect between pages. Keep expressive variation subtle and controlled, without changing the narrator identity or slowing the page. Do not allow the warmth, emotional engagement, or storytelling energy to fade as the passage continues. Give important moments gentle warmth while keeping quiet moments soft and comforting. Vary sentence endings naturally, but keep the same narrator pace and voice character throughout the whole page. "
                "Avoid sounding theatrical, commercial, robotic, cartoon-like, overly bright, or newly re-cast between pages. "
                "Speak slowly enough for a young child at bedtime, with tender reassurance and a peaceful tone, but do not slow down later pages or later paragraphs compared with earlier pages."
            )

        if lang == "fr":
            return (
                "Read as a warm French parent telling a bedtime story to a young child. "
                "Use soft, natural French intonation with gentle breathing pauses and a calm sleepy rhythm. "
                "Start cleanly and gently on the first word, without an audible breath, gulp, mouth sound, or hard consonant attack. For every page after page one, begin immediately in the same accent, voice character, rhythm, warmth, and pacing as the rest of the passage, as though continuing one uninterrupted bedtime reading. Do not make the first sentence or first paragraph sound like a new recording, warm-up, reset, different accent, slower introduction, or separate narration take. When continuing later story sections, flow naturally from the previous page and do not add an audible inhale, gulp, mouth sound, or restart effect between pages. Maintain gentle expressive variation from beginning to end. Do not allow the warmth, emotional engagement, or storytelling energy to fade as the passage continues. Give important moments slightly more warmth and emphasis while keeping quiet moments soft and comforting. Vary sentence endings naturally so the narration never becomes flat, monotone, or lower-energy near the end of a page. "
                "Avoid robotic, formal, academic, theatrical, or announcement-style delivery. "
                "Keep the voice tender, reassuring, emotionally warm, and suitable for falling asleep."
            )

        if lang == "de":
            return (
                "Read as a warm German parent telling a bedtime story to a young child. "
                "Use soft natural German intonation, gentle pauses, and a slow comforting bedtime rhythm. "
                "Start cleanly and gently on the first word, without an audible breath, gulp, mouth sound, or hard consonant attack. For every page after page one, begin immediately in the same accent, voice character, rhythm, warmth, and pacing as the rest of the passage, as though continuing one uninterrupted bedtime reading. Do not make the first sentence or first paragraph sound like a new recording, warm-up, reset, different accent, slower introduction, or separate narration take. When continuing later story sections, flow naturally from the previous page and do not add an audible inhale, gulp, mouth sound, or restart effect between pages. Maintain gentle expressive variation from beginning to end. Do not allow the warmth, emotional engagement, or storytelling energy to fade as the passage continues. Give important moments slightly more warmth and emphasis while keeping quiet moments soft and comforting. Vary sentence endings naturally so the narration never becomes flat, monotone, or lower-energy near the end of a page. "
                "Avoid stiff, robotic, formal, theatrical, or audiobook-announcer delivery. "
                "Keep the voice calm, tender, reassuring, and sleepy."
            )

        if lang == "it":
            return (
                "Read as a warm Italian parent telling a bedtime story to a young child. "
                "Use soft natural Italian intonation, gentle musical rhythm, and calm bedtime pacing. "
                "Start cleanly and gently on the first word, without an audible breath, gulp, mouth sound, or hard consonant attack. For every page after page one, begin immediately in the same accent, voice character, rhythm, warmth, and pacing as the rest of the passage, as though continuing one uninterrupted bedtime reading. Do not make the first sentence or first paragraph sound like a new recording, warm-up, reset, different accent, slower introduction, or separate narration take. When continuing later story sections, flow naturally from the previous page and do not add an audible inhale, gulp, mouth sound, or restart effect between pages. Maintain gentle expressive variation from beginning to end. Do not allow the warmth, emotional engagement, or storytelling energy to fade as the passage continues. Give important moments slightly more warmth and emphasis while keeping quiet moments soft and comforting. Vary sentence endings naturally so the narration never becomes flat, monotone, or lower-energy near the end of a page. "
                "Avoid robotic, theatrical, overly energetic, or announcement-style delivery. "
                "Keep the voice tender, reassuring, dreamy, and suitable for sleep."
            )

        if lang == "ja":
            return (
                "Read as the same calm Japanese parent telling one continuous bedtime story in natural Japanese. "
                "Use a warm, gentle, reassuring Japanese delivery with clear native pronunciation and soft bedtime pacing. "
                "Treat every page as part of the exact same recording session. Keep narrator identity, speed, warmth, energy, and emotional tone consistent across pages. "
                "Start each page cleanly on the first word without an audible breath, gulp, mouth sound, or reset effect. "
                "Avoid exaggerated anime, announcer, commercial, theatrical, or overly energetic delivery. "
                "Keep the voice intimate, natural, peaceful, and easy for a young child to follow."
            )

        if lang == "ar":
            return (
                "Read as the same calm Arabic-speaking parent telling one continuous bedtime story in clear, natural Arabic. "
                "Use warm, gentle, reassuring pronunciation and a soft bedtime rhythm. "
                "Treat every page as part of the exact same recording session. Keep narrator identity, speed, warmth, energy, and emotional tone consistent across pages. "
                "Start each page cleanly on the first word without an audible breath, gulp, mouth sound, or reset effect. "
                "Avoid announcer, newsreader, theatrical, commercial, or overly formal delivery. "
                "Keep the voice intimate, natural, peaceful, and easy for a young child to follow."
            )

        return (
            "Read as a calm, warm bedtime storyteller for a young child, with the gentle reassurance of a loving grandparent. "
            "Use soft, sleepy pacing, natural breathing pauses, and tender sentence endings. "
            "Start cleanly and gently on the first word, without an audible breath, gulp, mouth sound, or hard consonant attack. For every page after page one, begin immediately in the same accent, voice character, rhythm, warmth, and pacing as the rest of the passage, as though continuing one uninterrupted bedtime reading. Do not make the first sentence or first paragraph sound like a new recording, warm-up, reset, different accent, slower introduction, or separate narration take. When continuing later story sections, flow naturally from the previous page and do not add an audible inhale, gulp, mouth sound, or restart effect between pages. Maintain gentle expressive variation from beginning to end. Do not allow the warmth, emotional engagement, or storytelling energy to fade as the passage continues. Give important moments slightly more warmth and emphasis while keeping quiet moments soft and comforting. Vary sentence endings naturally so the narration never becomes flat, monotone, or lower-energy near the end of a page. "
            "Keep the delivery comforting, intimate, unhurried, and emotionally safe, as if helping a child settle peacefully for sleep. "
            "Avoid robotic, theatrical, commercial, audiobook-announcer, cartoon-granny, or overly energetic delivery."
        )

    async def _generate_openai_tts(self, text: str, voice: str, language_code: Optional[str] = None) -> bytes:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")

        provider_voice = VOICE_PRESETS.get(voice, {}).get("voice_id") or "shimmer"
        instructions = self._openai_tts_instructions(voice, language_code)

        retries = 2

        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, read=180.0)) as client:
                    response = await client.post(
                        "https://api.openai.com/v1/audio/speech",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "gpt-4o-mini-tts",
                            "voice": provider_voice,
                            "input": text,
                            "instructions": instructions,
                            "format": "mp3",
                        },
                    )

                if response.status_code != 200:
                    raise RuntimeError(f"OpenAI TTS failed: {response.status_code} {response.text[:300]}")

                return response.content

            except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                print(f"[NARRATION] TTS timeout attempt {attempt+1}/{retries+1}")

                if attempt == retries:
                    raise RuntimeError("TTS failed after retries")

                await asyncio.sleep(1.5)  # small backoff

    async def _generate_elevenlabs_tts(self, text: str, voice_id: str, language_code: str) -> bytes:
        api_key = os.getenv("ELEVENLABS_API_KEY", "")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not configured")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": api_key,
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "language_code": base_language_code(language_code),
                    "voice_settings": {
                        # Keep the cloned parent's identity while allowing enough
                        # natural variation for conversational bedtime rhythm.
                        "stability": 0.55,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        # Slightly slower than natural speech, while still preserving a flowing
                        # parent-like bedtime rhythm.
                        "speed": 0.89,
                        "use_speaker_boost": True,
                    },
                    "output_format": "mp3_44100_128",
                },
            )

        if response.status_code != 200:
            raise RuntimeError(f"ElevenLabs TTS failed: {response.status_code} {response.text[:300]}")

        return response.content

    async def _upload_audio(self, storage_path: str, audio_bytes: bytes) -> None:
        self.story_repo.client.storage.from_("story-audio").upload(
            storage_path,
            audio_bytes,
            {"content-type": "audio/mpeg", "upsert": "true"},
        )


    def _translation_output_matches_target(
        self,
        original_text: str,
        translated_text: str,
        *,
        source_lang: str,
        target_lang: str,
    ) -> bool:
        """Reject obvious wrong-language translation results before TTS.

        This is intentionally conservative. It catches the failure mode seen in
        production where an Arabic -> Italian request returned Arabic unchanged,
        while avoiding heavyweight language detection on the Page-1 path.
        """
        candidate = (translated_text or "").strip()
        original = (original_text or "").strip()
        if not candidate:
            return False

        if source_lang and source_lang != target_lang:
            normalized_original = re.sub(r"\s+", " ", original).strip().casefold()
            normalized_candidate = re.sub(r"\s+", " ", candidate).strip().casefold()
            if normalized_original == normalized_candidate:
                return False

        total_letters = max(1, sum(1 for ch in candidate if ch.isalpha()))
        arabic_chars = sum(1 for ch in candidate if "\u0600" <= ch <= "\u06ff")
        japanese_chars = sum(
            1
            for ch in candidate
            if ("\u3040" <= ch <= "\u30ff") or ("\u4e00" <= ch <= "\u9fff")
        )
        arabic_ratio = arabic_chars / total_letters
        japanese_ratio = japanese_chars / total_letters

        if target_lang == "ar":
            return arabic_ratio >= 0.20
        if source_lang == "ar" and target_lang != "ar" and arabic_ratio >= 0.20:
            return False

        if target_lang == "ja":
            return japanese_ratio >= 0.20
        if source_lang == "ja" and target_lang != "ja" and japanese_ratio >= 0.20:
            return False

        return True

    async def _translate_text(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        if not text:
            return text

        target = base_language_code(target_lang or "en")
        source = base_language_code(source_lang) if source_lang else ""

        if source and source == target:
            print(f"[TRANSLATE] Skipping translation source_lang={source} target_lang={target}")
            return text

        if not source and target == "en":
            return text

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"[TRANSLATE] OPENAI_API_KEY missing for source_lang={source or 'unknown'} target_lang={target}")
            raise RuntimeError("OPENAI_API_KEY not configured for translation")

        language_names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "ja": "Japanese",
            "ar": "Arabic",
        }
        source_name = language_names.get(source, source or "the original language")
        target_name = language_names.get(target, target)

        last_error: Optional[Exception] = None
        for attempt in range(1, 3):
            try:
                system_prompt = (
                    f"Translate the following children's bedtime story text from {source_name} into {target_name}. "
                    f"The entire output MUST be written in {target_name}. Never return the source-language text. "
                    "Keep it warm, magical, emotionally comforting, and natural for young children. "
                    "Preserve names, tone, emotional pacing, and bedtime softness. "
                    "Do not translate literally. "
                    "Write as if the story was originally written in the target language. "
                    "If translating into Spanish, use Spain Spanish (Castellano), not Latin American Spanish. "
                    "If translating into French, use warm, natural bedtime French rather than formal or academic phrasing. "
                    "If translating into Italian, use warm, natural Italian with a gentle bedtime rhythm. "
                    "If translating into German, use warm, natural German suitable for children, not stiff or academic phrasing. "
                    "If translating into Japanese, use natural child-friendly Japanese that sounds written originally for a bedtime read-aloud, not literal translation. "
                    "If translating into Arabic, use clear, natural child-friendly Arabic with warm bedtime phrasing and avoid stiff machine-translated wording. "
                    "Return only the translated text."
                )
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=8.0,
                        read=18.0,
                        write=8.0,
                        pool=8.0,
                    )
                ) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": text},
                            ],
                            "temperature": 0.1,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    translated = data["choices"][0]["message"]["content"].strip()

                if not self._translation_output_matches_target(
                    text,
                    translated,
                    source_lang=source,
                    target_lang=target,
                ):
                    raise RuntimeError(
                        f"Translation output did not match requested target language {target_name}"
                    )

                print(
                    f"[TRANSLATE] Translation success source_lang={source or 'unknown'} "
                    f"target_lang={target} attempt={attempt} "
                    f"input_preview={text[:120]!r} output_preview={translated[:120]!r}"
                )
                return translated
            except Exception as e:
                last_error = e
                print(
                    f"[TRANSLATE] Attempt {attempt}/2 failed source_lang={source or 'unknown'} "
                    f"target_lang={target}: {repr(e)} input_preview={text[:120]!r}"
                )

        # Cross-language narration must fail closed. Returning the original source
        # text here would create and cache audio under the wrong narration language.
        raise RuntimeError(
            f"Translation failed from {source_name} to {target_name} after 2 attempts: {last_error}"
        )


    def _story_world_pronunciation_provider(
        self,
        *,
        voice_mode: str,
    ) -> str:
        """Map narration mode to pronunciation-provider configuration."""
        return "elevenlabs" if voice_mode == "parent" else "openai"

    def _apply_story_world_pronunciation(
        self,
        *,
        text: str,
        story_world_id: Optional[str],
        language_code: str,
        voice: str,
        voice_mode: str,
    ) -> str:
        """Apply the shared verified pronunciation layer to narration-only text.

        Despite the legacy method name, this now serves both ordinary PillowTales
        bedtime stories and Story Worlds:
        - ordinary story: global pronunciation rows only
        - Story World: global rows + world-specific rows

        The parent's explicit child-name pronunciation is applied before this
        method, so it remains the highest-priority user override.

        Pronunciation quality must never block Page 1 narration startup.
        """
        if not text or not self.story_world_pronunciation_service:
            return text

        provider = self._story_world_pronunciation_provider(voice_mode=voice_mode)
        try:
            adjusted = self.story_world_pronunciation_service.apply(
                text=text,
                world_id=(str(story_world_id) if story_world_id else None),
                language_code=language_code,
                provider=provider,
                voice=voice,
            )
            if adjusted != text:
                scope = f"story_world:{story_world_id}" if story_world_id else "global"
                print(
                    f"[NARRATION] Applied shared pronunciation "
                    f"scope={scope} provider={provider} voice={voice} "
                    f"language={language_code}"
                )
            return adjusted
        except Exception as exc:
            # Non-blocking by design: never delay or fail Page 1 because a
            # pronunciation lookup/override is unavailable.
            print(
                f"[NARRATION] Shared pronunciation skipped after error "
                f"world_id={story_world_id} provider={provider} voice={voice}: {repr(exc)}"
            )
            return text

    async def _generate_page_audio(
        self,
        *,
        user_id: str,
        story_id: str,
        page: int,
        page_text: str,
        voice: str,
        language_code: str,
        voice_mode: str,
        parent_voice_id: Optional[str],
        child_name: Optional[str] = None,
        child_name_pronunciation: Optional[str] = None,
        story_language_code: Optional[str] = None,
        story_world_id: Optional[str] = None,
        story_world_slug: Optional[str] = None,
    ) -> tuple[str, str]:
        page_text = self._clean_page_text(page_text)

        # PERFORMANCE SAFETY:
        # Same-language narration must stay on the fast path.
        # Do not send page text through the translation model unless the parent
        # explicitly selected a different narration language from the story language.
        story_lang = base_language_code(story_language_code or language_code or "en")
        narration_lang = base_language_code(language_code or "en")

        if story_lang == narration_lang:
            translated = page_text
            print(
                f"[NARRATION] Skipping translation for same-language narration "
                f"story_lang={story_lang} narration_lang={narration_lang}"
            )
        else:
            translated = await self._translate_text(page_text, narration_lang, story_lang)

        print(f"[NARRATION] Generating page {page} with voice={voice} language={language_code}")
        print(f"[NARRATION] Original page text preview={page_text[:160]!r}")
        print(f"[NARRATION] Narration text preview for {language_code}={translated[:160]!r}")
        tts_text = clean_text_for_tts(translated)
        tts_text = apply_pronunciation(tts_text, child_name, child_name_pronunciation)

        resolved_story_world_id = story_world_id
        if (
            not resolved_story_world_id
            and story_world_slug
            and self.story_world_pronunciation_service
        ):
            try:
                resolved_story_world_id = self.story_world_pronunciation_service.resolve_world_id(
                    story_world_slug
                )
                if resolved_story_world_id:
                    print(
                        f"[NARRATION] Resolved Story World pronunciation context "
                        f"slug={story_world_slug} world_id={resolved_story_world_id}"
                    )
            except Exception as exc:
                print(
                    f"[NARRATION] Story World id resolution failed "
                    f"slug={story_world_slug}: {repr(exc)}"
                )

        tts_text = self._apply_story_world_pronunciation(
            text=tts_text,
            story_world_id=resolved_story_world_id,
            language_code=language_code,
            voice=voice,
            voice_mode=voice_mode,
        )

        if voice_mode == "parent":
            # Parent Voice uses ElevenLabs and previously had good natural timing.
            # Keep it conservative, but preserve natural pauses after full stops
            # across all supported languages. Do not apply standard narrator
            # accent/bedtime wording shaping here.
            tts_text = prepare_parent_voice_text(tts_text, language_code)
        else:
            # Standard OpenAI narrators keep the bedtime/accent shaping added for
            # smoother page transitions and improved non-English delivery.
            tts_text = soften_bedtime_narration_text(tts_text, language_code)
            tts_text = prepare_narration_text(tts_text)

            # Keep page 1 as fast as possible for bedtime startup speed.
            # Apply the soft anti-gulp lead-in only for later pages where
            # page-transition mouth artefacts are more noticeable.
            if page > 1:
                tts_text = add_soft_chunk_leadin(tts_text)

        if not tts_text:
            raise RuntimeError("Page has no text")

        # DIAGNOSTIC ONLY:
        # Log the exact final text that will be handed to the TTS provider.
        # This is intentionally placed after Story World pronunciation handling
        # and all narration-only text shaping, immediately before provider TTS.
        print(
            f"[NARRATION] Final TTS text preview page={page} voice={voice} "
            f"language={language_code} text={tts_text[:300]!r}"
        )

        used_mode = voice_mode
        if voice_mode == "parent" and parent_voice_id:
            try:
                audio = await self._generate_elevenlabs_tts(tts_text, parent_voice_id, language_code)
            except Exception:
                # Bulletproof fallback: keep the whole job alive with standard narration.
                used_mode = "fallback_tts"
                fallback_voice = self.default_voice_for_language(language_code)
                audio = await self._generate_openai_tts(tts_text, fallback_voice, language_code)
        else:
            audio = await self._generate_openai_tts(tts_text, voice, language_code)

        storage_path = self._storage_path(user_id, story_id, voice, language_code, page)
        await self._upload_audio(storage_path, audio)
        return storage_path, used_mode

    def _refund_parent_voice_credit_once(self, user_id: str, job: dict) -> None:
        if not job.get("credit_charged") or job.get("credit_refunded"):
            return
        try:
            wallet = self.user_repo.get_parent_voice_wallet(user_id)
            credits = int(wallet.get("credits", 0))
            intro_used = bool(wallet.get("intro_used", False))

            if job.get("intro_charged"):
                self.user_repo.save_parent_voice_wallet(
                    user_id,
                    credits=credits,
                    intro_used=False,
                )
                print(f"[NARRATION] Refunded Parent Voice intro user_id={user_id}")
            else:
                self.user_repo.save_parent_voice_wallet(
                    user_id,
                    credits=credits + 1,
                    intro_used=intro_used,
                )
                print(f"[NARRATION] Refunded Parent Voice credit user_id={user_id}")

            job["credit_refunded"] = True
        except Exception as refund_err:
            print(f"[NARRATION] Parent Voice refund failed user_id={user_id}: {repr(refund_err)}")

    async def _process_chunked_job(
        self,
        *,
        job_id: str,
        user_id: str,
        story: dict,
        voice: str,
        language_code: str,
        parent_voice_id: Optional[str],
        start_page: int = 1,
    ) -> None:
        """Single-owner deterministic chunk worker.

        Backend ownership rule:
        - Generate the requested/priority page first, normally page 1.
        - As soon as page 1 is uploaded, it is marked ready for frontend playback.
        - Keep this same worker alive to prewarm pages 2+ as story text expands.
        - page-status remains a passive observer only.

        This replaces the fragile separate expansion watcher layer that could race
        the main worker and leave the frontend polling forever.
        """
        job = _chunked_jobs.get(job_id)
        if not job:
            return

        started = asyncio.get_event_loop().time()
        max_seconds = float(os.getenv("NARRATION_CHUNK_WORKER_MAX_SECONDS", "240"))
        requested_start = max(1, int(start_page or 1))
        job["priority_page"] = requested_start
        job["status"] = "generating"

        async def mark_ready(idx: int, storage_path: str, actual_mode: Optional[str] = None) -> None:
            if actual_mode:
                job["voice_mode"] = actual_mode
            if idx not in job["pages_ready"]:
                job["pages_ready"].append(idx)
            job["page_paths"][idx] = storage_path
            job["pages_ready"] = sorted(set(job["pages_ready"]))
            if idx in job["pages_failed"]:
                job["pages_failed"].remove(idx)
            job["last_error"] = None
            job["pages_generating"] = []

            if idx == 1:
                try:
                    first_url = self._signed_url(storage_path)
                    self.story_repo.update(
                        story_id,
                        user_id,
                        {
                            "audio_status": "ready",
                            "audio_url": first_url,
                            "audio_created_at": datetime.now(timezone.utc).isoformat(),
                            "audio_language_code": language_code,
                            "audio_voice_id": voice,
                            "narration_language_code": language_code,
                        },
                    )
                except Exception as update_err:
                    print(f"[NARRATION] Page 1 ready metadata update failed story_id={story_id}: {repr(update_err)}")

        try:
            story_id = story["id"]
            initial_voice_mode = "parent" if voice == "parent_voice" and parent_voice_id else "standard"
            job["voice_mode"] = job.get("voice_mode") or initial_voice_mode

            while asyncio.get_event_loop().time() - started < max_seconds:
                try:
                    fresh_story = self.story_repo.get(story_id, user_id) or story
                except Exception as read_err:
                    print(f"[NARRATION] Worker story refresh failed story_id={story_id}: {repr(read_err)}")
                    fresh_story = story

                pages = fresh_story.get("pages") or []
                pages_count = len(pages)
                expected_total_pages = max(int(job.get("total_pages") or 0), self._expected_total_pages(fresh_story))
                job["total_pages"] = expected_total_pages

                if not pages:
                    await asyncio.sleep(0.75)
                    continue

                storage_ready = set(self._list_ready_pages(user_id, story_id, voice, language_code))
                if storage_ready:
                    for ready_idx in storage_ready:
                        ready_path = self._storage_path(user_id, story_id, voice, language_code, ready_idx)
                        if ready_idx not in job["pages_ready"]:
                            job["pages_ready"].append(ready_idx)
                        job["page_paths"][ready_idx] = ready_path
                    job["pages_ready"] = sorted(set(job["pages_ready"]))

                ready = set(job.get("pages_ready", []))
                generation_status = str(fresh_story.get("generation_status") or "complete").strip().lower()
                text_complete = generation_status in {"complete", "completed", "failed"} or pages_count >= expected_total_pages

                if text_complete and expected_total_pages > 0 and len(ready) >= expected_total_pages:
                    job["pages_generating"] = []
                    job["status"] = "all_ready"
                    print(f"[NARRATION] Chunked job all_ready story_id={story_id} pages={len(ready)}/{expected_total_pages}")
                    return

                available_missing = [idx for idx in range(1, pages_count + 1) if idx not in ready]

                if available_missing:
                    if requested_start in available_missing:
                        idx = requested_start
                    elif 1 in available_missing:
                        idx = 1
                    elif 2 in available_missing:
                        idx = 2
                    else:
                        idx = available_missing[0]

                    storage_path = self._storage_path(user_id, story_id, voice, language_code, idx)

                    # IMPORTANT: do not use create_signed_url() as an existence check.
                    # Supabase can return a signed URL for a path even when the object
                    # has not been uploaded yet. Storage list results above are the
                    # source of truth for ready pages; missing pages must enter TTS.
                    page_text = pages[idx - 1]
                    job["pages_generating"] = [idx]
                    job["status"] = "generating"
                    print(
                        f"[NARRATION] Deterministic worker generating page {idx} "
                        f"story_id={story_id} pages_count={pages_count} expected={expected_total_pages}"
                    )
                    storage_path, actual_mode = await self._generate_page_audio(
                        user_id=user_id,
                        story_id=story_id,
                        page=idx,
                        page_text=page_text,
                        voice=voice,
                        language_code=language_code,
                        voice_mode=job["voice_mode"],
                        parent_voice_id=parent_voice_id,
                        child_name=fresh_story.get("child_name"),
                        child_name_pronunciation=fresh_story.get("child_name_pronunciation"),
                        story_language_code=self.resolve_language(
                            fresh_story,
                            fresh_story.get("story_language_code")
                            or fresh_story.get("language")
                            or fresh_story.get("language_code")
                            or fresh_story.get("story_language")
                            or fresh_story.get("preferred_language"),
                        ),
                        story_world_id=(
                            fresh_story.get("story_world_id")
                            or fresh_story.get("storyWorldId")
                        ),
                        story_world_slug=(
                            fresh_story.get("story_world_slug")
                            or fresh_story.get("storyWorldSlug")
                        ),
                    )
                    await mark_ready(idx, storage_path, actual_mode)
                    job["status"] = "page_ready" if len(job.get("pages_ready", [])) < expected_total_pages else "all_ready"
                    await asyncio.sleep(0.05)
                    continue

                # No available text pages are missing. Keep the single worker alive
                # briefly for lean story expansion so page 2+ can be prewarmed as soon
                # as text arrives, without a separate watcher racing this worker.
                job["pages_generating"] = []
                job["status"] = "page_ready" if job.get("pages_ready") else "generating"

                if text_complete:
                    # Text generation ended but there are no available pages left to generate.
                    # If we have at least the existing text pages ready, finish cleanly.
                    ready_count = len(set(job.get("pages_ready", [])))
                    if ready_count >= pages_count:
                        job["status"] = "all_ready" if ready_count >= expected_total_pages else "page_ready"
                        return

                await asyncio.sleep(0.75)

            # Worker timed out waiting for story text expansion. Do not fail if page 1
            # is ready; playback can continue and the frontend can retry/request again.
            job["pages_generating"] = []
            if job.get("pages_ready"):
                job["status"] = "page_ready"
                print(
                    f"[NARRATION] Chunked worker timed out after page readiness "
                    f"story_id={story_id} ready={job.get('pages_ready')} total={job.get('total_pages')}"
                )
            else:
                job["status"] = "failed"
                job["last_error"] = "Narration worker timed out before page 1 was ready"
                job["pages_failed"] = [1]
                self._refund_parent_voice_credit_once(user_id, job)
                print(f"[NARRATION] Chunked worker timed out before page 1 story_id={story_id}")

        except Exception as e:
            story_id = story.get("id", "unknown") if isinstance(story, dict) else "unknown"
            print(f"[NARRATION] Chunked worker crashed story_id={story_id}: {repr(e)}")
            job["pages_generating"] = []
            job["status"] = "failed"
            job["last_error"] = str(e)
            if 1 not in job.get("pages_ready", []):
                job["pages_failed"] = sorted(set([*job.get("pages_failed", []), 1]))
            self._refund_parent_voice_credit_once(user_id, job)

    def _get_story_for_user(self, story_id: str, user_id: str) -> dict:
        story = self.story_repo.get(story_id, user_id)
        if not story:
            raise HTTPException(status_code=404, detail="Story not found")
        if not isinstance(story.get("pages"), list) or not story.get("pages"):
            raise HTTPException(status_code=400, detail="Story has no pages")
        return story

    def _get_subscription(self, user_id: str) -> tuple[dict, SubscriptionResponse]:
        profile = self.user_repo.get_profile(user_id) or {}
        subscription = self.subscription_service.get_subscription(user_id, profile.get("email"))
        return profile, subscription

    def _expected_total_pages(self, story: dict) -> int:
        """Return the best known final page count without trusting early 1-page text.

        Lean story generation returns page 1 first and stores expected_pages=7.
        Narration must use that expected count for progress/prewarm decisions, while
        only generating pages whose text actually exists.
        """
        pages_count = len(story.get("pages") or [])
        raw_expected = story.get("expected_pages") or story.get("total_pages") or pages_count
        try:
            expected = int(raw_expected)
        except Exception:
            expected = pages_count
        return max(pages_count, expected, 1)

    async def _watch_story_expansion_and_continue_job(
        self,
        *,
        job_id: str,
        user_id: str,
        story_id: str,
        voice: str,
        language_code: str,
        parent_voice_id: Optional[str],
        preferred_next_page: int = 2,
        max_seconds: float = 120.0,
    ) -> None:
        """Request-owned watcher for lean story expansion.

        page-status must remain passive. This watcher is created only by
        request_narration(), so narration generation ownership stays with the
        request endpoint while allowing page 2+ to start as soon as text expands
        from 1 page to the final story.
        """
        started = asyncio.get_event_loop().time()
        last_seen_pages = 0

        while asyncio.get_event_loop().time() - started < max_seconds:
            await asyncio.sleep(0.75)

            job = _chunked_jobs.get(job_id)
            if not job or job.get("last_error"):
                return

            try:
                fresh_story = self.story_repo.get(story_id, user_id)
            except Exception as exc:
                print(f"[NARRATION] Story expansion watcher read failed story_id={story_id}: {repr(exc)}")
                continue

            if not fresh_story:
                return

            pages = fresh_story.get("pages") or []
            pages_count = len(pages)
            expected_total = self._expected_total_pages(fresh_story)
            generation_status = str(fresh_story.get("generation_status") or "complete").strip().lower()
            text_complete = generation_status in {"complete", "completed", "failed"} or pages_count >= expected_total

            if expected_total > int(job.get("total_pages") or 0):
                job["total_pages"] = expected_total

            if pages_count <= last_seen_pages and not text_complete:
                continue
            last_seen_pages = max(last_seen_pages, pages_count)

            ready = set(job.get("pages_ready", [])) | set(self._list_ready_pages(user_id, story_id, voice, language_code))
            generating = set(job.get("pages_generating", []))
            available_missing = [i for i in range(1, pages_count + 1) if i not in ready]

            if available_missing and not generating:
                if preferred_next_page in available_missing:
                    start_page = preferred_next_page
                else:
                    start_page = available_missing[0]

                job["pages_generating"] = [start_page]
                job["status"] = "generating"
                job["total_pages"] = expected_total
                print(
                    f"[NARRATION] Story text expanded; continuing chunked worker "
                    f"story_id={story_id} start_page={start_page} pages_count={pages_count} expected={expected_total}"
                )
                self._spawn_chunked_worker(
                    background_tasks=None,
                    job_id=job_id,
                    user_id=user_id,
                    story=fresh_story,
                    voice=voice,
                    language_code=language_code,
                    parent_voice_id=parent_voice_id,
                    start_page=start_page,
                )

            if text_complete:
                # If the full story text appears while Page 1 is still generating,
                # do NOT exit early. That was the source of intermittent Page 2
                # prewarm failures: the watcher returned while pages_generating=[1],
                # then Page 1 finished with no owner left to start Page 2.
                # Stay alive until either a missing-page worker has been started or
                # all available pages are ready. page-status remains passive.
                refreshed_ready = set(job.get("pages_ready", [])) | set(self._list_ready_pages(user_id, story_id, voice, language_code))
                refreshed_generating = set(job.get("pages_generating", []))
                refreshed_missing = [i for i in range(1, pages_count + 1) if i not in refreshed_ready]
                if refreshed_missing and refreshed_generating:
                    continue
                # One final pass above starts any available missing pages. After the
                # full text exists, the worker will continue through all remaining
                # pages, so the watcher can exit safely.
                return

    def request_narration(
        self,
        user_id: str,
        request: NarrationRequest,
        *,
        client_ip: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> NarrationResponse:
        story = self._get_story_for_user(request.storyId, user_id)
        profile, subscription = self._get_subscription(user_id)

        language_code = self.resolve_language(story, request.narrationLanguageCode)
        requested_voice = self.resolve_voice(request.voicePreference, language_code)
        language_code = self._enforce_standard_voice_language(requested_voice, language_code)
        cache_voice = requested_voice if requested_voice != "parent_voice" else "parent_voice"
        available_pages = len(story.get("pages") or [])
        total_pages = self._expected_total_pages(story)
        # Clamp requested playback to available text, but keep expected total pages
        # for progress/prewarm so early totalPages: 1 is never treated as final.
        requested_start_page = max(1, min(int(request.startPage or 1), max(available_pages, 1))) if total_pages > 0 else 1
        job_id = self._cache_key(
            user_id,
            story["id"],
            cache_voice,
            language_code,
            story.get("child_name_pronunciation"),
        )

        # PRODUCT RULE: generation can consume allowance/credits; cached playback never should.
        # So cached pages and existing jobs are handled before entitlement/credit checks.
        ready_pages = self._list_ready_pages(user_id, story["id"], cache_voice, language_code)
        continuing_cached_generation = bool(ready_pages)
        parent_voice_id = profile.get("parent_voice_id") if requested_voice == "parent_voice" else None

        # Only trust cache when the full page set exists for this narrator/language.
        # This prevents mixed-language playback caused by partial stale caches.
        if ready_pages and len(ready_pages) == total_pages:
            start_path = self._storage_path(user_id, story["id"], cache_voice, language_code, requested_start_page)
            start_url = self._signed_url(start_path)
            existing_job = _chunked_jobs.get(job_id)
            return NarrationResponse(
                status="all_ready",
                audioUrl=start_url,
                pageAudioUrl=start_url,
                currentPage=requested_start_page,
                totalPages=total_pages,
                pagesReady=ready_pages,
                message=f"Page {requested_start_page} ready. {len(ready_pages)}/{total_pages} pages complete.",
                voice_mode=(
                    existing_job.get("voice_mode")
                    if existing_job
                    else ("parent" if requested_voice == "parent_voice" else "standard")
                ),
                jobId=job_id,
            )

        existing = _chunked_jobs.get(job_id)
        if existing and existing.get("status") in {"generating", "page_ready", "all_ready"}:
            # Lean story generation can create page 1 first, then expand the same story
            # to 7 pages later. If narration was prewarmed while only page 1 existed,
            # the existing job may say all_ready for total_pages=1. Keep the job, but
            # expand its total page count and restart one worker for any missing pages.
            existing_total_pages = int(existing.get("total_pages") or 0)
            if total_pages > existing_total_pages:
                print(
                    f"[NARRATION] Expanding existing chunked job story_id={story['id']} "
                    f"from total_pages={existing_total_pages} to total_pages={total_pages}"
                )
                existing["total_pages"] = total_pages
                existing["status"] = "page_ready"

            current_ready_pages = sorted(set(self._list_ready_pages(user_id, story["id"], cache_voice, language_code)))
            if current_ready_pages:
                existing["pages_ready"] = sorted(set([*existing.get("pages_ready", []), *current_ready_pages]))

            existing_ready = set(existing.get("pages_ready", []))
            existing_generating = set(existing.get("pages_generating", []))
            missing_pages = [i for i in range(1, total_pages + 1) if i not in existing_ready]
            worker_should_start = bool(missing_pages) and not existing_generating and not existing.get("last_error")

            if requested_start_page not in existing_ready and requested_start_page not in existing_generating:
                existing["priority_page"] = requested_start_page

            if worker_should_start:
                priority_page = requested_start_page if requested_start_page in missing_pages else missing_pages[0]
                existing["pages_generating"] = [priority_page]
                existing["status"] = "generating"
                print(
                    f"[NARRATION] Restarting chunked worker for missing pages story_id={story['id']} "
                    f"priority_page={priority_page} missing_pages={missing_pages}"
                )
                self._spawn_chunked_worker(
                    background_tasks=background_tasks,
                    job_id=job_id,
                    user_id=user_id,
                    story=story,
                    voice=cache_voice,
                    language_code=language_code,
                    parent_voice_id=parent_voice_id,
                    start_page=priority_page,
                )

            start_storage = existing.get("page_paths", {}).get(requested_start_page)
            start_url = self._signed_url(start_storage) if start_storage else None
            page_is_ready = requested_start_page in existing.get("pages_ready", [])

            # No separate expansion watcher here. The deterministic chunk worker
            # owns page-2+ prewarm as story text expands.

            return NarrationResponse(
                status="page_ready" if page_is_ready else "generating",
                audioUrl=start_url,
                pageAudioUrl=start_url,
                currentPage=requested_start_page,
                totalPages=existing["total_pages"],
                pagesReady=existing.get("pages_ready", []),
                message=(
                    f"Page {requested_start_page} is being generated..."
                    if not page_is_ready
                    else f"Page {requested_start_page} ready. {len(existing.get('pages_ready', []))}/{existing['total_pages']} pages complete."
                ),
                voice_mode=existing.get("voice_mode"),
                jobId=job_id,
            )


        if continuing_cached_generation:
            print(
                f"[NARRATION] Continuing cached narration without new entitlement charge "
                f"story_id={story['id']} voice={cache_voice} language={language_code} ready_pages={ready_pages}"
            )
        else:
            narration_access = self.subscription_service.feature_allowed(subscription, "narration")
            if not narration_access["allowed"]:
                raise HTTPException(status_code=403, detail=narration_access)

            # Parent Voice is an add-on/credit feature, not a standard premium narrator.
            # Do not run it through the generic narrator entitlement gate here.
            # The dedicated Parent Voice block below owns setup, intro/credit,
            # language-lock, security, and charging rules.
            if requested_voice != "parent_voice":
                voice_access = self.subscription_service.feature_allowed(subscription, "narrator", requested_voice)
                if not voice_access["allowed"]:
                    detail = {
                        "error": "premium_narrator",
                        "message": "This narrator is part of PillowTales Premium.",
                        "upgrade_required": True,
                    }
                    raise HTTPException(status_code=403, detail=detail)

        if requested_voice == "parent_voice":
            parent_status = profile.get("parent_voice_status", "none")
            if not parent_voice_id or parent_status != "ready":
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "parent_voice_not_ready",
                        "message": "Parent Voice is not set up yet. Please record your voice first.",
                        "setup_required": True,
                    },
                )

            existing_parent_voice_languages = self._list_existing_parent_voice_languages(user_id, story["id"])
            if existing_parent_voice_languages and language_code not in existing_parent_voice_languages:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "parent_voice_language_locked",
                        "message": "Parent Voice is locked to the original language for this story. Please purchase a new Parent Voice narration to switch language.",
                        "locked_languages": existing_parent_voice_languages,
                    },
                )

            if not continuing_cached_generation:
                self._enforce_parent_voice_security(user_id=user_id, client_ip=client_ip)
                parent_voice_access = self.subscription_service.feature_allowed(subscription, "parent_voice")
                if not parent_voice_access["allowed"]:
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "insufficient_parent_voice_credits",
                            "message": "Parent Voice requires a credit or your first free story.",
                            "upgrade_required": True,
                            "credits": subscription.parent_voice_credits,
                            "intro_offer_available": subscription.parent_voice_intro_available,
                        },
                    )

        credit_charged = False
        intro_charged = False
        if requested_voice == "parent_voice" and not subscription.parent_voice_bypass and not continuing_cached_generation:
            wallet = self.user_repo.get_parent_voice_wallet(user_id)
            credits = int(wallet.get("credits", 0))
            intro_used = bool(wallet.get("intro_used", False))

            if credits > 0:
                self.user_repo.save_parent_voice_wallet(
                    user_id,
                    credits=credits - 1,
                    intro_used=intro_used,
                )
                credit_charged = True
                print(f"[NARRATION] Parent Voice credit charged user_id={user_id} ip={self._normalize_client_ip(client_ip)}")
            elif not intro_used:
                self.user_repo.save_parent_voice_wallet(
                    user_id,
                    credits=0,
                    intro_used=True,
                )
                credit_charged = True
                intro_charged = True
                print(f"[NARRATION] Parent Voice intro consumed user_id={user_id} ip={self._normalize_client_ip(client_ip)}")
            else:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "insufficient_parent_voice_credits",
                        "message": "No Parent Voice credits remaining.",
                        "upgrade_required": True,
                    },
                )

        initial_ready_pages = sorted(set(ready_pages))
        initial_page_paths = {
            idx: self._storage_path(user_id, story["id"], cache_voice, language_code, idx)
            for idx in initial_ready_pages
        }
        missing_pages = [i for i in range(1, total_pages + 1) if i not in set(initial_ready_pages)]
        priority_page = requested_start_page if requested_start_page in missing_pages else (missing_pages[0] if missing_pages else requested_start_page)
        initial_generating_pages = [priority_page] if missing_pages else []
        _chunked_jobs[job_id] = {
            "story_id": story["id"],
            "user_id": user_id,
            "voice": cache_voice,
            "language_code": language_code,
            "status": "generating",
            "pages_ready": initial_ready_pages,
            "pages_generating": initial_generating_pages,
            "pages_failed": [],
            "page_paths": initial_page_paths,
            "total_pages": total_pages,
            "voice_mode": "parent" if requested_voice == "parent_voice" else "standard",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_error": None,
            "credit_charged": credit_charged,
            "credit_refunded": False,
            "intro_charged": intro_charged,
            "priority_page": priority_page,
            "deterministic_worker": True,
            "expansion_watcher_started": False,
        }

        self._spawn_chunked_worker(
            background_tasks=background_tasks,
            job_id=job_id,
            user_id=user_id,
            story=story,
            voice=cache_voice,
            language_code=language_code,
            parent_voice_id=parent_voice_id,
            start_page=priority_page,
        )
        # No separate expansion watcher here. The deterministic chunk worker
        # stays alive briefly and owns page-2+ prewarm as story text expands.

        return NarrationResponse(
            status="generating",
            message=f"Generating page {requested_start_page} narration... (~10-15 seconds)",
            currentPage=requested_start_page,
            totalPages=total_pages,
            pagesReady=initial_ready_pages,
            voice_mode="parent" if requested_voice == "parent_voice" else "standard",
            jobId=job_id,
        )

    def get_page_status(self, user_id: str, story_id: str, narrator: Optional[str], lang: Optional[str]) -> PageStatusResponse:
        story = self._get_story_for_user(story_id, user_id)
        language_code = self.resolve_language(story, lang)
        voice = self.resolve_voice(narrator, language_code)
        language_code = self._enforce_standard_voice_language(voice, language_code)
        cache_voice = voice if voice != "parent_voice" else "parent_voice"
        job_id = self._cache_key(
            user_id,
            story_id,
            cache_voice,
            language_code,
            story.get("child_name_pronunciation"),
        )
        total_pages = self._expected_total_pages(story)
        ready_pages = self._list_ready_pages(user_id, story_id, cache_voice, language_code)
        job = _chunked_jobs.get(job_id)

        # Lean Chunking / Page 2 prewarm safety:
        # Early narration can legitimately be "1/1 ready" while the story text is
        # still expanding in the background. That must never be treated as final.
        generation_status = str(story.get("generation_status") or "complete").strip().lower()
        text_generation_complete = generation_status in {"complete", "completed", "failed"}

        # CRITICAL ARCHITECTURE RULE:
        # page-status is a passive observer only.
        # Do not start, restart, prewarm, or expand narration workers from here.
        # Generation ownership must remain inside the request-narration endpoint.

        if job and job.get("last_error"):
            print(f"[NARRATION] Job {job_id} last_error: {job['last_error']}")

        generating = job.get("pages_generating", []) if job else []
        failed = job.get("pages_failed", []) if job else []
        all_ready = (
            text_generation_complete
            and total_pages > 0
            and len(ready_pages) >= total_pages
        )

        if failed:
            generating = []

        return PageStatusResponse(
            storyId=story_id,
            totalPages=total_pages,
            pagesReady=ready_pages,
            pagesGenerating=[] if all_ready else generating,
            pagesFailed=failed,
            allReady=all_ready,
            voice_mode=(job.get("voice_mode") if job else ("parent" if cache_voice == "parent_voice" else "standard")),
        )

    def get_page_audio_url(self, user_id: str, story_id: str, page: int, narrator: Optional[str], lang: Optional[str]) -> dict:
        if page < 1:
            raise HTTPException(status_code=400, detail="Page must be 1 or greater")

        story = self._get_story_for_user(story_id, user_id)
        language_code = self.resolve_language(story, lang)
        voice = self.resolve_voice(narrator, language_code)
        language_code = self._enforce_standard_voice_language(voice, language_code)
        cache_voice = voice if voice != "parent_voice" else "parent_voice"
        storage_path = self._storage_path(user_id, story_id, cache_voice, language_code, page)
        signed = self._signed_url(storage_path)

        if not signed:
            raise HTTPException(status_code=404, detail=f"Audio for page {page} not found")

        return {"page": page, "audioUrl": signed, "expiresIn": 3600}
