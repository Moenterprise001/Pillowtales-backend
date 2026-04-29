from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import HTTPException

from app.domain.constants import SUPPORTED_LANGUAGES, VOICE_PRESETS
from app.models.narration import NarrationRequest, NarrationResponse, PageStatusResponse
from app.models.subscription import SubscriptionResponse
from app.repositories.story_repository import StoryRepository
from app.repositories.user_repository import UserRepository
from app.services.subscription_service import SubscriptionService
from app.services.text_cleaner import clean_text_for_tts, apply_pronunciation

# In-memory job state for active chunked narration generation.
# Good enough for a single Render instance launch setup.
_chunked_jobs: dict[str, dict] = {}

# In-memory abuse/rate limiting state.
# Also single-instance only, but enough for launch while on one Render instance.
_parent_voice_ip_log: dict[str, list[float]] = {}
_parent_voice_user_log: dict[str, list[float]] = {}

def prepare_narration_text(text: str) -> str:
    """Prepare story text for natural TTS pacing across all standard voices.

    Keep this provider-neutral: OpenAI TTS does not support SSML break tags,
    so we use punctuation and spacing that both OpenAI and ElevenLabs handle
    naturally.
    """
    if not text:
        return text

    text = text.strip()

    # Normalise existing whitespace first so repeated processing is safe.
    text = re.sub(r"\s+", " ", text)

    # Sentence-level pauses for all languages.
    text = re.sub(r"([.!?])\s+", r"\1  ", text)

    # Softer clause pauses.
    text = re.sub(r"([,;:])\s+", r"\1 ", text)

    # Dashes often signal a small dramatic beat in bedtime stories.
    text = text.replace(" — ", "... ").replace("--", "...")

    # Keep spacing tidy while preserving intentional double spaces after sentences.
    text = re.sub(r" {3,}", "  ", text)
    return text.strip()


def adapt_spanish_castellano(text: str) -> str:
    """Nudge Spanish narration toward European Spanish wording.

    OpenAI TTS does not expose a reliable es-ES accent flag, so this uses
    conservative vocabulary/phrase choices that bias the reading toward Spain
    without changing story meaning.
    """
    if not text:
        return text

    replacements = [
        (r"\bustedes están\b", "vosotros estáis"),
        (r"\bustedes son\b", "vosotros sois"),
        (r"\bustedes tienen\b", "vosotros tenéis"),
        (r"\bustedes pueden\b", "vosotros podéis"),
        (r"\bustedes\b", "vosotros"),
        (r"\bcomputadora\b", "ordenador"),
        (r"\bcarro\b", "coche"),
        (r"\bauto\b", "coche"),
        (r"\bplaticar\b", "charlar"),
        (r"\blindo\b", "bonito"),
        (r"\blinda\b", "bonita"),
    ]

    adjusted = text
    for pattern, replacement in replacements:
        adjusted = re.sub(pattern, replacement, adjusted, flags=re.IGNORECASE)

    return adjusted


class NarrationService:
    def __init__(self, story_repo: StoryRepository, user_repo: UserRepository, subscription_service: SubscriptionService):
        self.story_repo = story_repo
        self.user_repo = user_repo
        self.subscription_service = subscription_service

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
        language_code = (language_code or "en").strip().lower()[:2]
        if language_code not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail="Unsupported narration language")
        return language_code

    def default_voice_for_language(self, language_code: str) -> str:
        return {
            "en": "wise_owl",
            "es": "night_owl_spanish",
            "de": "night_owl_german",
            "fr": "night_owl_french",
            "it": "night_owl_italian",
        }.get(language_code, "wise_owl")

    def resolve_voice(self, requested_voice: Optional[str], language_code: str) -> str:
        # Product rule: default narrator must remain Wise Owl / standard narrator family,
        # never Parent Voice unless explicitly selected.
        default_voice = self.default_voice_for_language(language_code)

        if not requested_voice:
            return default_voice

        if requested_voice not in VOICE_PRESETS:
            raise HTTPException(status_code=400, detail="Unsupported narrator")

        # Parent Voice is allowed explicitly and handles multilingual separately.
        if requested_voice == "parent_voice":
            return requested_voice

        preset = VOICE_PRESETS.get(requested_voice, {}) or {}
        preset_lang = (
            preset.get("language_code")
            or preset.get("language")
            or "all"
        )
        preset_lang = str(preset_lang).strip().lower()[:2] if preset_lang != "all" else "all"

        # Only allow exact-language or universal narrators.
        if preset_lang == "all" or preset_lang == language_code:
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
        return f"{user_id}:{story_id}:{voice}:{language_code}:{safe_pronunciation}:v5"

    def _storage_prefix(self, user_id: str, story_id: str, voice: str, language_code: str) -> str:
        return f"{user_id}/{story_id}/chunked/{voice}_{language_code}_v5"

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
            match = re.match(r"parent_voice_([a-z]{2})$", name.strip())
            if match:
                languages.add(match.group(1))
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

    async def _generate_openai_tts(self, text: str, voice: str) -> bytes:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")

        provider_voice = VOICE_PRESETS.get(voice, {}).get("voice_id") or "shimmer"

        async with httpx.AsyncClient(timeout=120.0) as client:
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
                    "format": "mp3",
                },
            )

        if response.status_code != 200:
            raise RuntimeError(f"OpenAI TTS failed: {response.status_code} {response.text[:300]}")

        return response.content

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
                    "language_code": language_code,
                    "voice_settings": {
                        "stability": 0.75,
                        "similarity_boost": 0.5,
                        "style": 0.0,
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


    async def _translate_text(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        if not text:
            return text

        target = (target_lang or "en").lower()
        source = (source_lang or "").lower()

        if source and source == target:
            print(f"[TRANSLATE] Skipping translation source_lang={source} target_lang={target}")
            return text

        if not source and target == "en":
            return text

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"[TRANSLATE] OPENAI_API_KEY missing for source_lang={source or 'unknown'} target_lang={target}")
            raise RuntimeError("OPENAI_API_KEY not configured for translation")

        try:
            system_prompt = (
                f"Translate the following children's story text from {source or 'the original language'} into {target}. "
                "Keep it natural, child-friendly, and preserve names, tone, and meaning. "
                "Return only the translated text."
            )
            async with httpx.AsyncClient(timeout=30.0) as client:
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
                        "temperature": 0.2,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                translated = data["choices"][0]["message"]["content"].strip()
                print(
                    f"[TRANSLATE] Translation success source_lang={source or 'unknown'} "
                    f"target_lang={target} input_preview={text[:120]!r} output_preview={translated[:120]!r}"
                )
                return translated
        except Exception as e:
            print(
                f"[TRANSLATE] FAILED source_lang={source or 'unknown'} "
                f"target_lang={target}: {repr(e)} input_preview={text[:120]!r}"
            )
            raise RuntimeError(f"Translation failed from {source or 'unknown'} to {target}")


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
    ) -> tuple[str, str]:
        page_text = self._clean_page_text(page_text)
        translated = await self._translate_text(page_text, language_code, story_language_code)
        print(f"[NARRATION] Generating page {page} with voice={voice} language={language_code}")
        print(f"[NARRATION] Original page text preview={page_text[:160]!r}")
        print(f"[NARRATION] Translated text preview for {language_code}={translated[:160]!r}")
        tts_text = clean_text_for_tts(translated)
        tts_text = apply_pronunciation(tts_text, child_name, child_name_pronunciation)
        if language_code == "es":
            tts_text = adapt_spanish_castellano(tts_text)
        tts_text = prepare_narration_text(tts_text)

        if not tts_text:
            raise RuntimeError("Page has no text")

        used_mode = voice_mode
        if voice_mode == "parent" and parent_voice_id:
            try:
                audio = await self._generate_elevenlabs_tts(tts_text, parent_voice_id, language_code)
            except Exception:
                # Bulletproof fallback: keep the whole job alive with standard narration.
                used_mode = "fallback_tts"
                fallback_voice = self.default_voice_for_language(language_code)
                audio = await self._generate_openai_tts(tts_text, fallback_voice)
        else:
            audio = await self._generate_openai_tts(tts_text, voice)

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
        pages = story.get("pages") or []
        job = _chunked_jobs[job_id]

        if not pages:
            job["pages_failed"] = [1]
            job["pages_generating"] = []
            job["status"] = "failed"
            job["last_error"] = "Story has no pages"
            self._refund_parent_voice_credit_once(user_id, job)
            return

        initial_voice_mode = "parent" if voice == "parent_voice" and parent_voice_id else "standard"
        job["voice_mode"] = initial_voice_mode

        safe_start_page = max(1, min(int(start_page or 1), len(pages)))
        page_order = [safe_start_page] + [i for i in range(1, len(pages) + 1) if i != safe_start_page]
        job["priority_page"] = safe_start_page

        for idx in page_order:
            storage_path = self._storage_path(user_id, story["id"], voice, language_code, idx)
            if self._signed_url(storage_path):
                if idx not in job["pages_ready"]:
                    job["pages_ready"].append(idx)
                job["page_paths"][idx] = storage_path
                job["pages_ready"] = sorted(job["pages_ready"])
                if idx in job["pages_failed"]:
                    job["pages_failed"].remove(idx)
                job["last_error"] = None
                job["status"] = "page_ready" if idx < len(pages) else "all_ready"
                await asyncio.sleep(0.05)
                continue
            page_text = pages[idx - 1]
            job["pages_generating"] = [idx]
            try:
                storage_path, actual_mode = await self._generate_page_audio(
                    user_id=user_id,
                    story_id=story["id"],
                    page=idx,
                    page_text=page_text,
                    voice=voice,
                    language_code=language_code,
                    voice_mode=job["voice_mode"],
                    parent_voice_id=parent_voice_id,
                    child_name=story.get("child_name"),
                    child_name_pronunciation=story.get("child_name_pronunciation"),
                    story_language_code=self.resolve_language(story, story.get("language") or story.get("language_code") or story.get("story_language") or story.get("preferred_language")),
                )

                job["voice_mode"] = actual_mode
                if idx not in job["pages_ready"]:
                    job["pages_ready"].append(idx)
                job["page_paths"][idx] = storage_path
                job["pages_ready"] = sorted(job["pages_ready"])
                if idx in job["pages_failed"]:
                    job["pages_failed"].remove(idx)
                job["last_error"] = None
                job["status"] = "page_ready" if idx < len(pages) else "all_ready"

                if idx == 1:
                    try:
                        first_url = self._signed_url(storage_path)
                        update_payload = {
                            "audio_status": "ready",
                            "audio_url": first_url,
                            "audio_created_at": datetime.now(timezone.utc).isoformat(),
                            "audio_language_code": language_code,
                            "audio_voice_id": voice,
                        }
                        self.story_repo.update(story["id"], user_id, update_payload)
                    except Exception:
                        pass

            except Exception as e:
                print(f"[NARRATION] Page {idx} generation failed for story {story['id']}: {repr(e)}")
                if idx not in job["pages_failed"]:
                    job["pages_failed"].append(idx)
                job["pages_generating"] = []
                job["status"] = "failed"
                job["last_error"] = str(e)
                self._refund_parent_voice_credit_once(user_id, job)
                return

            await asyncio.sleep(0.2)

        job["pages_generating"] = []
        job["status"] = "all_ready"

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

    def request_narration(
        self,
        user_id: str,
        request: NarrationRequest,
        *,
        client_ip: Optional[str] = None,
    ) -> NarrationResponse:
        story = self._get_story_for_user(request.storyId, user_id)
        profile, subscription = self._get_subscription(user_id)

        language_code = self.resolve_language(story, request.narrationLanguageCode)
        requested_voice = self.resolve_voice(request.voicePreference, language_code)
        cache_voice = requested_voice if requested_voice != "parent_voice" else "parent_voice"
        total_pages = len(story.get("pages") or [])
        requested_start_page = max(1, min(int(request.startPage or 1), total_pages)) if total_pages > 0 else 1
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
                asyncio.create_task(
                    self._process_chunked_job(
                        job_id=job_id,
                        user_id=user_id,
                        story=story,
                        voice=cache_voice,
                        language_code=language_code,
                        parent_voice_id=parent_voice_id,
                        start_page=priority_page,
                    )
                )

            start_storage = existing.get("page_paths", {}).get(requested_start_page)
            start_url = self._signed_url(start_storage) if start_storage else None
            page_is_ready = requested_start_page in existing.get("pages_ready", [])

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
        }

        asyncio.create_task(
            self._process_chunked_job(
                job_id=job_id,
                user_id=user_id,
                story=story,
                voice=cache_voice,
                language_code=language_code,
                parent_voice_id=parent_voice_id,
                start_page=priority_page,
            )
        )

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
        cache_voice = voice if voice != "parent_voice" else "parent_voice"
        job_id = self._cache_key(
            user_id,
            story_id,
            cache_voice,
            language_code,
            story.get("child_name_pronunciation"),
        )
        total_pages = len(story.get("pages") or [])
        ready_pages = self._list_ready_pages(user_id, story_id, cache_voice, language_code)
        job = _chunked_jobs.get(job_id)

        if job and job.get("last_error"):
            print(f"[NARRATION] Job {job_id} last_error: {job['last_error']}")

        generating = job.get("pages_generating", []) if job else []
        failed = job.get("pages_failed", []) if job else []
        all_ready = total_pages > 0 and len(ready_pages) >= total_pages

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
        cache_voice = voice if voice != "parent_voice" else "parent_voice"
        storage_path = self._storage_path(user_id, story_id, cache_voice, language_code, page)
        signed = self._signed_url(storage_path)

        if not signed:
            raise HTTPException(status_code=404, detail=f"Audio for page {page} not found")

        return {"page": page, "audioUrl": signed, "expiresIn": 3600}
