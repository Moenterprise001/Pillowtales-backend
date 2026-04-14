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
        language_code = requested_language or story.get('narration_language_code') or story.get('story_language_code') or story.get('language') or 'en'
        language_code = (language_code or 'en').strip().lower()[:2]
        if language_code not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail='Unsupported narration language')
        return language_code

    def default_voice_for_language(self, language_code: str) -> str:
        return {
            'en': 'wise_owl',
            'es': 'night_owl_spanish',
            'de': 'night_owl_german',
            'fr': 'night_owl_french',
            'it': 'night_owl_italian',
        }.get(language_code, 'wise_owl')

    def resolve_voice(self, requested_voice: Optional[str], language_code: str) -> str:
        # Product rule: default narrator must remain Wise Owl / standard narrator family,
        # never Parent Voice unless explicitly selected.
        if requested_voice:
            if requested_voice not in VOICE_PRESETS:
                raise HTTPException(status_code=400, detail='Unsupported narrator')
            return requested_voice
        return self.default_voice_for_language(language_code)

    def _cache_key(self, user_id: str, story_id: str, voice: str, language_code: str, pronunciation: Optional[str] = None) -> str:
        safe_pronunciation = (pronunciation or '').strip().lower()
        return f"{user_id}:{story_id}:{voice}:{language_code}:{safe_pronunciation}"

    def _storage_prefix(self, user_id: str, story_id: str, voice: str, language_code: str) -> str:
        return f"{user_id}/{story_id}/chunked/{voice}_{language_code}"

    def _storage_path(self, user_id: str, story_id: str, voice: str, language_code: str, page: int) -> str:
        return f"{self._storage_prefix(user_id, story_id, voice, language_code)}/page_{page}.mp3"

    def _signed_url(self, storage_path: str, expires_in: int = 3600) -> Optional[str]:
        try:
            result = self.story_repo.client.storage.from_('story-audio').create_signed_url(storage_path, expires_in)
            return result.get('signedURL') or result.get('signedUrl')
        except Exception:
            return None

    def _list_ready_pages(self, user_id: str, story_id: str, voice: str, language_code: str) -> list[int]:
        prefix = self._storage_prefix(user_id, story_id, voice, language_code)
        try:
            items = self.story_repo.client.storage.from_('story-audio').list(prefix)
        except Exception:
            return []
        ready: list[int] = []
        for item in items or []:
            name = item.get('name') or ''
            m = re.match(r'page_(\d+)\.mp3$', name)
            if m:
                ready.append(int(m.group(1)))
        return sorted(set(ready))

    def _clean_page_text(self, page_text: str) -> str:
        text = page_text or ''
        for marker in ('[NARRATION_START]', '[NARRATION_END]', '[whisper]', '[softly]', '[chuckle]', '[pause]', '[gently]'):
            text = text.replace(marker, '')
        return text.strip()

    async def _generate_openai_tts(self, text: str, voice: str) -> bytes:
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY not configured')
        provider_voice = VOICE_PRESETS.get(voice, {}).get('voice_id') or 'shimmer'
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                'https://api.openai.com/v1/audio/speech',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': 'gpt-4o-mini-tts',
                    'voice': provider_voice,
                    'input': text,
                    'format': 'mp3',
                },
            )
        if response.status_code != 200:
            raise RuntimeError(f'OpenAI TTS failed: {response.status_code} {response.text[:300]}')
        return response.content

    async def _generate_elevenlabs_tts(self, text: str, voice_id: str, language_code: str) -> bytes:
        api_key = os.getenv('ELEVENLABS_API_KEY', '')
        if not api_key:
            raise RuntimeError('ELEVENLABS_API_KEY not configured')
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
                headers={
                    'xi-api-key': api_key,
                    'Accept': 'audio/mpeg',
                    'Content-Type': 'application/json',
                },
                json={
                    'text': text,
                    'model_id': 'eleven_multilingual_v2',
                    'language_code': language_code,
                    'voice_settings': {
                        'stability': 0.75,
                        'similarity_boost': 0.5,
                        'style': 0.0,
                        'use_speaker_boost': True,
                    },
                    'output_format': 'mp3_44100_128',
                },
            )
        if response.status_code != 200:
            raise RuntimeError(f'ElevenLabs TTS failed: {response.status_code} {response.text[:300]}')
        return response.content

    async def _upload_audio(self, storage_path: str, audio_bytes: bytes) -> None:
        self.story_repo.client.storage.from_('story-audio').upload(
            storage_path,
            audio_bytes,
            {'content-type': 'audio/mpeg', 'upsert': 'true'},
        )

    async def _generate_page_audio(self, *, user_id: str, story_id: str, page: int, page_text: str, voice: str, language_code: str, voice_mode: str, parent_voice_id: Optional[str], child_name: Optional[str] = None, child_name_pronunciation: Optional[str] = None) -> tuple[str, str]:
        page_text = self._clean_page_text(page_text)
        tts_text = clean_text_for_tts(page_text)
        tts_text = apply_pronunciation(tts_text, child_name, child_name_pronunciation)

        if not tts_text:
            raise RuntimeError('Page has no text')

        used_mode = voice_mode
        if voice_mode == 'parent' and parent_voice_id:
            try:
                audio = await self._generate_elevenlabs_tts(tts_text, parent_voice_id, language_code)
            except Exception:
                # Bulletproof fallback: keep the whole job alive with standard narration.
                used_mode = 'fallback_tts'
                fallback_voice = self.default_voice_for_language(language_code)
                audio = await self._generate_openai_tts(tts_text, fallback_voice)
        else:
            audio = await self._generate_openai_tts(tts_text, voice)

        storage_path = self._storage_path(user_id, story_id, voice, language_code, page)
        await self._upload_audio(storage_path, audio)
        return storage_path, used_mode

    async def _process_chunked_job(self, *, job_id: str, user_id: str, story: dict, voice: str, language_code: str, parent_voice_id: Optional[str]) -> None:
        pages = story.get('pages') or []
        job = _chunked_jobs[job_id]
        if not pages:
            job['pages_failed'] = [1]
            job['pages_generating'] = []
            job['status'] = 'failed'
            return

        initial_voice_mode = 'parent' if voice == 'parent_voice' and parent_voice_id else 'standard'
        job['voice_mode'] = initial_voice_mode

        for idx, page_text in enumerate(pages, start=1):
            job['pages_generating'] = [p for p in range(idx, len(pages) + 1)]
            try:
                storage_path, actual_mode = await self._generate_page_audio(
                    user_id=user_id,
                    story_id=story['id'],
                    page=idx,
                    page_text=page_text,
                    voice=voice,
                    language_code=language_code,
                    voice_mode=job['voice_mode'],
                    parent_voice_id=parent_voice_id,
                    child_name=story.get('child_name'),
                    child_name_pronunciation=story.get('child_name_pronunciation'),
                )
                job['voice_mode'] = actual_mode
                if idx not in job['pages_ready']:
                    job['pages_ready'].append(idx)
                job['page_paths'][idx] = storage_path
                job['pages_ready'] = sorted(job['pages_ready'])
                if idx in job['pages_failed']:
                    job['pages_failed'].remove(idx)
                job['status'] = 'page_ready' if idx < len(pages) else 'all_ready'

                if idx == 1:
                    try:
                        first_url = self._signed_url(storage_path)
                        update_payload = {
                            'audio_status': 'ready',
                            'audio_url': first_url,
                            'audio_created_at': datetime.now(timezone.utc).isoformat(),
                            'audio_language_code': language_code,
                            'audio_voice_id': voice,
                        }
                        self.story_repo.update(story['id'], user_id, update_payload)
                    except Exception:
                        pass
            except Exception:
                if idx not in job['pages_failed']:
                    job['pages_failed'].append(idx)
                job['pages_generating'] = [p for p in range(idx + 1, len(pages) + 1)]
                job['status'] = 'failed'
                return

            await asyncio.sleep(0.2)

        job['pages_generating'] = []
        job['status'] = 'all_ready'

    def _get_story_for_user(self, story_id: str, user_id: str) -> dict:
        story = self.story_repo.get(story_id, user_id)
        if not story:
            raise HTTPException(status_code=404, detail='Story not found')
        if not isinstance(story.get('pages'), list) or not story.get('pages'):
            raise HTTPException(status_code=400, detail='Story has no pages')
        return story

    def _get_subscription(self, user_id: str) -> tuple[dict, SubscriptionResponse]:
        profile = self.user_repo.get_profile(user_id) or {}
        subscription = self.subscription_service.get_subscription(user_id, profile.get('email'))
        return profile, subscription

    def request_narration(self, user_id: str, request: NarrationRequest) -> NarrationResponse:
        story = self._get_story_for_user(request.storyId, user_id)
        profile, subscription = self._get_subscription(user_id)

        narration_access = self.subscription_service.feature_allowed(subscription, 'narration')
        if not narration_access['allowed']:
            raise HTTPException(status_code=403, detail=narration_access)

        language_code = self.resolve_language(story, request.narrationLanguageCode)
        requested_voice = self.resolve_voice(request.voicePreference, language_code)
        voice_access = self.subscription_service.feature_allowed(subscription, 'narrator', requested_voice)
        if not voice_access['allowed']:
            detail = {
                'error': 'premium_narrator',
                'message': 'This narrator is part of PillowTales Premium.',
                'upgrade_required': True,
            }
            raise HTTPException(status_code=403, detail=detail)

        parent_voice_id = None
        if requested_voice == 'parent_voice':
            parent_voice_access = self.subscription_service.feature_allowed(subscription, 'parent_voice')
            if not parent_voice_access['allowed']:
                raise HTTPException(
                    status_code=403,
                    detail={
                        'error': 'insufficient_parent_voice_credits',
                        'message': 'Parent Voice requires a credit or Premium access.',
                        'upgrade_required': True,
                        'credits': subscription.parent_voice_credits,
                        'intro_offer_available': subscription.parent_voice_intro_available,
                    },
                )
            parent_voice_id = profile.get('parent_voice_id')
            parent_status = profile.get('parent_voice_status', 'none')
            if not parent_voice_id or parent_status != 'ready':
                raise HTTPException(
                    status_code=400,
                    detail={
                        'error': 'parent_voice_not_ready',
                        'message': 'Parent Voice is not set up yet. Please record your voice first.',
                        'setup_required': True,
                    },
                )

        # Only explicit selection should use Parent Voice.
        cache_voice = requested_voice if requested_voice != 'parent_voice' else 'parent_voice'
        total_pages = len(story.get('pages') or [])
        job_id = self._cache_key(
            user_id,
            story['id'],
            cache_voice,
            language_code,
            story.get('child_name_pronunciation'),
        )

        ready_pages = self._list_ready_pages(user_id, story['id'], cache_voice, language_code)
        if ready_pages:
            page1_path = self._storage_path(user_id, story['id'], cache_voice, language_code, 1)
            page1_url = self._signed_url(page1_path)
            existing_job = _chunked_jobs.get(job_id)
            return NarrationResponse(
                status='all_ready' if len(ready_pages) >= total_pages else 'page_ready',
                audioUrl=page1_url,
                pageAudioUrl=page1_url,
                currentPage=1,
                totalPages=total_pages,
                pagesReady=ready_pages,
                message=f'Page 1 ready. {len(ready_pages)}/{total_pages} pages complete.',
                voice_mode=(existing_job.get('voice_mode') if existing_job else ('parent' if requested_voice == 'parent_voice' else 'standard')),
                jobId=job_id,
            )

        existing = _chunked_jobs.get(job_id)
        if existing and existing.get('status') in {'generating', 'page_ready', 'all_ready'}:
            page1_storage = existing.get('page_paths', {}).get(1)
            page1_url = self._signed_url(page1_storage) if page1_storage else None
            return NarrationResponse(
                status='page_ready' if 1 in existing.get('pages_ready', []) else 'generating',
                audioUrl=page1_url,
                pageAudioUrl=page1_url,
                currentPage=1,
                totalPages=existing['total_pages'],
                pagesReady=existing.get('pages_ready', []),
                message='Page 1 is being generated...' if 1 not in existing.get('pages_ready', []) else f"Page 1 ready. {len(existing.get('pages_ready', []))}/{existing['total_pages']} pages complete.",
                voice_mode=existing.get('voice_mode'),
                jobId=job_id,
            )

        if requested_voice == 'parent_voice' and not subscription.is_premium:
            wallet = self.user_repo.get_parent_voice_wallet(user_id)
            credits = int(wallet.get('credits', 0))
            if credits <= 0:
                raise HTTPException(
                    status_code=403,
                    detail={
                        'error': 'insufficient_parent_voice_credits',
                        'message': 'No Parent Voice credits remaining.',
                        'upgrade_required': True,
                    },
                )
            self.user_repo.save_parent_voice_wallet(
                user_id,
                credits=credits - 1,
                intro_used=bool(wallet.get('intro_used', False)),
            )

        _chunked_jobs[job_id] = {
            'story_id': story['id'],
            'user_id': user_id,
            'voice': cache_voice,
            'language_code': language_code,
            'status': 'generating',
            'pages_ready': [],
            'pages_generating': list(range(1, total_pages + 1)),
            'pages_failed': [],
            'page_paths': {},
            'total_pages': total_pages,
            'voice_mode': 'parent' if requested_voice == 'parent_voice' else 'standard',
            'started_at': datetime.now(timezone.utc).isoformat(),
        }
        asyncio.create_task(
            self._process_chunked_job(
                job_id=job_id,
                user_id=user_id,
                story=story,
                voice=cache_voice,
                language_code=language_code,
                parent_voice_id=parent_voice_id,
            )
        )

        return NarrationResponse(
            status='generating',
            message='Generating Page 1 narration... (~10-15 seconds)',
            currentPage=1,
            totalPages=total_pages,
            pagesReady=[],
            voice_mode='parent' if requested_voice == 'parent_voice' else 'standard',
            jobId=job_id,
        )

    def get_page_status(self, user_id: str, story_id: str, narrator: Optional[str], lang: Optional[str]) -> PageStatusResponse:
        story = self._get_story_for_user(story_id, user_id)
        language_code = self.resolve_language(story, lang)
        voice = self.resolve_voice(narrator, language_code)
        cache_voice = voice if voice != 'parent_voice' else 'parent_voice'
        job_id = self._cache_key(
            user_id,
            story_id,
            cache_voice,
            language_code,
            story.get('child_name_pronunciation'),
        )
        total_pages = len(story.get('pages') or [])
        ready_pages = self._list_ready_pages(user_id, story_id, cache_voice, language_code)
        job = _chunked_jobs.get(job_id)
        generating = job.get('pages_generating', []) if job else []
        failed = job.get('pages_failed', []) if job else []
        all_ready = total_pages > 0 and len(ready_pages) >= total_pages
        return PageStatusResponse(
            storyId=story_id,
            totalPages=total_pages,
            pagesReady=ready_pages,
            pagesGenerating=[] if all_ready else generating,
            pagesFailed=failed,
            allReady=all_ready,
            voice_mode=(job.get('voice_mode') if job else ('parent' if cache_voice == 'parent_voice' else 'standard')),
        )

    def get_page_audio_url(self, user_id: str, story_id: str, page: int, narrator: Optional[str], lang: Optional[str]) -> dict:
        if page < 1:
            raise HTTPException(status_code=400, detail='Page must be 1 or greater')
        story = self._get_story_for_user(story_id, user_id)
        language_code = self.resolve_language(story, lang)
        voice = self.resolve_voice(narrator, language_code)
        cache_voice = voice if voice != 'parent_voice' else 'parent_voice'
        storage_path = self._storage_path(user_id, story_id, cache_voice, language_code, page)
        signed = self._signed_url(storage_path)
        if not signed:
            raise HTTPException(status_code=404, detail=f'Audio for page {page} not found')
        return {'page': page, 'audioUrl': signed, 'expiresIn': 3600}
