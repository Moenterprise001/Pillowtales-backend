from __future__ import annotations

from typing import Optional

from fastapi import HTTPException

from app.core.config import settings
from app.domain.constants import SUPPORTED_LANGUAGES, VOICE_PRESETS
from app.models.narration import NarrationRequest, NarrationResponse
from app.models.subscription import SubscriptionResponse
from app.repositories.story_repository import StoryRepository
from app.repositories.user_repository import UserRepository
from app.services.subscription_service import SubscriptionService


class NarrationService:
    def __init__(self, story_repo: StoryRepository, user_repo: UserRepository, subscription_service: SubscriptionService):
        self.story_repo = story_repo
        self.user_repo = user_repo
        self.subscription_service = subscription_service

    def resolve_language(self, story: dict, requested_language: Optional[str]) -> str:
        language_code = requested_language or story.get('narration_language_code') or story.get('story_language_code') or 'en'
        if language_code not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail='Unsupported narration language')
        return language_code

    def resolve_voice(self, requested_voice: Optional[str], language_code: str) -> str:
        if requested_voice:
            if requested_voice not in VOICE_PRESETS:
                raise HTTPException(status_code=400, detail='Unsupported narrator')
            return requested_voice
        return {'en': 'wise_owl', 'es': 'night_owl_spanish', 'de': 'night_owl_german', 'fr': 'night_owl_french', 'it': 'night_owl_italian'}.get(language_code, 'wise_owl')

    def enqueue_job(self, story: dict, language_code: str, voice: str) -> dict:
        return {
            'queue_backend': settings.narration_backend,
            'queue_name': settings.narration_queue_name,
            'story_id': story['id'],
            'language_code': language_code,
            'voice': voice,
            'status': 'queued',
        }

    def request_narration(self, user_id: str, request: NarrationRequest) -> NarrationResponse:
        story = self.story_repo.get(request.storyId, user_id)
        if not story:
            raise HTTPException(status_code=404, detail='Story not found')
        profile = self.user_repo.get_profile(user_id) or {}
        subscription: SubscriptionResponse = self.subscription_service.get_subscription(user_id, profile.get('email'))
        narration_access = self.subscription_service.feature_allowed(subscription, 'narration')
        if not narration_access['allowed']:
            raise HTTPException(status_code=403, detail=narration_access)
        language_code = self.resolve_language(story, request.narrationLanguageCode)
        voice = self.resolve_voice(request.voicePreference, language_code)
        voice_access = self.subscription_service.feature_allowed(subscription, 'narrator', voice)
        if not voice_access['allowed']:
            raise HTTPException(status_code=403, detail=voice_access)
        job = self.enqueue_job(story, language_code, voice)
        return NarrationResponse(
            status='pending',
            audioUrl=story.get('audio_url'),
            message=f"Narration queued using {job['voice']} for {job['language_code']}. Connect this service to Celery/RQ/Cloud Tasks next for real async generation.",
        )
