from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import google.generativeai as genai
from fastapi import HTTPException

from app.core.config import settings
from app.domain.constants import STORY_COMPANIONS, SUBSCRIPTION_TIERS, SUPPORTED_LANGUAGES
from app.models.story import GenerateStoryRequest
from app.models.subscription import SubscriptionResponse
from app.repositories.story_repository import StoryRepository
from app.utils.story_text import postprocess_story_pages


class StoryService:
    def __init__(self, story_repo: StoryRepository):
        self.story_repo = story_repo
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model) if settings.gemini_api_key else None

    def _select_companion(self, request: GenerateStoryRequest, subscription: SubscriptionResponse) -> Optional[dict]:
        if request.companionId and request.companionId in STORY_COMPANIONS:
            companion = STORY_COMPANIONS[request.companionId].copy()
            companion['id'] = request.companionId
            return companion
        if random.random() > 0.30:
            return None
        tier = SUBSCRIPTION_TIERS['premium' if subscription.is_premium else 'free']
        available = [cid for cid in tier['companions'] if cid in STORY_COMPANIONS]
        if not available:
            return None
        selected = random.choice(available)
        companion = STORY_COMPANIONS[selected].copy()
        companion['id'] = selected
        return companion

    def _build_prompt(self, request: GenerateStoryRequest, companion: Optional[dict]) -> str:
        language_name = SUPPORTED_LANGUAGES.get(request.storyLanguageCode, 'English')
        effective_theme = request.customTheme or request.theme
        companion_line = 'No companion is required.'
        if companion:
            companion_line = f"Include {companion['name']} naturally in the story. They are described as: {companion['description']}. Make them warm, helpful, and bedtime-appropriate."

        family_characters = request.characters or []
        if family_characters:
            character_lines = []
            for character in family_characters[:3]:
                character_lines.append(f"- {character.name} ({character.relationship})")
            characters_block = '\n'.join(character_lines)
            character_instruction = (
                'Include these family members, friends, or pets naturally in the story if possible. '
                'Make sure each named character appears clearly at least once without overwhelming the bedtime tone:\n'
                f"{characters_block}"
            )
        else:
            character_instruction = 'No extra family members or friends are required.'

        return f"""You are a premium children's bedtime storyteller.
Write a calming, original bedtime story in {language_name}.

Rules:
- Child name: {request.childName}
- Age: {request.age}
- Theme: {effective_theme}
- Moral: {request.moral}
- Calm level: {request.calmLevel}
- Keep the tone warm, magical, and safe.
- Use simple narration that sounds natural when read aloud.
- Do not use 'The end'.
- End with a peaceful natural bedtime finish.
- Format for readability with 2-3 sentences per paragraph and visible breaks.
- Create approximately {request.durationMin} minutes of read-aloud content.
- {companion_line}
- {character_instruction}
Return ONLY valid JSON with this schema:
{{"title": "...", "pages": ["...", "..."]}}"""

    async def generate_story(self, request: GenerateStoryRequest, subscription: SubscriptionResponse) -> Dict[str, Any]:
        companion = self._select_companion(request, subscription)
        if not self.model:
            pages = [
                f"Once upon a time, {request.childName} discovered a quiet little path full of wonder.",
                f"The path led to a gentle adventure about {request.customTheme or request.theme}, where kindness mattered most.",
                f"Soon, everything grew peaceful again, and {request.childName} felt calm enough for sleep.",
            ]
            return {'title': f"{request.childName}'s Bedtime Adventure", 'pages': postprocess_story_pages(pages), 'companion': companion}

        response = self.model.generate_content(self._build_prompt(request, companion))
        response_text = getattr(response, 'text', None)
        if not response_text or not isinstance(response_text, str):
            raise HTTPException(status_code=500, detail='Failed to generate story')

        cleaned = response_text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]

        story_data = json.loads(cleaned.strip())
        if not isinstance(story_data, dict) or 'title' not in story_data or 'pages' not in story_data:
            raise HTTPException(status_code=500, detail='Invalid story format returned by AI')
        story_data['pages'] = postprocess_story_pages(story_data.get('pages', []))
        story_data['companion'] = companion
        return story_data

    async def extract_metadata(self, title: str, full_text: str) -> Dict[str, Any]:
        if not self.model:
            return {'summary': '', 'characters': [], 'setting': ''}
        prompt = (
            'Analyze this bedtime story and return only valid JSON. '
            'Schema: {"summary":"...","characters":[{"name":"...","description":"...","role":"..."}],"setting":"..."}\n'
            f'Title: {title}\nStory:\n{full_text}'
        )
        try:
            response = self.model.generate_content(prompt)
            text = getattr(response, 'text', '')
            start = text.find('{')
            end = text.rfind('}')
            if start == -1 or end == -1:
                return {'summary': '', 'characters': [], 'setting': ''}
            return json.loads(text[start:end + 1])
        except Exception:
            return {'summary': '', 'characters': [], 'setting': ''}

    def validate_story_limits(self, user_id: str, subscription: SubscriptionResponse) -> None:
        tier = SUBSCRIPTION_TIERS['premium' if subscription.is_premium else 'free']
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        stories_this_week = self.story_repo.count_since(user_id, week_ago)
        if tier['weekly_story_limit'] is not None and stories_this_week >= tier['weekly_story_limit']:
            raise HTTPException(status_code=403, detail={'error': 'story_limit_reached', 'message': "You've created 2 free stories this week. Upgrade to create unlimited bedtime stories.", 'upgrade_required': True})
        stories_saved = self.story_repo.count_all(user_id)
        if tier['max_saved_stories'] is not None and stories_saved >= tier['max_saved_stories']:
            raise HTTPException(status_code=403, detail={'error': 'storage_limit', 'message': "You've reached the maximum number of saved stories.", 'upgrade_required': True})
