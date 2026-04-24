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
            companion_line = (
                f"Include {companion['name']} naturally in the story. "
                f"They are described as: {companion['description']}. "
                "Make them warm, helpful, and bedtime-appropriate."
            )

        family_characters = request.characters or []
        if family_characters:
            character_lines = []
            for character in family_characters[:3]:
                character_lines.append(f"- {character.name} ({character.relationship})")
            characters_block = '\n'.join(character_lines)
            character_instruction = (
                "Include these family members, friends, or pets naturally in the story if possible. "
                "Make sure each named character appears clearly at least once without overwhelming the bedtime tone:\n"
                f"{characters_block}"
            )
        else:
            character_instruction = "No extra family members or friends are required."

        if request.durationMin >= 11:
            target_pages = "10"
            paragraphs_per_page = "1"
            sentence_range = "2-4"
            target_words = "750-950"
            pacing_note = (
                "This should feel like a full, rich bedtime journey with gentle detail, "
                "quiet discovery, and a soft, satisfying wind-down."
            )
        else:
            target_pages = "8"
            paragraphs_per_page = "1"
            sentence_range = "2-4"
            target_words = "550-750"
            pacing_note = (
                "This should feel full and immersive rather than brief. "
                "Take time with the middle of the story and the calming ending."
            )

        return f"""You are a premium children's bedtime storyteller.

IMPORTANT LANGUAGE RULE:
- The ENTIRE story MUST be written ONLY in {language_name}.
- Do NOT use English unless the language is English.
- Do NOT mix languages.
- All narration, title, and dialogue MUST be in {language_name}.

STORY REQUIREMENTS:
- Child name: {request.childName}
- Age: {request.age}
- Theme: {effective_theme}
- Moral: {request.moral}
- Calm level: {request.calmLevel}
- Tone: warm, magical, calming, bedtime-safe
- Use simple, natural language suitable for reading aloud
- No scary content
- No rushed ending
- Do NOT write "The end"
- End peacefully and softly

LENGTH AND STRUCTURE REQUIREMENTS (IMPORTANT):
- Approximate reading time target: {request.durationMin} minutes
- Target total word count: {target_words} words
- Target page count: {target_pages} pages
- Each page should contain {paragraphs_per_page} full paragraphs
- Each paragraph should contain {sentence_range} sentences
- Each page must be one clear paragraph only
- Do not exceed the target page count
- Every page should feel meaningful, gentle, and concise
- Keep the pacing calm, descriptive, and immersive
- Spend enough time in the middle of the adventure before winding down
- The final pages should slow down naturally into sleep rather than ending abruptly
- {pacing_note}

COMPANION:
- {companion_line}

CHARACTERS:
- {character_instruction}

OUTPUT FORMAT (STRICT):
Return ONLY valid JSON:
{{"title": "...", "pages": ["page 1 text", "page 2 text", "page 3 text"]}}

OUTPUT QUALITY RULES:
- Return a complete bedtime story, not an outline
- Do not include notes, markdown, or explanations outside the JSON
- Make each page rich enough to narrate properly
- Ensure the story length and page fullness match the requested reading time"""

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
