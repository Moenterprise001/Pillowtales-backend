from __future__ import annotations

import asyncio
import json
import random
import time
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

        # Production performance target:
        # Keep stories concise enough for fast generation, affordable narration,
        # and reliable sentence/page sync in the mobile reader.
        if request.durationMin >= 11:
            target_pages = "9"
            paragraphs_per_page = "1"
            sentence_range = "2-3"
            target_words = "650-800"
            max_words_per_page = "90"
            pacing_note = (
                "Create a complete but concise bedtime journey. Avoid long descriptions, "
                "side quests, or extra scenes. Keep the ending calm and satisfying."
            )
        else:
            target_pages = "7"
            paragraphs_per_page = "1"
            sentence_range = "2-3"
            target_words = "450-600"
            max_words_per_page = "85"
            pacing_note = (
                "Create a gentle, concise bedtime story with a clear beginning, "
                "middle, and peaceful ending. Do not pad the story."
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

LENGTH AND STRUCTURE REQUIREMENTS (STRICT PERFORMANCE RULES):
- EXACTLY {target_pages} pages. Do not return more or fewer pages.
- EXACTLY {paragraphs_per_page} short paragraph per page.
- EACH paragraph MUST contain {sentence_range} short sentences.
- TOTAL story length MUST be {target_words} words.
- DO NOT exceed {max_words_per_page} words on any single page.
- Use short, simple sentences suitable for spoken bedtime narration.
- Avoid long descriptions, repeated phrases, extra subplots, or unnecessary scenes.
- Every page must move the story forward gently.
- The final page must end peacefully and softly.
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
- Keep the story concise. Speed and predictability are more important than extra detail.
- If unsure, write fewer words rather than more words.
- The JSON pages array must contain exactly {target_pages} strings."""


    def _intended_page_count(self, request: GenerateStoryRequest) -> int:
        return 9 if request.durationMin >= 11 else 7

    def _clean_json_response(self, response_text: str) -> Dict[str, Any]:
        cleaned = response_text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())

    def _language_and_character_blocks(self, request: GenerateStoryRequest, companion: Optional[dict]) -> Dict[str, str]:
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

        return {
            'language_name': language_name,
            'effective_theme': effective_theme,
            'companion_line': companion_line,
            'character_instruction': character_instruction,
        }

    def _build_first_page_prompt(self, request: GenerateStoryRequest, companion: Optional[dict]) -> str:
        blocks = self._language_and_character_blocks(request, companion)
        return f"""You are a premium children's bedtime storyteller.

IMPORTANT LANGUAGE RULE:
- The title and page 1 MUST be written ONLY in {blocks['language_name']}.
- Do NOT use English unless the language is English.
- Do NOT mix languages.

STORY REQUIREMENTS:
- Child name: {request.childName}
- Age: {request.age}
- Theme: {blocks['effective_theme']}
- Moral: {request.moral}
- Calm level: {request.calmLevel}
- Tone: warm, magical, calming, bedtime-safe
- Use simple, natural language suitable for reading aloud
- No scary content

PAGE 1 REQUIREMENTS:
- Write ONLY the title and page 1.
- Page 1 should be slightly longer than the other pages: 3-5 gentle bedtime sentences.
- Page 1 should establish the child, setting, companion if any, and emotional hook.
- Do not resolve the story.
- Do not make page 1 feel complete.
- Do NOT write "The end".

COMPANION:
- {blocks['companion_line']}

CHARACTERS:
- {blocks['character_instruction']}

OUTPUT FORMAT (STRICT):
Return ONLY valid JSON:
{{"title": "...", "pages": ["page 1 text"]}}

OUTPUT QUALITY RULES:
- Do not include notes, markdown, or explanations outside the JSON.
- The JSON pages array must contain exactly 1 string."""

    def _build_remaining_pages_prompt(
        self,
        request: GenerateStoryRequest,
        companion: Optional[dict],
        title: str,
        page_one: str,
        remaining_page_count: int,
    ) -> str:
        blocks = self._language_and_character_blocks(request, companion)
        return f"""You are continuing a premium children's bedtime story.

IMPORTANT LANGUAGE RULE:
- Continue ONLY in {blocks['language_name']}.
- Do NOT use English unless the language is English.
- Do NOT mix languages.

ORIGINAL STORY REQUIREMENTS:
- Child name: {request.childName}
- Age: {request.age}
- Theme: {blocks['effective_theme']}
- Moral: {request.moral}
- Calm level: {request.calmLevel}
- Tone: warm, magical, calming, bedtime-safe
- Use simple, natural language suitable for reading aloud
- No scary content
- No rushed ending
- Do NOT write "The end"
- End peacefully and softly on the final page.

EXISTING STORY START:
Title: {title}
Page 1: {page_one}

CONTINUATION REQUIREMENTS:
- Write exactly {remaining_page_count} remaining pages.
- Continue naturally from page 1.
- Do not recap page 1.
- Do not contradict page 1.
- Each page should be 1 short paragraph.
- Each page should contain 2-3 short bedtime-friendly sentences.
- Every page must move the story forward gently.
- The final page must end peacefully and softly.

COMPANION:
- {blocks['companion_line']}

CHARACTERS:
- {blocks['character_instruction']}

OUTPUT FORMAT (STRICT):
Return ONLY valid JSON:
{{"pages": ["page 2 text", "page 3 text"]}}

OUTPUT QUALITY RULES:
- Return continuation pages only.
- Do not include notes, markdown, or explanations outside the JSON.
- The JSON pages array must contain exactly {remaining_page_count} strings."""

    async def generate_story_first_page(self, request: GenerateStoryRequest, subscription: SubscriptionResponse) -> Dict[str, Any]:
        start_total = time.time()
        print("[PERF] ========================================")
        print(f"[PERF] generate_story_first_page START lang={request.storyLanguageCode} duration={request.durationMin}")

        companion = self._select_companion(request, subscription)
        expected_pages = self._intended_page_count(request)

        if not self.model:
            page_one = f"Once upon a time, {request.childName} discovered a quiet little path full of wonder. The stars seemed to listen as the bedtime adventure began. With a calm heart, {request.childName} stepped forward to learn something kind about {request.customTheme or request.theme}."
            pages = postprocess_story_pages([page_one])
            return {
                'title': f"{request.childName}'s Bedtime Adventure",
                'pages': pages[:1],
                'companion': companion,
                'expected_pages': expected_pages,
                'generation_status': 'partial',
            }

        try:
            prompt = self._build_first_page_prompt(request, companion)
            print(f"[PERF] first_page prompt chars={len(prompt)}")
            t_gemini = time.time()
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            print(f"[PERF] first_page Gemini took {time.time() - t_gemini:.2f}s")

            response_text = getattr(response, 'text', None)
            if not response_text or not isinstance(response_text, str):
                raise ValueError('Failed to generate first page')

            story_data = self._clean_json_response(response_text)
            if not isinstance(story_data, dict) or 'title' not in story_data or 'pages' not in story_data:
                raise ValueError('Invalid first-page story format returned by AI')

            pages = postprocess_story_pages(story_data.get('pages', []))[:1]
            if not pages:
                raise ValueError('First-page story returned no pages')

            print(f"[PERF] generate_story_first_page DONE total={time.time() - start_total:.2f}s")
            print("[PERF] ========================================")
            return {
                'title': story_data['title'],
                'pages': pages,
                'companion': companion,
                'expected_pages': expected_pages,
                'generation_status': 'partial',
            }
        except Exception as exc:
            print(f"[PERF] first_page failed, falling back to full story: {exc}")
            full_story = await self.generate_story(request, subscription)
            full_story['expected_pages'] = len(full_story.get('pages') or [])
            full_story['generation_status'] = 'complete'
            return full_story

    async def complete_story_background(
        self,
        request: GenerateStoryRequest,
        user_id: str,
        story_id: str,
        title: str,
        current_pages: list[str],
        companion: Optional[dict],
        expected_pages: int,
    ) -> None:
        print(f"[PERF] complete_story_background START story_id={story_id}")
        try:
            if not self.model:
                remaining = [
                    f"On the next part of the path, {request.childName} found a small kindness waiting to be shared.",
                    f"The quiet adventure grew softer and brighter as {request.childName} remembered what mattered most.",
                    f"At last, the moon smiled gently, and {request.childName} felt safe, loved, and ready for sleep.",
                ]
                while len(current_pages) + len(remaining) < expected_pages:
                    remaining.append(f"A peaceful little moment helped {request.childName} feel even calmer.")
            else:
                remaining_count = max(expected_pages - len(current_pages), 0)
                if remaining_count <= 0:
                    self.story_repo.update(story_id, user_id, {
                        'generation_status': 'complete',
                        'expected_pages': expected_pages,
                        'generation_error': None,
                    })
                    return

                prompt = self._build_remaining_pages_prompt(
                    request=request,
                    companion=companion,
                    title=title,
                    page_one=current_pages[0] if current_pages else '',
                    remaining_page_count=remaining_count,
                )
                print(f"[PERF] remaining_pages prompt chars={len(prompt)} expected_remaining={remaining_count}")
                t_gemini = time.time()
                response = await asyncio.to_thread(self.model.generate_content, prompt)
                print(f"[PERF] remaining_pages Gemini took {time.time() - t_gemini:.2f}s")

                response_text = getattr(response, 'text', None)
                if not response_text or not isinstance(response_text, str):
                    raise ValueError('Failed to generate remaining pages')

                story_data = self._clean_json_response(response_text)
                if not isinstance(story_data, dict) or 'pages' not in story_data:
                    raise ValueError('Invalid remaining-pages story format returned by AI')

                remaining = postprocess_story_pages(story_data.get('pages', []))

            all_pages = postprocess_story_pages([*current_pages, *remaining])[:expected_pages]
            if len(all_pages) < expected_pages:
                raise ValueError(f'Remaining generation produced only {len(all_pages)} of {expected_pages} pages')

            full_text = '\n\n'.join(all_pages)
            update_payload = {
                'pages': all_pages,
                'full_text': full_text,
                'generation_status': 'complete',
                'expected_pages': expected_pages,
                'generation_error': None,
            }

            metadata = await self.extract_metadata(title, full_text)
            update_payload.update({
                'story_summary': metadata.get('summary', ''),
                'characters': metadata.get('characters', []),
                'setting': metadata.get('setting', ''),
            })

            self.story_repo.update(story_id, user_id, update_payload)
            print(f"[PERF] complete_story_background DONE story_id={story_id} pages={len(all_pages)}")
        except Exception as exc:
            print(f"[PERF] complete_story_background FAILED story_id={story_id}: {exc}")
            try:
                self.story_repo.update(story_id, user_id, {
                    'generation_status': 'failed',
                    'expected_pages': expected_pages,
                    'generation_error': str(exc)[:500],
                })
            except Exception as update_exc:
                print(f"[PERF] failed to mark story generation failed story_id={story_id}: {update_exc}")

    async def generate_story(self, request: GenerateStoryRequest, subscription: SubscriptionResponse) -> Dict[str, Any]:
        start_total = time.time()
        print("[PERF] ========================================")
        print(f"[PERF] generate_story START lang={request.storyLanguageCode} duration={request.durationMin}")

        companion = self._select_companion(request, subscription)
        print(f"[PERF] companion selected in {time.time() - start_total:.2f}s has_companion={bool(companion)}")

        if not self.model:
            pages = [
                f"Once upon a time, {request.childName} discovered a quiet little path full of wonder.",
                f"The path led to a gentle adventure about {request.customTheme or request.theme}, where kindness mattered most.",
                f"Soon, everything grew peaceful again, and {request.childName} felt calm enough for sleep.",
            ]
            print(f"[PERF] fallback story returned in {time.time() - start_total:.2f}s")
            print("[PERF] ========================================")
            return {'title': f"{request.childName}'s Bedtime Adventure", 'pages': postprocess_story_pages(pages), 'companion': companion}

        t_prompt = time.time()
        prompt = self._build_prompt(request, companion)
        print(f"[PERF] prompt built in {time.time() - t_prompt:.2f}s chars={len(prompt)}")

        t_gemini = time.time()
        response = self.model.generate_content(prompt)
        print(f"[PERF] Gemini generate_content took {time.time() - t_gemini:.2f}s")

        response_text = getattr(response, 'text', None)
        if not response_text or not isinstance(response_text, str):
            print(f"[PERF] generate_story FAILED no response text total={time.time() - start_total:.2f}s")
            print("[PERF] ========================================")
            raise HTTPException(status_code=500, detail='Failed to generate story')

        t_clean = time.time()
        cleaned = response_text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        print(f"[PERF] cleaning took {time.time() - t_clean:.2f}s response_chars={len(response_text)}")

        t_parse = time.time()
        story_data = json.loads(cleaned.strip())
        print(f"[PERF] JSON parse took {time.time() - t_parse:.2f}s")

        if not isinstance(story_data, dict) or 'title' not in story_data or 'pages' not in story_data:
            print(f"[PERF] generate_story FAILED invalid format total={time.time() - start_total:.2f}s")
            print("[PERF] ========================================")
            raise HTTPException(status_code=500, detail='Invalid story format returned by AI')

        t_post = time.time()
        pages = postprocess_story_pages(story_data.get('pages', []))
        print(f"[PERF] postprocess took {time.time() - t_post:.2f}s pages_before_trim={len(pages)}")

        # Hard guard for production performance: Gemini may occasionally exceed
        # the requested page count. Trim to the intended count so narration cost,
        # timing, and reader sync remain predictable.
        intended_page_count = 9 if request.durationMin >= 11 else 7
        story_data['pages'] = pages[:intended_page_count]
        story_data['companion'] = companion

        total_words = sum(len(str(page).split()) for page in story_data['pages'])
        print(
            f"[PERF] generate_story DONE total={time.time() - start_total:.2f}s "
            f"lang={request.storyLanguageCode} pages={len(story_data['pages'])} words={total_words}"
        )
        print("[PERF] ========================================")
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
