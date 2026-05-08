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


OPENING_SEEDS = [
    "Under a soft silver moon, {childName} snuggled into bed when something tiny and magical flickered near the window.",
    "As the stars began to glow, {childName} pulled the blanket close and noticed a gentle shimmer in the room.",
    "The night was calm and quiet when {childName} felt a soft, magical presence nearby.",
    "Just as {childName} was getting cosy, a small glowing light appeared, as if the night had a secret to share.",
    "The moonlight stretched across the room, and {childName} felt something special was about to begin."
]

FIRST_PAGE_TIMEOUT_SECONDS = 30
# User-facing consistency target: if Gemini has not produced page 1
# quickly enough, return a deterministic page-1 fallback so Reader can open.
# The full story still completes in the normal background Gemini path.
FIRST_PAGE_SOFT_LIMIT_SECONDS = 22


class StoryService:
    def __init__(self, story_repo: StoryRepository):
        self.story_repo = story_repo
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model) if settings.gemini_api_key else None

    def _select_companion(self, request: GenerateStoryRequest, subscription: SubscriptionResponse) -> Optional[dict]:
        # V1 production focus: do not randomly introduce companions.
        # The frontend currently uses the PillowTales bear as the single visual anchor.
        # We still honour an explicit valid companionId from existing clients/backward compatibility,
        # but no longer auto-select a random companion when none is requested.
        if request.companionId and request.companionId in STORY_COMPANIONS:
            companion = STORY_COMPANIONS[request.companionId].copy()
            companion['id'] = request.companionId
            return companion
        return None

    def _storycraft_rules(self) -> str:
        return """STORYCRAFT QUALITY RULES:
- Make the story feel like a premium illustrated children's fantasy tale: imaginative, emotionally warm, cinematic, and magical, while remaining original and bedtime-safe.
- Use a classic storybook arc: wonder-filled opening, gentle discovery, small emotional challenge, magical or meaningful helper moment, moral learned through action, and a satisfying peaceful resolution.
- Let the child make choices, notice details, and grow through the story; do not simply describe events happening around them.
- Include sensory storybook details: soft light, gentle sounds, cozy textures, moonlight, stars, nature, kindness, friendship, courage, patience, or wonder where appropriate.
- Every page should have a clear story purpose: discovery, decision, challenge, help, transformation, reflection, or peaceful closure.
- Avoid flat summaries. Write immersive scenes that feel read-aloud, memorable, and emotionally rewarding.
- Keep the mood safe for bedtime: no danger, no frightening villains, no peril, no sadness-heavy ending.
- Do not copy or imitate any existing franchise, character, studio, film, song, or copyrighted story world."""

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

        # Standard bedtime narration target:
        # The product now uses one optimal story length. Age controls complexity;
        # duration is kept as an internal compatibility field only.
        target_pages = "7"
        paragraphs_per_page = "2"
        sentence_range = "5-7"
        target_words = "1050-1300"
        max_words_per_page = "210"
        pacing_note = (
            "Create a substantial but calm bedtime story suitable for an approximately eight-minute bedtime experience. "
            "Do not compress the plot into a short summary; let each page include a gentle, memorable story moment with sensory detail, child agency, and emotional warmth."
        )

        return f"""You are a premium children's bedtime storyteller.

IMPORTANT LANGUAGE RULE:
- The ENTIRE story MUST be written ONLY in {language_name}.
- Do NOT use English unless the language is English.
- Do NOT mix languages.
- All narration, title, and dialogue MUST be in {language_name}.
- Write naturally for native-speaking children in {language_name}.
- The story must feel like it was originally written in {language_name}, not translated from English.
- Use warm, magical, emotionally comforting bedtime storytelling.
- Avoid overly formal, academic, rigid, or literal phrasing.
- Use natural rhythm and gentle emotional pacing suitable for read-aloud bedtime stories.
{language_style_block}

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

{self._storycraft_rules()}

LENGTH AND STRUCTURE REQUIREMENTS (STRICT PERFORMANCE RULES):
- EXACTLY {target_pages} pages. Do not return more or fewer pages.
- EACH page should contain {paragraphs_per_page} gentle paragraphs.
- EACH page should contain approximately {sentence_range} bedtime-friendly sentences in total.
- TOTAL story length MUST be approximately {target_words} words.
- Each page should be substantial, normally 145-190 words, but DO NOT exceed {max_words_per_page} words on any single page.
- Use simple, natural sentences suitable for spoken bedtime narration.
- Do not make pages too short. Avoid summarising scenes in only one or two sentences.
- Every page must move the story forward gently and include one memorable story beat, a small choice or discovery from the child, and warm bedtime imagery.
- The moral should be discovered through the child's actions, not explained like a lesson.
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
- Keep the story calm and readable, but do not make it too short.
- If unsure, prioritise reaching the requested narration length while staying bedtime-safe.
- The JSON pages array must contain exactly {target_pages} strings."""


    def _intended_page_count(self, request: GenerateStoryRequest) -> int:
        return 7

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

        opening = random.choice(OPENING_SEEDS).replace("{childName}", request.childName)
        language_code = (request.storyLanguageCode or "en").lower()
        is_english = language_code == "en"

        if is_english:
            page_length_rule = "110-140 words total, including the opening sentence"
            sentence_rule = "4-6 calm, read-aloud sentences"
            instruction_block = f"""- Continue naturally from the opening above
- Keep the tone warm, magical, calm, and bedtime-safe
- Use soft sensory details (light, stars, quiet, comfort)
- Let {request.childName} notice or choose something meaningful
- Do NOT introduce danger, fear, or fast pacing
- Do NOT resolve the story yet"""
        else:
            # Non-English first pages can be slower because the model must obey
            # language-only output while generating valid JSON. Keep the same
            # bedtime shape, but reduce output length and instruction load so
            # page 1 is ready faster. The full 7-page story remains unchanged.
            # German TTS is naturally longer/slower, so give page 1 a slightly
            # longer prewarm window to avoid a pause before page 2.
            if language_code == "de":
                page_length_rule = "105-135 words total, including the opening sentence"
                sentence_rule = "4-6 calm, read-aloud sentences"
            else:
                page_length_rule = "85-115 words total, including the opening sentence"
                sentence_rule = "3-5 calm, read-aloud sentences"
            instruction_block = f"""- Continue naturally from the opening above
- Keep the tone warm, magical, calm, and bedtime-safe
- Include one clear, gentle story moment for {request.childName}
- Do NOT introduce danger, fear, or fast pacing
- Do NOT resolve the story yet"""

        return f"""You are continuing a premium children's bedtime story.

IMPORTANT LANGUAGE RULE:
- Write ONLY in {blocks['language_name']}
- Do NOT mix languages

STORY CONTEXT:
- Child: {request.childName}, age {request.age}
- Theme: {blocks['effective_theme']}
- Moral: {request.moral}
- Calm level: {request.calmLevel}

START THE STORY WITH THIS EXACT SENTENCE:
"{opening}"

Then continue immediately from it.

INSTRUCTIONS:
{instruction_block}

PAGE 1 STRUCTURE:
- {page_length_rule}
- 1-2 gentle paragraphs
- {sentence_rule}
- Clear emotional hook into the story

COMPANION:
- {blocks['companion_line']}

CHARACTERS:
- {blocks['character_instruction']}

OUTPUT FORMAT (STRICT):
Return ONLY valid JSON:
{{"title":"Short magical title","pages":["page 1 text"]}}
"""

    def _build_first_page_fallback(self, request: GenerateStoryRequest, companion: Optional[dict]) -> Dict[str, Any]:
        """Fast deterministic page-1 fallback used only when Gemini is too slow.

        This protects the launch UX from occasional LLM latency spikes. The
        remaining pages still complete through Gemini in the normal background
        flow, so full story quality is preserved after the reader opens.
        """
        child = request.childName or "the child"
        language_code = (request.storyLanguageCode or "en").lower()
        expected_pages = self._intended_page_count(request)

        fallback_by_lang = {
            "es": {
                "title": f"El brillo tranquilo de {child}",
                "page": (
                    f"Bajo una luna plateada y suave, {child} se acurrucó en la cama cuando un pequeño brillo mágico apareció junto a la ventana. "
                    "La luz parecía respirar despacio, como si trajera un secreto amable de la noche. "
                    f"{child} levantó la manta con cuidado y sonrió al notar que el cuarto se llenaba de estrellas diminutas. "
                    "Todo estaba en calma, y aquella chispa invitaba a comenzar un cuento lleno de ternura, valor y dulces sueños."
                ),
            },
            "it": {
                "title": f"Il dolce luccichio di {child}",
                "page": (
                    f"Sotto una luna d'argento morbida e silenziosa, {child} si rannicchiò nel letto quando un piccolo bagliore magico brillò vicino alla finestra. "
                    "La luce si muoveva piano, come se portasse un gentile segreto della notte. "
                    f"{child} sollevò la coperta con curiosità e vide minuscole stelle danzare nell'aria. "
                    "Tutto era calmo, caldo e sicuro, e quel luccichio sembrava invitare a un racconto pieno di meraviglia, gentilezza e sogni sereni."
                ),
            },
            "fr": {
                "title": f"La douce lueur de {child}",
                "page": (
                    f"Sous une lune argentée et douce, {child} se blottit dans son lit lorsqu'une petite lueur magique scintilla près de la fenêtre. "
                    "La lumière avançait lentement, comme si elle portait un secret tendre de la nuit. "
                    f"{child} souleva la couverture avec curiosité et vit de minuscules étoiles flotter dans la chambre. "
                    "Tout était calme, chaud et rassurant, et cette lueur semblait inviter à une histoire pleine de douceur, de courage et de beaux rêves."
                ),
            },
            "de": {
                "title": f"Das sanfte Leuchten von {child}",
                "page": (
                    f"Unter einem weichen silbernen Mond kuschelte sich {child} ins Bett, als ein kleines magisches Licht am Fenster funkelte. "
                    "Es schwebte ganz langsam durch das Zimmer, als trüge es ein freundliches Geheimnis der Nacht bei sich. "
                    f"{child} zog die Decke ein wenig höher und lächelte, während winzige Sterne leise in der Luft glitzerten. "
                    "Alles fühlte sich ruhig, warm und sicher an, und das Licht lud zu einer sanften Geschichte voller Wunder, Mut und schöner Träume ein."
                ),
            },
            "en": {
                "title": f"{child}'s Gentle Glow",
                "page": (
                    f"Under a soft silver moon, {child} snuggled into bed when something tiny and magical flickered near the window. "
                    "The little light drifted slowly through the room, as if it carried a kind secret from the night. "
                    f"{child} pulled the blanket close and smiled as tiny stars shimmered in the quiet air. "
                    "Everything felt calm, warm, and safe, and the gentle glow seemed to invite a story filled with wonder, kindness, and peaceful dreams."
                ),
            },
        }

        fallback = fallback_by_lang.get(language_code, fallback_by_lang["en"])
        pages = postprocess_story_pages([fallback["page"]])[:1]
        return {
            'title': fallback['title'],
            'pages': pages,
            'companion': companion,
            'expected_pages': expected_pages,
            'generation_status': 'partial',
            'generation_fallback_reason': 'first_page_timeout',
        }

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
- Continue as if the story was originally written in {blocks['language_name']}.
- Keep the language natural, warm, magical, emotionally comforting, and read-aloud friendly.
- Avoid formal, stiff, academic, or literal phrasing.

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
- The complete story should feel suitable for an approximately eight-minute bedtime experience.

{self._storycraft_rules()}

EXISTING STORY START:
Title: {title}
Page 1: {page_one}

CONTINUATION REQUIREMENTS:
- Write exactly {remaining_page_count} remaining pages.
- Continue naturally from page 1.
- Do not recap page 1.
- Do not contradict page 1.
- Each page should contain 2-3 gentle, story-rich paragraphs.
- Each page should be approximately 150-205 words.
- Each page should contain around 5-7 bedtime-friendly sentences in total.
- Do not make pages too short; each page should feel like a complete story scene with action, sensory detail, and emotional progression, not a summary.
- Every page must move the story forward gently and include one memorable story beat, a small choice or discovery from the child, and warm bedtime imagery.
- The moral should be discovered through the child's actions, not explained like a lesson.
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
            try:
                # Consistency guard: do not let a slow Gemini first-page call hold
                # the user on the generation screen. If page 1 is not back within
                # the soft limit, return a polished deterministic page 1 and let
                # the remaining story continue through the normal background path.
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.model.generate_content, prompt),
                    timeout=FIRST_PAGE_SOFT_LIMIT_SECONDS,
                )
            except asyncio.TimeoutError:
                elapsed = time.time() - t_gemini
                print(
                    f"[PERF] first_page Gemini soft limit hit after {elapsed:.2f}s; "
                    "using fast fallback page 1"
                )
                fallback = self._build_first_page_fallback(request, companion)
                print(f"[PERF] generate_story_first_page DONE fallback total={time.time() - start_total:.2f}s")
                print("[PERF] ========================================")
                return fallback

            elapsed = time.time() - t_gemini
            print(f"[PERF] first_page Gemini took {elapsed:.2f}s")

            response_text = getattr(response, 'text', None)
            if not response_text or not isinstance(response_text, str):
                raise ValueError('Failed to generate first page')

            story_data = self._clean_json_response(response_text)
            if not isinstance(story_data, dict) or 'title' not in story_data or 'pages' not in story_data:
                raise ValueError('Invalid first-page story format returned by AI')

            pages = postprocess_story_pages(story_data.get('pages', []))[:1]
            if not pages:
                raise ValueError('First-page story returned no pages')

            print(f"[PERF] first_page_ready_for_response pages=1 expected_pages={expected_pages}")
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
        start_total = time.time()
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

            t_metadata = time.time()
            print(f"[PERF] metadata_extract START story_id={story_id}")
            metadata = await self.extract_metadata(title, full_text)
            print(f"[PERF] metadata_extract DONE story_id={story_id} total={time.time() - t_metadata:.2f}s")
            update_payload.update({
                'story_summary': metadata.get('summary', ''),
                'characters': metadata.get('characters', []),
                'setting': metadata.get('setting', ''),
            })

            t_update = time.time()
            print(f"[PERF] story_update_complete START story_id={story_id}")
            self.story_repo.update(story_id, user_id, update_payload)
            print(f"[PERF] story_update_complete DONE story_id={story_id} total={time.time() - t_update:.2f}s")
            print(f"[PERF] complete_story_background DONE story_id={story_id} pages={len(all_pages)} total={time.time() - start_total:.2f}s")
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
        intended_page_count = 7
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
        start_total = time.time()
        if not self.model:
            print(f"[PERF] extract_metadata skipped no_model total={time.time() - start_total:.2f}s")
            return {'summary': '', 'characters': [], 'setting': ''}
        prompt = (
            'Analyze this bedtime story and return only valid JSON. '
            'Schema: {"summary":"...","characters":[{"name":"...","description":"...","role":"..."}],"setting":"..."}\n'
            f'Title: {title}\nStory:\n{full_text}'
        )
        try:
            t_gemini = time.time()
            response = self.model.generate_content(prompt)
            print(f"[PERF] extract_metadata Gemini took {time.time() - t_gemini:.2f}s")
            text = getattr(response, 'text', '')
            start = text.find('{')
            end = text.rfind('}')
            if start == -1 or end == -1:
                print(f"[PERF] extract_metadata invalid_json total={time.time() - start_total:.2f}s")
                return {'summary': '', 'characters': [], 'setting': ''}
            result = json.loads(text[start:end + 1])
            print(f"[PERF] extract_metadata DONE total={time.time() - start_total:.2f}s")
            return result
        except Exception as exc:
            print(f"[PERF] extract_metadata FAILED total={time.time() - start_total:.2f}s error={exc}")
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
