from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user, get_story_repo, get_story_service, get_subscription_service, get_user_repo
from app.models.story import (
    GenerateStoryRequest,
    StoryFeedbackRequest,
    StoryFeedbackResponse,
    StoryResponse,
    UpdateStoryRequest,
)
from app.repositories.story_repository import StoryRepository
from app.repositories.user_repository import UserRepository
from app.services import story_quality

_BACKGROUND_TASKS: set = set()
from app.services.story_service import StoryService
from app.services.subscription_service import SubscriptionService
from app.utils.story_text import preview_from_pages

router = APIRouter(tags=['stories'])
logger = logging.getLogger(__name__)


@router.post('/generateStory', response_model=StoryResponse)
async def generate_story(request: GenerateStoryRequest, user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo), story_repo: StoryRepository = Depends(get_story_repo), story_service: StoryService = Depends(get_story_service), subscription_service: SubscriptionService = Depends(get_subscription_service)) -> StoryResponse:
    if request.userId != user_id:
        logger.warning('[STORY_REQUEST_REJECTED] user_id=%s request_user_id=%s reason=user_id_mismatch', user_id, request.userId)
        raise HTTPException(status_code=403, detail='Unauthorized: user_id mismatch')
    logger.info(
        '[STORY_REQUEST] user_id=%s child_age=%s story_lang=%s narration_lang=%s theme=%s moral=%s custom_moral=%s companion=%s',
        user_id,
        request.age,
        request.storyLanguageCode,
        request.narrationLanguageCode,
        request.customTheme or request.theme,
        request.moral,
        bool(getattr(request, 'customMoral', None)),
        getattr(request, 'companionId', None),
    )
    # Consistency 20261109 (fix 4): reject unsuitable custom theme/moral text
    # before any generation or quota use. Preset themes/morals never match.
    unsafe_term = story_quality.unsafe_custom_text(
        request.customTheme, request.customMoral,
        request.theme if request.customTheme is None else None,
        request.moral if request.customMoral is None else None,
    )
    if unsafe_term:
        logger.warning('[STORY_REQUEST_REJECTED] user_id=%s reason=unsafe_custom_text term=%s', user_id, unsafe_term)
        raise HTTPException(
            status_code=400,
            detail='That theme or moral is not suitable for a children\'s bedtime story. Please choose another.',
        )

    profile = user_repo.get_profile(user_id)
    if not profile:
        logger.warning('[PROFILE_MISSING] user_id=%s endpoint=/api/generateStory', user_id)
        raise HTTPException(status_code=404, detail='User profile not found')
    subscription = subscription_service.get_subscription(user_id, profile.get('email'))
    story_service.validate_story_limits(user_id, subscription)

    # Consistency 20261109 (fix 4): the parent's custom moral must reach the
    # story prompts. Previously the service received the literal 'other'.
    custom_moral = (request.customMoral or '').strip()[:15] if getattr(request, 'customMoral', None) else ''
    requested_moral = request.moral
    effective_moral = (custom_moral if request.moral == 'other' and custom_moral else request.moral)
    request.moral = effective_moral

    story_quality.trace_set(request, 'story_type', request.storyType or 'unknown')
    story_quality.trace_set(request, 'inputs', {
        'age': request.age,
        'theme': request.theme,
        'custom_theme': request.customTheme,
        'moral_sent': requested_moral,
        'custom_moral': custom_moral or None,
        'effective_moral': effective_moral,
        'moral_branch': story_service._moral_requested(request),
        'language': request.storyLanguageCode,
        'calm_level': request.calmLevel,
        'companion_id': request.companionId,
        'extra_characters': len(request.characters or []),
        'story_world_slug': request.storyWorldSlug,
    })

    # Page-1-first story generation is required for PillowTales performance.
    # Return page 1 as soon as it is ready, then complete pages 2+ in the
    # background. Do not block this route on the full 7-page story.
    t_page1 = time.time()
    story_data = await story_service.generate_story_first_page(request, subscription)
    story_service.freshen_page_one_names(request, story_data)
    story_quality.trace_set(request, 'page1', {
        'source': story_data.get('first_page_generation_source'),
        'fallback_reason': story_data.get('generation_fallback_reason'),
        'seconds': round(time.time() - t_page1, 2),
        'title': story_data.get('title'),
    })
    pages = story_data.get('pages') or []
    full_text = '\n\n'.join(pages)
    expected_pages = story_data.get('expected_pages') or 7
    generation_status = story_data.get('generation_status') or ('complete' if len(pages) >= expected_pages else 'partial')
    story_world_metadata = story_service.get_story_world_generation_metadata(request)

    record = {
        'user_id': user_id,
        'title': story_data['title'],
        'child_name': request.childName,
        'age': request.age,
        'theme': request.customTheme or request.theme,
        'moral': effective_moral,
        'calm_level': request.calmLevel,
        'duration_min': request.durationMin,
        'language': request.storyLanguageCode,
        'story_language_code': request.storyLanguageCode,
        'narration_language_code': request.narrationLanguageCode or request.storyLanguageCode,
        'child_name_pronunciation': request.childNamePronunciation,
        'pages': pages,
        'full_text': full_text,
        'audio_url': None,
        'audio_status': 'none',
        'is_favorite': False,
        'companion_id': story_data.get('companion', {}).get('id') if story_data.get('companion') else None,
        'companion_name': story_data.get('companion', {}).get('name') if story_data.get('companion') else None,
        'generation_status': generation_status,
        'expected_pages': expected_pages,
        'generation_error': None,
        **story_world_metadata,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    saved_story = story_repo.insert(record)
    # Fix 5: initial trace (Page 1 facts). The background task overwrites it
    # with the complete trace. Separate update so a missing column (before
    # the migration is applied) can never break story saving.
    story_service.persist_generation_trace(request, saved_story['id'], user_id)
    logger.info(
        '[STORY_PAGE1_READY] user_id=%s story_id=%s title=%s pages_ready=%s expected_pages=%s generation_status=%s',
        user_id,
        saved_story['id'],
        story_data.get('title'),
        len(pages),
        expected_pages,
        generation_status,
    )

    if generation_status == 'complete' or len(pages) >= expected_pages:
        metadata = await story_service.extract_metadata(story_data['title'], full_text)
        try:
            story_repo.update(saved_story['id'], user_id, {
                'story_summary': metadata.get('summary', ''),
                'characters': metadata.get('characters', []),
                'setting': metadata.get('setting', ''),
                'generation_status': 'complete',
                'expected_pages': expected_pages,
                'generation_error': None,
            })
        except Exception:
            pass
    else:
        # Background completion owns pages 2+. This keeps generation fast while
        # allowing the frontend to poll /stories/{id} until expected_pages arrive.
        logger.info(
            '[STORY_BACKGROUND_STARTED] user_id=%s story_id=%s expected_pages=%s current_pages=%s',
            user_id,
            saved_story['id'],
            expected_pages,
            len(pages),
        )
        task = asyncio.create_task(story_service.complete_story_background(
            request=request,
            user_id=user_id,
            story_id=saved_story['id'],
            title=story_data['title'],
            current_pages=pages,
            companion=story_data.get('companion'),
            expected_pages=expected_pages,
        ))
        # Keep a strong reference until the task finishes. asyncio only holds
        # weak references to tasks, so an unreferenced background task can be
        # garbage-collected mid-run (pages 2+ or the trace silently lost).
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)

    return StoryResponse(
        storyId=saved_story['id'],
        title=story_data['title'],
        pages=pages,
        generation_status=generation_status,
        expected_pages=expected_pages,
        generation_error=None,
    )

@router.get('/stories')
async def list_stories(user_id: str = Depends(get_current_user), story_repo: StoryRepository = Depends(get_story_repo)) -> dict:
    stories = story_repo.list_for_user(user_id)
    logger.info('[STORY_LIBRARY_LOAD] user_id=%s count=%s', user_id, len(stories or []))
    return {'stories': stories}


@router.get('/stories/{story_id}/artwork')
async def get_story_artwork(
    story_id: str,
    user_id: str = Depends(get_current_user),
    story_repo: StoryRepository = Depends(get_story_repo),
    story_service: StoryService = Depends(get_story_service),
) -> dict:
    """Return optional Story World artwork without blocking the story GET path."""
    story = story_repo.get(story_id, user_id)
    if not story:
        raise HTTPException(status_code=404, detail='Story not found')

    artwork = story_service.get_story_world_artwork_for_saved_story(story)
    return {'artwork': artwork}


@router.get('/stories/{story_id}')
async def get_story(story_id: str, user_id: str = Depends(get_current_user), story_repo: StoryRepository = Depends(get_story_repo)) -> dict:
    story = story_repo.get(story_id, user_id)
    if not story:
        raise HTTPException(status_code=404, detail='Story not found')

    pages = story.get('pages') or []
    final_page = str(pages[-1] or '') if pages else ''
    logger.info(
        '[API_STORY_RESPONSE] user_id=%s story_id=%s status=%s expected_pages=%s page_count=%s final_page_chars=%s final_page_tail=%r',
        user_id,
        story_id,
        story.get('generation_status'),
        story.get('expected_pages'),
        len(pages),
        len(final_page),
        final_page[-120:],
    )
    return story


@router.post('/stories/{story_id}/feedback', response_model=StoryFeedbackResponse)
async def submit_story_feedback(
    story_id: str,
    request: StoryFeedbackRequest,
    user_id: str = Depends(get_current_user),
    story_repo: StoryRepository = Depends(get_story_repo),
) -> StoryFeedbackResponse:
    story = story_repo.get(story_id, user_id)
    if not story:
        raise HTTPException(status_code=404, detail='Story not found')

    pages = story.get('pages') or []
    parent_voice_used = request.parentVoiceUsed
    if parent_voice_used is None:
        parent_voice_used = bool(
            story.get('parent_voice_used')
            or story.get('voice_recording_id')
            or str(story.get('audio_provider') or '').lower() == 'elevenlabs'
        )

    narrator = request.narrator or story.get('narrator') or story.get('audio_voice') or story.get('voice_name')

    feedback_record = {
        'story_id': story_id,
        'user_id': user_id,
        'rating': request.rating,
        'feedback': request.feedback,
        'would_like_similar_stories': request.wouldLikeSimilarStories,
        'child_fell_asleep': request.childFellAsleep,
        'comment': request.comment,
        'child_age': story.get('age'),
        'theme': story.get('theme'),
        'moral': story.get('moral'),
        'story_language_code': story.get('story_language_code') or story.get('language'),
        'narration_language_code': (
            story.get('narration_language_code')
            or story.get('audio_language_code')
            or story.get('story_language_code')
            or story.get('language')
        ),
        'narrator': narrator,
        'parent_voice_used': parent_voice_used,
        'duration_min': story.get('duration_min'),
        'page_count': len(pages),
        'generation_status': story.get('generation_status'),
        'generation_time_seconds': story.get('generation_time_seconds'),
        'is_continuation': bool(story.get('continue_from_story_id') or story.get('parent_story_id')),
        'story_world_slug': story.get('story_world_slug'),
        'story_world_mode': story.get('story_world_mode'),
        'story_world_anchor_slug': story.get('story_world_anchor_slug'),
        'story_world_prompt_pack_version': story.get('story_world_prompt_pack_version'),
    }

    saved_feedback = story_repo.submit_feedback(feedback_record)
    logger.info(
        '[STORY_FEEDBACK_SAVED] user_id=%s story_id=%s rating=%s feedback_count=%s',
        user_id,
        story_id,
        request.rating,
        len(request.feedback),
    )
    return StoryFeedbackResponse(
        message='Story feedback saved successfully',
        feedback=saved_feedback,
    )


@router.put('/stories/{story_id}')
async def update_story(story_id: str, request: UpdateStoryRequest, user_id: str = Depends(get_current_user), story_repo: StoryRepository = Depends(get_story_repo)) -> dict:
    values = {}
    if request.isFavorite is not None:
        values['is_favorite'] = request.isFavorite
    if not values:
        raise HTTPException(status_code=400, detail='No update data provided')
    story = story_repo.update(story_id, user_id, values)
    return {'message': 'Story updated successfully', 'story': story}


@router.delete('/stories/{story_id}')
async def delete_story(story_id: str, user_id: str = Depends(get_current_user), story_repo: StoryRepository = Depends(get_story_repo)) -> dict:
    story_repo.delete(story_id, user_id)
    return {'message': 'Story deleted successfully'}


@router.get('/story-preview/{story_id}')
async def story_preview(story_id: str, story_repo: StoryRepository = Depends(get_story_repo)) -> dict:
    story = story_repo.get(story_id)
    if not story:
        raise HTTPException(status_code=404, detail='Story not found')
    pages = story.get('pages') or []
    return {'id': story.get('id'), 'title': story.get('title'), 'childName': story.get('child_name'), 'firstParagraph': preview_from_pages(pages), 'pageCount': len(pages), 'duration': f"~{story.get('duration_min', 8)} min", 'language': story.get('language', 'en'), 'createdAt': story.get('created_at')}
