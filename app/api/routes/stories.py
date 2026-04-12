from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user, get_story_repo, get_story_service, get_subscription_service, get_user_repo
from app.models.story import GenerateStoryRequest, StoryResponse, UpdateStoryRequest
from app.repositories.story_repository import StoryRepository
from app.repositories.user_repository import UserRepository
from app.services.story_service import StoryService
from app.services.subscription_service import SubscriptionService
from app.utils.story_text import preview_from_pages

router = APIRouter(tags=['stories'])


@router.post('/generateStory', response_model=StoryResponse)
async def generate_story(request: GenerateStoryRequest, user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo), story_repo: StoryRepository = Depends(get_story_repo), story_service: StoryService = Depends(get_story_service), subscription_service: SubscriptionService = Depends(get_subscription_service)) -> StoryResponse:
    if request.userId != user_id:
        raise HTTPException(status_code=403, detail='Unauthorized: user_id mismatch')
    profile = user_repo.get_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail='User profile not found')
    subscription = subscription_service.get_subscription(user_id, profile.get('email'))
    story_service.validate_story_limits(user_id, subscription)
    story_data = await story_service.generate_story(request, subscription)
    full_text = '\n\n'.join(story_data['pages'])
    record = {
        'user_id': user_id,
        'title': story_data['title'],
        'child_name': request.childName,
        'age': request.age,
        'theme': request.customTheme or request.theme,
        'moral': request.moral,
        'calm_level': request.calmLevel,
        'duration_min': request.durationMin,
        'language': request.storyLanguageCode,
        'story_language_code': request.storyLanguageCode,
        'narration_language_code': request.narrationLanguageCode or request.storyLanguageCode,
        'child_name_pronunciation': request.childNamePronunciation,
        'pages': story_data['pages'],
        'full_text': full_text,
        'audio_url': None,
        'audio_status': 'none',
        'is_favorite': False,
        'companion_id': story_data.get('companion', {}).get('id') if story_data.get('companion') else None,
        'companion_name': story_data.get('companion', {}).get('name') if story_data.get('companion') else None,
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    saved_story = story_repo.insert(record)
    metadata = await story_service.extract_metadata(story_data['title'], full_text)
    try:
        story_repo.update(saved_story['id'], user_id, {'story_summary': metadata.get('summary', ''), 'characters': metadata.get('characters', []), 'setting': metadata.get('setting', '')})
    except Exception:
        pass
    return StoryResponse(storyId=saved_story['id'], title=story_data['title'], pages=story_data['pages'])


@router.get('/stories')
async def list_stories(user_id: str = Depends(get_current_user), story_repo: StoryRepository = Depends(get_story_repo)) -> dict:
    return {'stories': story_repo.list_for_user(user_id)}


@router.get('/stories/{story_id}')
async def get_story(story_id: str, user_id: str = Depends(get_current_user), story_repo: StoryRepository = Depends(get_story_repo)) -> dict:
    story = story_repo.get(story_id, user_id)
    if not story:
        raise HTTPException(status_code=404, detail='Story not found')
    return story


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
