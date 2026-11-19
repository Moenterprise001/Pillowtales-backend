from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import (
    get_client,
    get_current_user,
    get_story_repo,
    get_story_service,
    get_subscription_service,
    get_user_repo,
)
from app.repositories.story_repository import StoryRepository
from app.repositories.user_repository import UserRepository
from app.seasonal.repository import SeasonalRepository
from app.seasonal.service import SeasonalError, SeasonalService
from app.services.story_service import StoryService
from app.services.subscription_service import SubscriptionService

logger = logging.getLogger('pillowtales.seasonal')
router = APIRouter(prefix='/seasonal-stories', tags=['seasonal-stories'])

# Seasonal stories count toward the free weekly limit like Story Worlds stories.
# Set SEASONAL_ENFORCE_STORY_LIMITS=false on Render to make them free.
ENFORCE_STORY_LIMITS = os.getenv('SEASONAL_ENFORCE_STORY_LIMITS', 'true').strip().lower() != 'false'


def get_seasonal_service(client=Depends(get_client)) -> SeasonalService:
    return SeasonalService(SeasonalRepository(client))


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except SeasonalError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error


class SeasonalPlayRequest(BaseModel):
    language: str = Field(default='en', min_length=2, max_length=10)
    audience: str = Field(default='child')
    age: Optional[int] = Field(default=None, ge=1, le=12)


@router.get('')
async def list_seasonal_celebrations(
    language: str = Query(default='en', min_length=2, max_length=10),
    include_coming_soon: bool = Query(default=False),
    service: SeasonalService = Depends(get_seasonal_service),
) -> dict:
    return _call(service.list_celebrations, language=language, include_coming_soon=include_coming_soon)


@router.get('/{slug}/stories')
async def list_seasonal_stories(
    slug: str,
    language: str = Query(default='en', min_length=2, max_length=10),
    age: Optional[int] = Query(default=None, ge=1, le=12),
    audience: str = Query(default='child'),
    service: SeasonalService = Depends(get_seasonal_service),
) -> dict:
    return _call(service.list_stories, slug, language=language, age=age, audience=audience)


@router.post('/{slug}/stories/{story_slug}/play')
async def play_seasonal_story(
    slug: str,
    story_slug: str,
    request: SeasonalPlayRequest,
    user_id: str = Depends(get_current_user),
    service: SeasonalService = Depends(get_seasonal_service),
    user_repo: UserRepository = Depends(get_user_repo),
    story_repo: StoryRepository = Depends(get_story_repo),
    story_service: StoryService = Depends(get_story_service),
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> dict:
    """Save the exact frozen edition to the family's library and return it."""
    record = _call(service.build_library_record, slug, story_slug, user_id=user_id,
                   language=request.language, audience=request.audience, age=request.age)
    if ENFORCE_STORY_LIMITS:
        profile = user_repo.get_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail='User profile not found')
        subscription = subscription_service.get_subscription(user_id, profile.get('email'))
        story_service.validate_story_limits(user_id, subscription)
    saved = story_repo.insert(record)
    logger.info('[SEASONAL_STORY_SAVED] user_id=%s story_id=%s theme=%s language=%s audience=%s age=%s pages=%s',
                user_id, saved['id'], record['theme'], record['language'], request.audience,
                record['age'], len(record['pages']))
    return {
        'storyId': saved['id'],
        'title': record['title'],
        'pages': record['pages'],
        'generation_status': 'complete',
        'expected_pages': record['expected_pages'],
        'generation_error': None,
        'language': record['language'],
    }


@router.get('/{slug}')
async def get_seasonal_celebration(
    slug: str,
    language: str = Query(default='en', min_length=2, max_length=10),
    service: SeasonalService = Depends(get_seasonal_service),
) -> dict:
    return _call(service.get_celebration, slug, language=language)


def seasonal_artwork_for_saved_story(story: dict) -> Optional[dict]:
    """Pictures for a Seasonal story already in the library (GET /stories/{id}/artwork)."""
    return SeasonalService(SeasonalRepository(get_client())).artwork_for_saved_story(story)
