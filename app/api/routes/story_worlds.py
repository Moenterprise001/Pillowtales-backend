from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_story_world_service
from app.models.story_world import StoryWorldAdventureListResponse, StoryWorldCanonStoryListResponse, StoryWorldListResponse, StoryWorldPublic
from app.services.story_world_service import StoryWorldService

router = APIRouter(prefix='/story-worlds', tags=['story-worlds'])


@router.get('', response_model=StoryWorldListResponse)
async def list_story_worlds(
    language: str = Query(default='en', min_length=2, max_length=10),
    region: Optional[str] = Query(default=None, max_length=80),
    category: Optional[str] = Query(default=None, max_length=80),
    age: Optional[int] = Query(default=None, ge=1, le=12),
    service: StoryWorldService = Depends(get_story_world_service),
) -> StoryWorldListResponse:
    worlds = service.list_public(language=language, region=region, category=category, age=age)
    return StoryWorldListResponse(storyWorlds=worlds, count=len(worlds))


@router.get('/featured', response_model=StoryWorldListResponse)
async def list_featured_story_worlds(
    language: str = Query(default='en', min_length=2, max_length=10),
    month: Optional[int] = Query(default=None, ge=1, le=12),
    age: Optional[int] = Query(default=None, ge=1, le=12),
    service: StoryWorldService = Depends(get_story_world_service),
) -> StoryWorldListResponse:
    worlds = service.list_featured(language=language, month=month, age=age)
    return StoryWorldListResponse(storyWorlds=worlds, count=len(worlds))


@router.get('/{slug}/stories', response_model=StoryWorldCanonStoryListResponse)
async def list_story_world_original_stories(
    slug: str,
    age: int = Query(..., ge=1, le=12),
    language: str = Query(default='en', min_length=2, max_length=10),
    service: StoryWorldService = Depends(get_story_world_service),
) -> StoryWorldCanonStoryListResponse:
    return service.list_original_stories(slug, age=age, language=language)


@router.get('/{slug}/adventures', response_model=StoryWorldAdventureListResponse)
async def list_story_world_adventures(
    slug: str,
    age: int = Query(..., ge=1, le=12),
    language: str = Query(default='en', min_length=2, max_length=10),
    service: StoryWorldService = Depends(get_story_world_service),
) -> StoryWorldAdventureListResponse:
    return service.list_adventures(slug, age=age, language=language)


@router.get('/{slug}', response_model=StoryWorldPublic)
async def get_story_world(
    slug: str,
    language: str = Query(default='en', min_length=2, max_length=10),
    service: StoryWorldService = Depends(get_story_world_service),
) -> StoryWorldPublic:
    return service.get_public(slug, language=language)
