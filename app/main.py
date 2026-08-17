from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware

from app.api.deps import get_story_repo
from app.api.routes import admin, auth, catalog, narration, parent_voice, revenuecat, stories, story_worlds, subscription, system, user
from app.core.config import settings
from app.core.logging import configure_logging
from app.repositories.story_repository import StoryRepository
from app.utils.story_text import preview_from_pages

configure_logging()

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(system.router, prefix=settings.api_prefix)
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(catalog.router, prefix=settings.api_prefix)
app.include_router(subscription.router, prefix=settings.api_prefix)
app.include_router(user.router, prefix=settings.api_prefix)
app.include_router(parent_voice.router, prefix=settings.api_prefix)
app.include_router(stories.router, prefix=settings.api_prefix)
app.include_router(story_worlds.router, prefix=settings.api_prefix)
app.include_router(narration.router, prefix=settings.api_prefix)
app.include_router(revenuecat.router, prefix=settings.api_prefix)
app.include_router(admin.router, prefix=settings.api_prefix)


def _public_story_payload(story: dict) -> dict:
    """Return only the fields needed by the public shared-story page.

    This keeps the share page public without exposing internal user/account fields.
    """
    pages = story.get('pages') or []
    if isinstance(pages, str):
        # Defensive fallback in case older rows stored pages as a string.
        pages = [pages]

    return {
        'id': story.get('id'),
        'storyId': story.get('id'),
        'title': story.get('title'),
        'childName': story.get('child_name'),
        'child_name': story.get('child_name'),
        'pages': pages,
        'firstParagraph': preview_from_pages(pages),
        'pageCount': len(pages),
        'duration': f"~{story.get('duration_min', 8)} min",
        'durationMin': story.get('duration_min', 8),
        'language': story.get('language') or story.get('story_language_code') or 'en',
        'story_language_code': story.get('story_language_code') or story.get('language') or 'en',
        'createdAt': story.get('created_at'),
        'created_at': story.get('created_at'),
        'generation_status': story.get('generation_status', 'complete'),
        'expected_pages': story.get('expected_pages'),
    }


async def _get_public_story(story_id: str, story_repo: StoryRepository) -> dict:
    story = story_repo.get(story_id)
    if not story:
        raise HTTPException(status_code=404, detail='Story not found')
    return _public_story_payload(story)


# Public shared-story endpoints.
# These are intentionally unauthenticated so links like
# https://pillowtales.co/story/{story_id} can load the story on the website.
# Existing authenticated app routes remain under settings.api_prefix via stories.router.
@app.get('/story/{story_id}')
async def public_story_root(
    story_id: str,
    story_repo: StoryRepository = Depends(get_story_repo),
) -> dict:
    return await _get_public_story(story_id, story_repo)


@app.get('/story-preview/{story_id}')
async def public_story_preview_root(
    story_id: str,
    story_repo: StoryRepository = Depends(get_story_repo),
) -> dict:
    story = story_repo.get(story_id)
    if not story:
        raise HTTPException(status_code=404, detail='Story not found')
    pages = story.get('pages') or []
    if isinstance(pages, str):
        pages = [pages]
    return {
        'id': story.get('id'),
        'title': story.get('title'),
        'childName': story.get('child_name'),
        'firstParagraph': preview_from_pages(pages),
        'pageCount': len(pages),
        'duration': f"~{story.get('duration_min', 8)} min",
        'language': story.get('language', 'en'),
        'createdAt': story.get('created_at'),
    }


@app.get(f'{settings.api_prefix}/story/{{story_id}}')
async def public_story_api_alias(
    story_id: str,
    story_repo: StoryRepository = Depends(get_story_repo),
) -> dict:
    return await _get_public_story(story_id, story_repo)


@app.on_event('startup')
async def startup() -> None:
    return None


@app.on_event('shutdown')
async def shutdown() -> None:
    return None
