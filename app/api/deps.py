from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import Client

from app.db.supabase import get_supabase_client
from app.repositories.story_repository import StoryRepository
from app.repositories.story_world_repository import StoryWorldRepository
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthService
from app.services.narration_service import NarrationService
from app.services.story_service import StoryService
from app.services.story_world_service import StoryWorldService
from app.services.story_world_pronunciation_service import StoryWorldPronunciationService
from app.services.subscription_service import SubscriptionService

security = HTTPBearer(auto_error=True)
_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = get_supabase_client()
    return _client


def get_user_repo(client: Client = Depends(get_client)) -> UserRepository:
    return UserRepository(client)


def get_story_repo(client: Client = Depends(get_client)) -> StoryRepository:
    return StoryRepository(client)


def get_story_world_repo(client: Client = Depends(get_client)) -> StoryWorldRepository:
    return StoryWorldRepository(client)


def get_story_world_service(
    repository: StoryWorldRepository = Depends(get_story_world_repo),
) -> StoryWorldService:
    return StoryWorldService(repository)


def get_story_world_pronunciation_service(
    repository: StoryWorldRepository = Depends(get_story_world_repo),
) -> StoryWorldPronunciationService:
    return StoryWorldPronunciationService(
    	repository,
    	allow_unverified=True,
)


def get_auth_service(client: Client = Depends(get_client)) -> AuthService:
    return AuthService(client)


def get_subscription_service(user_repo: UserRepository = Depends(get_user_repo), story_repo: StoryRepository = Depends(get_story_repo)) -> SubscriptionService:
    return SubscriptionService(user_repo, story_repo)


def get_story_service(
    story_repo: StoryRepository = Depends(get_story_repo),
    story_world_repo: StoryWorldRepository = Depends(get_story_world_repo),
) -> StoryService:
    return StoryService(story_repo, story_world_repo)


def get_narration_service(
    story_repo: StoryRepository = Depends(get_story_repo),
    user_repo: UserRepository = Depends(get_user_repo),
    subscription_service: SubscriptionService = Depends(get_subscription_service),
    story_world_pronunciation_service: StoryWorldPronunciationService = Depends(get_story_world_pronunciation_service),
) -> NarrationService:
    return NarrationService(
        story_repo,
        user_repo,
        subscription_service,
        story_world_pronunciation_service,
    )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), auth_service: AuthService = Depends(get_auth_service)) -> str:
    payload = auth_service.verify_token(credentials.credentials)
    user_id = payload.get('user_id') or payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=401, detail='Invalid token')
    return user_id
