from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user, get_narration_service, get_subscription_service, get_user_repo
from app.models.narration import NarrationRequest, NarrationResponse, PageStatusResponse
from app.models.subscription import SubscriptionResponse
from app.repositories.user_repository import UserRepository
from app.services.narration_service import NarrationService
from app.services.subscription_service import SubscriptionService

router = APIRouter(prefix='/narration', tags=['narration'])


@router.post('/request', response_model=NarrationResponse)
@router.post('/request-chunked', response_model=NarrationResponse)
async def request_narration(
    request: NarrationRequest,
    user_id: str = Depends(get_current_user),
    narration_service: NarrationService = Depends(get_narration_service),
) -> NarrationResponse:
    return narration_service.request_narration(user_id, request)


@router.get('/page-status', response_model=PageStatusResponse)
async def get_page_status(
    story_id: str,
    narrator: str | None = None,
    lang: str | None = None,
    user_id: str = Depends(get_current_user),
    narration_service: NarrationService = Depends(get_narration_service),
) -> PageStatusResponse:
    return narration_service.get_page_status(user_id, story_id, narrator, lang)


@router.get('/page-audio')
async def get_page_audio(
    story_id: str,
    page: int,
    narrator: str | None = None,
    lang: str | None = None,
    user_id: str = Depends(get_current_user),
    narration_service: NarrationService = Depends(get_narration_service),
) -> dict:
    return narration_service.get_page_audio_url(user_id, story_id, page, narrator, lang)


@router.get('/usage', response_model=SubscriptionResponse)
async def get_narration_usage(
    user_id: str = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repo),
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> SubscriptionResponse:
    profile = user_repo.get_profile(user_id) or {}
    return subscription_service.get_subscription(user_id, profile.get('email'))
