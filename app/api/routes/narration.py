from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from app.api.deps import get_current_user, get_narration_service
from app.models.narration import NarrationRequest, NarrationResponse, PageStatusResponse
from app.services.narration_service import NarrationService

router = APIRouter(prefix='/narration', tags=['narration'])


def _get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@router.post('/request', response_model=NarrationResponse)
@router.post('/request-chunked', response_model=NarrationResponse)
async def request_narration(
    request: Request,
    background_tasks: BackgroundTasks,
    narration_request: NarrationRequest,
    user_id: str = Depends(get_current_user),
    narration_service: NarrationService = Depends(get_narration_service),
) -> NarrationResponse:
    client_ip = _get_client_ip(request)
    return narration_service.request_narration(
        user_id,
        narration_request,
        client_ip=client_ip,
        background_tasks=background_tasks,
    )


@router.get('/usage')
async def get_narration_usage(
    user_id: str = Depends(get_current_user),
    narration_service: NarrationService = Depends(get_narration_service),
) -> dict:
    return narration_service.get_narration_usage(user_id)


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
