from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user, get_narration_service
from app.models.narration import NarrationRequest, NarrationResponse
from app.services.narration_service import NarrationService

router = APIRouter(prefix='/narration', tags=['narration'])


@router.post('/request', response_model=NarrationResponse)
async def request_narration(request: NarrationRequest, user_id: str = Depends(get_current_user), narration_service: NarrationService = Depends(get_narration_service)) -> NarrationResponse:
    return narration_service.request_narration(user_id, request)
