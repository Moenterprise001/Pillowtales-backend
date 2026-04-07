from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_auth_service, get_user_repo
from app.models.auth import AuthResponse, LoginRequest, SignupRequest
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthService

router = APIRouter(prefix='/auth', tags=['auth'])


@router.post('/signup', response_model=AuthResponse)
async def signup(request: SignupRequest, auth_service: AuthService = Depends(get_auth_service), user_repo: UserRepository = Depends(get_user_repo)) -> AuthResponse:
    return auth_service.signup(request, user_repo)


@router.post('/login', response_model=AuthResponse)
async def login(request: LoginRequest, auth_service: AuthService = Depends(get_auth_service), user_repo: UserRepository = Depends(get_user_repo)) -> AuthResponse:
    return auth_service.login(request, user_repo)
