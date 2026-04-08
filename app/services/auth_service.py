from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from jose import JWTError, jwt
from supabase import Client

from app.core.config import settings
from app.models.auth import AuthResponse, LoginRequest, SignupRequest
from app.repositories.user_repository import UserRepository


class AuthService:
    def __init__(self, client: Client):
        self.client = client

    def create_access_token(self, data: dict) -> str:
        payload = data.copy()
        payload["exp"] = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expiration_hours)
        return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

    def verify_token(self, token: str) -> dict:
        try:
            return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        except JWTError:
            pass

        try:
            response = self.client.auth.get_user(token)
            if response and response.user:
                return {
                    "user_id": response.user.id,
                    "sub": response.user.id,
                    "email": response.user.email,
                }
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    def signup(self, request: SignupRequest, user_repo: UserRepository) -> AuthResponse:
        try:
            auth_response = self.client.auth.sign_up(
                {
                    "email": request.email,
                    "password": request.password,
                }
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Supabase signup failed: {str(e)}")

        if not auth_response.user:
            raise HTTPException(status_code=400, detail="Failed to create auth user")

        existing_profile = user_repo.get_by_id(auth_response.user.id)

        if existing_profile:
            profile = existing_profile
        else:
            profile = user_repo.create_profile(
                {
                    "id": auth_response.user.id,
                    "preferred_language": request.preferredLanguage,
                    "bedtime_mode": False,
                    "plan": "free",
                }
            )

        token = self.create_access_token(
            {
                "user_id": profile["id"],
                "email": request.email,
            }
        )

        return AuthResponse(
            token=token,
            userId=profile["id"],
            email=request.email,
            preferredLanguage=profile.get("preferred_language", "en"),
        )

    def login(self, request: LoginRequest, user_repo: UserRepository) -> AuthResponse:
        try:
            auth_response = self.client.auth.sign_in_with_password(
                {
                    "email": request.email,
                    "password": request.password,
                }
            )
        except Exception as e:
            if "Invalid login credentials" in str(e):
                raise HTTPException(status_code=401, detail="Invalid email or password")
            raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

        if not auth_response.user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        profile = user_repo.get_profile(auth_response.user.id)
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")

        token = self.create_access_token(
            {
                "user_id": auth_response.user.id,
                "email": request.email,
            }
        )

        return AuthResponse(
            token=token,
            userId=auth_response.user.id,
            email=request.email,
            preferredLanguage=profile.get("preferred_language", "en"),
        )