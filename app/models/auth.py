from __future__ import annotations

from pydantic import BaseModel


class SignupRequest(BaseModel):
    email: str
    password: str
    preferredLanguage: str = 'en'


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    userId: str
    email: str
    preferredLanguage: str
