from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv()

@dataclass(frozen=True)
class Settings:
    app_name: str = 'PillowTales API'
    api_prefix: str = '/api'
    gemini_model: str = os.getenv('GEMINI_MODEL=gemini-2.5-flash')
    gemini_api_key: str = os.getenv('GEMINI_API_KEY', '')
    supabase_url: str = os.getenv('SUPABASE_URL', '')
    supabase_service_role_key: str = os.getenv('SUPABASE_SERVICE_ROLE_KEY', '')
    jwt_secret: str = os.getenv('JWT_SECRET', '')
    jwt_algorithm: str = 'HS256'
    jwt_expiration_hours: int = 24 * 30
    log_level: str = os.getenv('LOG_LEVEL', 'INFO').upper()
    cors_allow_origins_raw: str = os.getenv('CORS_ALLOW_ORIGINS', '*')
    narration_backend: str = os.getenv('NARRATION_BACKEND', 'queue_stub')
    narration_queue_name: str = os.getenv('NARRATION_QUEUE_NAME', 'pillowtales-narration')
    allow_origins: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'allow_origins',
            [v.strip() for v in self.cors_allow_origins_raw.split(',') if v.strip()],
        )


settings = Settings()
