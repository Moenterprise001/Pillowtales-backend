from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.cors import CORSMiddleware
from supabase import Client, create_client
import google.generativeai as genai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("pillowtales.api")


@dataclass(frozen=True)
class Settings:
    app_name: str = "PillowTales API"
    api_prefix: str = "/api"
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    jwt_secret: str = os.getenv("JWT_SECRET", "")
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24 * 30
    allow_origins: List[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allow_origins",
            [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if origin.strip()],
        )


settings = Settings()

if not settings.supabase_url or not settings.supabase_service_role_key:
    logger.warning("Supabase environment variables are not fully configured.")
if not settings.jwt_secret:
    logger.warning("JWT_SECRET is missing. Authentication will not be secure until configured.")

# ---------------------------------------------------------------------------
# Static domain config
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English (British)",
    "es": "Spanish (Castilian/Spain)",
    "fr": "French",
    "de": "German",
    "it": "Italian",
}

VOICE_PRESETS: Dict[str, Dict[str, Any]] = {
    "wise_owl": {
        "provider": "openai",
        "voice_id": "shimmer",
        "name": "Wise Owl",
        "description": "Calm British bedtime narration, gentle and reassuring",
        "icon": "🦉",
        "language_code": "en",
        "tier": "free",
    },
    "parent_voice": {
        "provider": "elevenlabs",
        "voice_id": None,
        "name": "Parent Voice",
        "description": "Your own voice reads stories",
        "icon": "❤️",
        "language_code": "all",
        "tier": "premium",
        "requires_setup": True,
    },
    "night_owl_spanish": {
        "provider": "openai",
        "voice_id": "shimmer",
        "name": "Búho Sabio",
        "description": "Suave voz española para dormir",
        "icon": "🦉",
        "language_code": "es",
        "tier": "free",
    },
    "night_owl_german": {
        "provider": "openai",
        "voice_id": "shimmer",
        "name": "Weise Eule",
        "description": "Sanfte deutsche Stimme zum Einschlafen",
        "icon": "🦉",
        "language_code": "de",
        "tier": "free",
    },
    "night_owl_french": {
        "provider": "openai",
        "voice_id": "shimmer",
        "name": "Hibou Sage",
        "description": "Douce voix française pour dormir",
        "icon": "🦉",
        "language_code": "fr",
        "tier": "free",
    },
    "night_owl_italian": {
        "provider": "openai",
        "voice_id": "shimmer",
        "name": "Gufo Saggio",
        "description": "Dolce voce italiana per dormire",
        "icon": "🦉",
        "language_code": "it",
        "tier": "free",
    },
}

STORY_COMPANIONS: Dict[str, Dict[str, Any]] = {
    "luna_owl": {
        "name": "Luna the Moon Owl",
        "short_name": "Luna",
        "icon": "🦉",
        "description": "A wise little owl who glows softly in moonlight",
        "tier": "free",
    },
    "milo_fox": {
        "name": "Milo the Sleepy Fox",
        "short_name": "Milo",
        "icon": "🦊",
        "description": "A cozy fox who knows all the best sleeping spots",
        "tier": "free",
    },
    "spark_dragon": {
        "name": "Spark the Tiny Dragon",
        "short_name": "Spark",
        "icon": "🐉",
        "description": "A palm-sized dragon who breathes warm, sparkly light",
        "tier": "premium",
    },
    "stella_fairy": {
        "name": "Stella the Star Fairy",
        "short_name": "Stella",
        "icon": "✨",
        "description": "A tiny fairy who sprinkles sleepy stardust",
        "tier": "premium",
    },
    "bramble_bear": {
        "name": "Bramble the Gentle Bear",
        "short_name": "Bramble",
        "icon": "🐻",
        "description": "A soft, cuddly bear who gives the best hugs",
        "tier": "premium",
    },
}

SUBSCRIPTION_TIERS: Dict[str, Dict[str, Any]] = {
    "free": {
        "weekly_story_limit": 2,
        "weekly_narration_limit": 2,
        "max_saved_stories": 10,
        "narrators": [
            "wise_owl",
            "night_owl_spanish",
            "night_owl_german",
            "night_owl_french",
            "night_owl_italian",
        ],
        "companions": ["luna_owl", "milo_fox"],
        "parent_voice": False,
    },
    "premium": {
        "weekly_story_limit": None,
        "weekly_narration_limit": None,
        "max_saved_stories": None,
        "narrators": list(VOICE_PRESETS.keys()),
        "companions": list(STORY_COMPANIONS.keys()),
        "parent_voice": True,
    },
}

TESTER_EMAILS = {
    "qa@pillowtales.app",
    "test@pillowtales.app",
    "qa@pillowtales.co",
    "test@pillowtales.co",
    "logintest@pillowtales.app",
    "dev@pillowtales.app",
    "dev@pillowtales.co",
    "support@pillowtales.co",
    "hello@pillowtales.co",
}

security = HTTPBearer(auto_error=True)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SignupRequest(BaseModel):
    email: str
    password: str
    preferredLanguage: str = "en"


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    userId: str
    email: str
    preferredLanguage: str


class StoryCharacter(BaseModel):
    name: str
    relationship: str


class GenerateStoryRequest(BaseModel):
    userId: str
    childName: str
    age: int = Field(ge=1, le=12)
    theme: str
    moral: str
    calmLevel: str
    durationMin: int = Field(ge=5, le=20)
    storyLanguageCode: str = "en"
    narrationLanguageCode: Optional[str] = None
    continueFromStoryId: Optional[str] = None
    characters: Optional[List[StoryCharacter]] = None
    customTheme: Optional[str] = None
    companionId: Optional[str] = None
    gender: str = "neutral"


class StoryResponse(BaseModel):
    storyId: str
    title: str
    pages: List[str]


class UpdateStoryRequest(BaseModel):
    isFavorite: Optional[bool] = None


class UserProfileResponse(BaseModel):
    id: str
    email: str
    plan: str
    preferred_language: str
    stories_this_week: int
    stories_saved: int
    can_generate: bool
    can_save_more: bool


class SubscriptionResponse(BaseModel):
    status: str
    is_premium: bool
    is_tester: bool = False
    weekly_narrations_used: int = 0
    weekly_limit: Optional[int] = None
    can_narrate: bool = True
    narrations_remaining: Optional[int] = None


class NarrationRequest(BaseModel):
    storyId: str
    narrationLanguageCode: Optional[str] = None
    voicePreference: Optional[str] = None


class NarrationResponse(BaseModel):
    status: str
    audioUrl: Optional[str] = None
    message: Optional[str] = None


class StoryRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    user_id: str
    title: str
    child_name: str
    age: int
    theme: str
    moral: str
    calm_level: str
    duration_min: int
    language: str
    story_language_code: str = "en"
    narration_language_code: Optional[str] = None
    pages: List[str]
    full_text: str
    is_favorite: bool = False
    created_at: str


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------


class SupabaseRepository:
    def __init__(self, client: Client):
        self.client = client

    # ------------------------------ users ------------------------------
    def get_user_profile(self, user_id: str) -> Optional[dict]:
        result = self.client.table("users_profile").select("*").eq("id", user_id).limit(1).execute()
        return result.data[0] if result.data else None

    def create_user_profile(self, profile: dict) -> dict:
        result = self.client.table("users_profile").insert(profile).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create user profile")
        return result.data[0]

    def update_user_profile(self, user_id: str, values: dict) -> dict:
        result = self.client.table("users_profile").update(values).eq("id", user_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="User not found")
        return result.data[0]

    # ------------------------------ stories ----------------------------
    def insert_story(self, record: dict) -> dict:
        result = self.client.table("stories").insert(record).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to save story")
        return result.data[0]

    def list_stories(self, user_id: str) -> List[dict]:
        result = self.client.table("stories").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return result.data or []

    def get_story(self, story_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        query = self.client.table("stories").select("*").eq("id", story_id)
        if user_id:
            query = query.eq("user_id", user_id)
        result = query.limit(1).execute()
        return result.data[0] if result.data else None

    def update_story(self, story_id: str, user_id: str, values: dict) -> dict:
        result = self.client.table("stories").update(values).eq("id", story_id).eq("user_id", user_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Story not found")
        return result.data[0]

    def delete_story(self, story_id: str, user_id: str) -> None:
        existing = self.get_story(story_id, user_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Story not found")
        self.client.table("stories").delete().eq("id", story_id).eq("user_id", user_id).execute()

    def count_stories_since(self, user_id: str, since_iso: str) -> int:
        result = self.client.table("stories").select("id", count="exact").eq("user_id", user_id).gte("created_at", since_iso).execute()
        return getattr(result, "count", 0) or 0

    def count_stories(self, user_id: str) -> int:
        result = self.client.table("stories").select("id", count="exact").eq("user_id", user_id).execute()
        return getattr(result, "count", 0) or 0


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
        except Exception as exc:  # pragma: no cover - external dependency
            logger.warning("Supabase token verification failed: %s", exc)

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    def signup(self, request: SignupRequest, repo: SupabaseRepository) -> AuthResponse:
        auth_response = self.client.auth.sign_up({"email": request.email, "password": request.password})
        if not auth_response.user:
            raise HTTPException(status_code=400, detail="Failed to create auth user")

        profile = repo.create_user_profile(
            {
                "id": auth_response.user.id,
                "email": request.email,
                "preferred_language": request.preferredLanguage,
                "plan": "free",
                "subscription_status": "free",
                "bedtime_mode": False,
            }
        )
        token = self.create_access_token({"user_id": profile["id"], "email": request.email})
        return AuthResponse(
            token=token,
            userId=profile["id"],
            email=request.email,
            preferredLanguage=profile.get("preferred_language", "en"),
        )

    def login(self, request: LoginRequest, repo: SupabaseRepository) -> AuthResponse:
        auth_response = self.client.auth.sign_in_with_password({"email": request.email, "password": request.password})
        if not auth_response.user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        profile = repo.get_user_profile(auth_response.user.id)
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")

        token = self.create_access_token({"user_id": auth_response.user.id, "email": request.email})
        return AuthResponse(
            token=token,
            userId=auth_response.user.id,
            email=request.email,
            preferredLanguage=profile.get("preferred_language", "en"),
        )


class SubscriptionService:
    def __init__(self, repo: SupabaseRepository):
        self.repo = repo

    def get_subscription(self, user_id: str, user_email: Optional[str]) -> SubscriptionResponse:
        is_tester = bool(user_email and user_email.lower() in TESTER_EMAILS)
        profile = self.repo.get_user_profile(user_id) or {}

        status_value = profile.get("subscription_status") or profile.get("plan") or "free"
        is_premium = is_tester or status_value == "premium"

        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        narration_count = 0
        try:
            result = self.repo.client.table("stories").select("id", count="exact").eq("user_id", user_id).not_.is_("audio_created_at", "null").gte("audio_created_at", week_ago).execute()
            narration_count = getattr(result, "count", 0) or 0
        except Exception as exc:
            logger.warning("Failed to count narrations for user %s: %s", user_id, exc)

        weekly_limit = None if is_premium else SUBSCRIPTION_TIERS["free"]["weekly_narration_limit"]
        can_narrate = is_premium or narration_count < (weekly_limit or 0)
        remaining = None if weekly_limit is None else max(0, weekly_limit - narration_count)
        return SubscriptionResponse(
            status="premium" if is_premium else "free",
            is_premium=is_premium,
            is_tester=is_tester,
            weekly_narrations_used=narration_count,
            weekly_limit=weekly_limit,
            can_narrate=can_narrate,
            narrations_remaining=remaining,
        )

    def feature_allowed(self, subscription: SubscriptionResponse, feature: str, item_id: Optional[str] = None) -> dict:
        tier_key = "premium" if subscription.is_premium else "free"
        tier = SUBSCRIPTION_TIERS[tier_key]

        if feature == "narrator" and item_id and item_id not in tier["narrators"]:
            return {"allowed": False, "upgrade_required": True, "reason": "premium_narrator"}
        if feature == "companion" and item_id and item_id not in tier["companions"]:
            return {"allowed": False, "upgrade_required": True, "reason": "premium_companion"}
        if feature == "parent_voice" and not tier["parent_voice"]:
            return {"allowed": False, "upgrade_required": True, "reason": "parent_voice_premium"}
        if feature == "narration" and not subscription.can_narrate:
            return {
                "allowed": False,
                "upgrade_required": True,
                "reason": "weekly_narration_limit",
                "used": subscription.weekly_narrations_used,
                "limit": subscription.weekly_limit,
            }
        return {"allowed": True, "upgrade_required": False, "reason": None}


class StoryService:
    def __init__(self, repo: SupabaseRepository):
        self.repo = repo
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_model) if settings.gemini_api_key else None

    @staticmethod
    def clean_story_text(text: str) -> str:
        if not text:
            return text
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        for pattern in [
            r"(?im)^\s*the end\.?\s*$",
            r"(?im)^\s*fin\.?\s*$",
            r"(?im)^\s*finis\.?\s*$",
            r"(?im)^\s*ende\.?\s*$",
            r"(?im)^\s*fine\.?\s*$",
        ]:
            cleaned = re.sub(pattern, "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        deduped: List[str] = []
        for line in lines:
            if not deduped or deduped[-1] != line:
                deduped.append(line)
        return "\n".join(deduped).strip()

    @staticmethod
    def paragraphize(text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        groups: List[str] = []
        for i in range(0, len(sentences), 2):
            groups.append(" ".join(sentences[i : i + 2]))
        return "\n\n".join(groups)

    def postprocess_pages(self, pages: List[str]) -> List[str]:
        cleaned = [self.paragraphize(self.clean_story_text(page)) for page in pages if page and page.strip()]
        if cleaned and cleaned[-1][-1:] not in {".", "!", "?"}:
            cleaned[-1] = f"{cleaned[-1]}."
        return cleaned

    def pick_companion(self, requested_companion_id: Optional[str], subscription: SubscriptionResponse) -> Optional[dict]:
        if requested_companion_id:
            companion = STORY_COMPANIONS.get(requested_companion_id)
            if not companion:
                raise HTTPException(status_code=400, detail="Unknown companionId")
            if not subscription.is_premium and companion.get("tier") == "premium":
                raise HTTPException(status_code=403, detail="Selected companion requires Premium")
            return {"id": requested_companion_id, **companion}

        available = [
            {"id": key, **value}
            for key, value in STORY_COMPANIONS.items()
            if subscription.is_premium or value.get("tier") == "free"
        ]
        return random.choice(available) if available and random.random() < 0.30 else None

    def build_prompt(self, request: GenerateStoryRequest, companion: Optional[dict]) -> str:
        language_name = SUPPORTED_LANGUAGES.get(request.storyLanguageCode, "English")
        theme = request.customTheme or request.theme
        character_lines = ""
        if request.characters:
            items = [f"- {c.name} ({c.relationship})" for c in request.characters]
            character_lines = "\nREAL PEOPLE TO INCLUDE:\n" + "\n".join(items)
        companion_lines = ""
        if companion:
            companion_lines = (
                f"\nCOMPANION:\n"
                f"- Name: {companion['name']}\n"
                f"- Description: {companion['description']}\n"
                f"- Use them naturally and warmly in the story.\n"
            )

        page_target = 8 if request.durationMin <= 8 else 11 if request.durationMin <= 11 else 15

        return f"""
You are a premium children's bedtime storyteller.
Return ONLY valid JSON with keys title and pages.

RULES:
- Language: {language_name}
- Child name: {request.childName}
- Age: {request.age}
- Theme: {theme}
- Moral: {request.moral}
- Calm level: {request.calmLevel}
- Write a soothing bedtime story.
- Keep the story emotionally warm and sleep-friendly.
- Avoid the phrase 'The end'.
- Do not duplicate goodnight lines.
- Use British English when language is English.
- Create about {page_target} pages.
- Each page should be narration-friendly, with short sentences.
- Make the child the hero.
- Finish with a calm complete ending.
{character_lines}
{companion_lines}
FORMAT:
{{"title": "...", "pages": ["...", "..."]}}
""".strip()

    async def generate_story(self, request: GenerateStoryRequest, subscription: SubscriptionResponse) -> dict:
        if request.storyLanguageCode not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail="Unsupported story language")
        companion = self.pick_companion(request.companionId, subscription)
        prompt = self.build_prompt(request, companion)
        user_payload = {
            "childName": request.childName,
            "age": request.age,
            "theme": request.customTheme or request.theme,
            "moral": request.moral,
            "calmLevel": request.calmLevel,
            "durationMin": request.durationMin,
            "gender": request.gender,
        }

        if not self._model:
            # Safe deterministic fallback for local testing if Gemini is unavailable.
            title = f"{request.childName} and the Gentle Night"
            pages = [
                f"Once upon a time, {request.childName} looked up at the quiet evening sky and felt ready for a gentle adventure.",
                f"A soft glow appeared nearby, bringing a feeling of wonder and calm.",
                f"The little adventure helped {request.childName} practise {request.moral} in a warm and caring way.",
                f"Soon the world grew still, the stars shone softly, and {request.childName} felt safe, sleepy, and proud.",
            ]
            return {"title": title, "pages": self.postprocess_pages(pages), "companion": companion}

        try:
            response = self._model.generate_content(f"{prompt}\n\nSTORY INPUT:\n{json.dumps(user_payload, ensure_ascii=False)}")
            text = getattr(response, "text", None)
            if not text or not isinstance(text, str):
                raise ValueError("Gemini returned empty content")
            cleaned = text.strip()
            cleaned = re.sub(r"^```json\s*", "", cleaned)
            cleaned = re.sub(r"^```\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            story = json.loads(cleaned)
            if not isinstance(story, dict) or "title" not in story or "pages" not in story:
                raise ValueError("Invalid story JSON shape")
            pages = story.get("pages") or []
            if not isinstance(pages, list) or not pages:
                raise ValueError("Story pages are missing")
            story["pages"] = self.postprocess_pages([str(page) for page in pages])
            story["companion"] = companion
            return story
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Story generation failed")
            raise HTTPException(status_code=500, detail=f"Failed to generate story: {exc}") from exc

    async def extract_metadata(self, title: str, full_text: str) -> dict:
        if not self._model:
            return {"summary": "", "characters": [], "setting": ""}
        prompt = f"""
Extract metadata from this children's bedtime story.
Return ONLY JSON in this format:
{{"summary":"...","characters":[{{"name":"...","description":"...","role":"..."}}],"setting":"..."}}

TITLE: {title}
TEXT:
{full_text}
"""
        try:
            response = self._model.generate_content(prompt)
            text = getattr(response, "text", None) or ""
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return {"summary": "", "characters": [], "setting": ""}
            return json.loads(match.group())
        except Exception as exc:
            logger.warning("Metadata extraction failed: %s", exc)
            return {"summary": "", "characters": [], "setting": ""}


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------

supabase_client: Client = create_client(settings.supabase_url, settings.supabase_service_role_key)
repo = SupabaseRepository(supabase_client)
auth_service = AuthService(supabase_client)
subscription_service = SubscriptionService(repo)
story_service = StoryService(repo)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    payload = auth_service.verify_token(credentials.credentials)
    user_id = payload.get("user_id") or payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id


# ---------------------------------------------------------------------------
# App + routes
# ---------------------------------------------------------------------------

app = FastAPI(title=settings.app_name)
api_router = APIRouter(prefix=settings.api_prefix)


@app.get("/")
async def root() -> dict:
    return {"message": "PillowTales API is running"}


@api_router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@api_router.post("/auth/signup", response_model=AuthResponse)
async def signup(request: SignupRequest) -> AuthResponse:
    return auth_service.signup(request, repo)


@api_router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest) -> AuthResponse:
    return auth_service.login(request, repo)


@api_router.get("/languages")
async def get_supported_languages() -> dict:
    return {
        "languages": [{"code": code, "name": name} for code, name in SUPPORTED_LANGUAGES.items()],
        "voices": {
            code: [voice for voice in VOICE_PRESETS.values() if voice["language_code"] in {code, "all"}]
            for code in SUPPORTED_LANGUAGES.keys()
        },
    }


@api_router.get("/subscription")
async def get_subscription(user_id: str = Depends(get_current_user)) -> dict:
    profile = repo.get_user_profile(user_id) or {}
    subscription = subscription_service.get_subscription(user_id, profile.get("email"))
    tier_config = SUBSCRIPTION_TIERS["premium" if subscription.is_premium else "free"]
    return {
        "subscription": subscription.model_dump(),
        "features": {
            "narrators": tier_config["narrators"],
            "companions": tier_config["companions"],
            "parent_voice": tier_config["parent_voice"],
        },
    }


@api_router.get("/subscription/check-feature")
async def check_feature(feature: str, item_id: Optional[str] = None, user_id: str = Depends(get_current_user)) -> dict:
    profile = repo.get_user_profile(user_id) or {}
    subscription = subscription_service.get_subscription(user_id, profile.get("email"))
    return subscription_service.feature_allowed(subscription, feature, item_id)


@api_router.get("/voices")
async def get_voices(user_id: str = Depends(get_current_user)) -> dict:
    profile = repo.get_user_profile(user_id) or {}
    subscription = subscription_service.get_subscription(user_id, profile.get("email"))
    allowed = set(SUBSCRIPTION_TIERS["premium" if subscription.is_premium else "free"]["narrators"])
    voices = []
    for preset_id, preset in VOICE_PRESETS.items():
        voices.append(
            {
                "id": preset_id,
                "name": preset["name"],
                "description": preset["description"],
                "icon": preset["icon"],
                "provider": preset["provider"],
                "language_code": preset["language_code"],
                "is_locked": preset_id not in allowed,
                "requires_setup": preset.get("requires_setup", False),
            }
        )
    return {"narrators": voices, "default_narrator": "wise_owl"}


@api_router.get("/companions")
async def get_companions(user_id: str = Depends(get_current_user)) -> dict:
    profile = repo.get_user_profile(user_id) or {}
    subscription = subscription_service.get_subscription(user_id, profile.get("email"))
    allowed = set(SUBSCRIPTION_TIERS["premium" if subscription.is_premium else "free"]["companions"])
    companions = []
    for companion_id, companion in STORY_COMPANIONS.items():
        companions.append({
            "id": companion_id,
            "name": companion["name"],
            "short_name": companion["short_name"],
            "description": companion["description"],
            "icon": companion["icon"],
            "is_locked": companion_id not in allowed,
        })
    return {"companions": companions}


@api_router.get("/user/profile", response_model=UserProfileResponse)
async def get_user_profile(user_id: str = Depends(get_current_user)) -> UserProfileResponse:
    profile = repo.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    stories_this_week = repo.count_stories_since(user_id, week_ago)
    stories_saved = repo.count_stories(user_id)
    plan = profile.get("subscription_status") or profile.get("plan") or "free"
    tier = SUBSCRIPTION_TIERS["premium" if plan == "premium" else "free"]
    weekly_story_limit = tier["weekly_story_limit"]
    max_saved_stories = tier["max_saved_stories"]
    return UserProfileResponse(
        id=user_id,
        email=profile.get("email", ""),
        plan=plan,
        preferred_language=profile.get("preferred_language", "en"),
        stories_this_week=stories_this_week,
        stories_saved=stories_saved,
        can_generate=True if weekly_story_limit is None else stories_this_week < weekly_story_limit,
        can_save_more=True if max_saved_stories is None else stories_saved < max_saved_stories,
    )


@api_router.post("/generateStory", response_model=StoryResponse)
async def generate_story(request: GenerateStoryRequest, user_id: str = Depends(get_current_user)) -> StoryResponse:
    if request.userId != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized: user_id mismatch")

    profile = repo.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")

    subscription = subscription_service.get_subscription(user_id, profile.get("email"))
    tier = SUBSCRIPTION_TIERS["premium" if subscription.is_premium else "free"]
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    stories_this_week = repo.count_stories_since(user_id, week_ago)
    if tier["weekly_story_limit"] is not None and stories_this_week >= tier["weekly_story_limit"]:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "story_limit_reached",
                "message": "You've created 2 free stories this week. Upgrade to create unlimited bedtime stories.",
                "upgrade_required": True,
            },
        )

    stories_saved = repo.count_stories(user_id)
    if tier["max_saved_stories"] is not None and stories_saved >= tier["max_saved_stories"]:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "storage_limit",
                "message": "You've reached the maximum number of saved stories.",
                "upgrade_required": True,
            },
        )

    story_data = await story_service.generate_story(request, subscription)
    full_text = "\n\n".join(story_data["pages"])
    created_at = datetime.now(timezone.utc).isoformat()

    record = {
        "user_id": user_id,
        "title": story_data["title"],
        "child_name": request.childName,
        "age": request.age,
        "theme": request.customTheme or request.theme,
        "moral": request.moral,
        "calm_level": request.calmLevel,
        "duration_min": request.durationMin,
        "language": request.storyLanguageCode,
        "story_language_code": request.storyLanguageCode,
        "narration_language_code": request.narrationLanguageCode or request.storyLanguageCode,
        "pages": story_data["pages"],
        "full_text": full_text,
        "audio_url": None,
        "audio_status": "none",
        "is_favorite": False,
        "companion_id": story_data.get("companion", {}).get("id") if story_data.get("companion") else None,
        "companion_name": story_data.get("companion", {}).get("name") if story_data.get("companion") else None,
        "created_at": created_at,
    }
    saved_story = repo.insert_story(record)

    metadata = await story_service.extract_metadata(story_data["title"], full_text)
    metadata_update = {
        "story_summary": metadata.get("summary", ""),
        "characters": metadata.get("characters", []),
        "setting": metadata.get("setting", ""),
    }
    try:
        repo.update_story(saved_story["id"], user_id, metadata_update)
    except Exception as exc:
        logger.warning("Non-blocking metadata update failed for story %s: %s", saved_story["id"], exc)

    return StoryResponse(storyId=saved_story["id"], title=story_data["title"], pages=story_data["pages"])


@api_router.get("/stories")
async def list_stories(user_id: str = Depends(get_current_user)) -> dict:
    return {"stories": repo.list_stories(user_id)}


@api_router.get("/stories/{story_id}")
async def get_story(story_id: str, user_id: str = Depends(get_current_user)) -> dict:
    story = repo.get_story(story_id, user_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    return story


@api_router.put("/stories/{story_id}")
async def update_story(story_id: str, request: UpdateStoryRequest, user_id: str = Depends(get_current_user)) -> dict:
    values = {}
    if request.isFavorite is not None:
        values["is_favorite"] = request.isFavorite
    if not values:
        raise HTTPException(status_code=400, detail="No update data provided")
    story = repo.update_story(story_id, user_id, values)
    return {"message": "Story updated successfully", "story": story}


@api_router.delete("/stories/{story_id}")
async def delete_story(story_id: str, user_id: str = Depends(get_current_user)) -> dict:
    repo.delete_story(story_id, user_id)
    return {"message": "Story deleted successfully"}


@api_router.get("/story-preview/{story_id}")
async def story_preview(story_id: str) -> dict:
    story = repo.get_story(story_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")
    pages = story.get("pages") or []
    first_page = pages[0] if pages else ""
    preview = re.sub(r"\[(whisper|softly|chuckle|pause|gently)\]", "", first_page).strip()
    if len(preview) > 500:
        preview = preview[:497] + "..."
    return {
        "id": story.get("id"),
        "title": story.get("title"),
        "childName": story.get("child_name"),
        "firstParagraph": preview,
        "pageCount": len(pages),
        "duration": f"~{story.get('duration_min', 8)} min",
        "language": story.get("language", "en"),
        "createdAt": story.get("created_at"),
    }


@api_router.get("/user/settings")
async def get_user_settings(user_id: str = Depends(get_current_user)) -> dict:
    profile = repo.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "preferred_language": profile.get("preferred_language", "en"),
        "bedtime_mode": profile.get("bedtime_mode", False),
        "plan": profile.get("subscription_status") or profile.get("plan") or "free",
    }


@api_router.put("/user/settings")
async def update_user_settings(
    preferred_language: Optional[str] = None,
    bedtime_mode: Optional[bool] = None,
    user_id: str = Depends(get_current_user),
) -> dict:
    updates: Dict[str, Any] = {}
    if preferred_language is not None:
        if preferred_language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail="Unsupported language")
        updates["preferred_language"] = preferred_language
    if bedtime_mode is not None:
        updates["bedtime_mode"] = bedtime_mode
    if not updates:
        raise HTTPException(status_code=400, detail="No update data provided")
    settings_row = repo.update_user_profile(user_id, updates)
    return {"message": "Settings updated successfully", "settings": settings_row}


@api_router.post("/narration/request", response_model=NarrationResponse)
async def request_narration(request: NarrationRequest, user_id: str = Depends(get_current_user)) -> NarrationResponse:
    story = repo.get_story(request.storyId, user_id)
    if not story:
        raise HTTPException(status_code=404, detail="Story not found")

    profile = repo.get_user_profile(user_id) or {}
    subscription = subscription_service.get_subscription(user_id, profile.get("email"))
    narration_access = subscription_service.feature_allowed(subscription, "narration")
    if not narration_access["allowed"]:
        raise HTTPException(status_code=403, detail=narration_access)

    requested_voice = request.voicePreference or "wise_owl"
    voice_access = subscription_service.feature_allowed(subscription, "narrator", requested_voice)
    if not voice_access["allowed"]:
        raise HTTPException(status_code=403, detail=voice_access)

    # Transitional Phase C behavior: this endpoint is safe and stable, but the
    # async TTS workers should be implemented in a dedicated narration service.
    return NarrationResponse(
        status="pending",
        audioUrl=story.get("audio_url"),
        message="Narration worker not included in this clean rebuild yet. Wire this endpoint to your queue/worker next.",
    )


@api_router.get("/download/{filename}")
async def download_file(filename: str) -> FileResponse:
    file_path = ROOT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    allowed_extensions = {".zip", ".png", ".jpg", ".jpeg", ".pdf"}
    if file_path.suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=403, detail="File type not allowed")
    return FileResponse(path=str(file_path), filename=filename, media_type="application/octet-stream")


app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    logger.info("Starting %s", settings.app_name)


@app.on_event("shutdown")
async def shutdown() -> None:
    logger.info("Stopping %s", settings.app_name)
