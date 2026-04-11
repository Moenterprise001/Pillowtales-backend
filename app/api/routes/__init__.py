from fastapi import APIRouter

from app.api.routes import auth
from app.api.routes import catalog
from app.api.routes import narration
from app.api.routes import subscription
from app.api.routes import user
from app.api.routes import parent_voice

api_router = APIRouter()

api_router.include_router(auth.router)
api_router.include_router(catalog.router)
api_router.include_router(narration.router)
api_router.include_router(subscription.router)
api_router.include_router(user.router)
api_router.include_router(parent_voice.router)from app.api.routes import parent_voice
api_router.include_router(parent_voice.router)