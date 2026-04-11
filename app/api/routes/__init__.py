from fastapi import APIRouter

from app.api.routes import auth, catalog, narration, stories, subscription, system, user, parent_voice

api_router = APIRouter()

api_router.include_router(auth.router)
api_router.include_router(catalog.router)
api_router.include_router(narration.router)
api_router.include_router(stories.router)
api_router.include_router(subscription.router)
api_router.include_router(system.router)
api_router.include_router(user.router)
api_router.include_router(parent_voice.router)