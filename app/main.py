from __future__ import annotations

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.routes import auth, catalog, narration, stories, subscription, system, user
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(title=settings.app_name)
app.add_middleware(CORSMiddleware, allow_origins=settings.allow_origins, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.include_router(system.router, prefix=settings.api_prefix)
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(catalog.router, prefix=settings.api_prefix)
app.include_router(subscription.router, prefix=settings.api_prefix)
app.include_router(user.router, prefix=settings.api_prefix)
app.include_router(stories.router, prefix=settings.api_prefix)
app.include_router(narration.router, prefix=settings.api_prefix)


@app.on_event('startup')
async def startup() -> None:
    return None


@app.on_event('shutdown')
async def shutdown() -> None:
    return None
