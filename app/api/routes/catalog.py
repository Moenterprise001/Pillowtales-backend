from __future__ import annotations

from fastapi import APIRouter

from app.domain.constants import STORY_COMPANIONS, SUPPORTED_LANGUAGES, VOICE_PRESETS

router = APIRouter(tags=['catalog'])


@router.get('/languages')
async def get_supported_languages() -> dict:
    return {
        'languages': [{'code': code, 'name': name} for code, name in SUPPORTED_LANGUAGES.items()],
        'voices': {code: next((v['name'] for v in VOICE_PRESETS.values() if v.get('language_code') == code), None) for code in SUPPORTED_LANGUAGES},
    }


@router.get('/voices')
async def get_voices() -> dict:
    return {'narrators': [{'id': key, **value} for key, value in VOICE_PRESETS.items()]}


@router.get('/companions')
async def get_companions() -> dict:
    return {'companions': [{'id': key, **value} for key, value in STORY_COMPANIONS.items()]}
