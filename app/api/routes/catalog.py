from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user, get_subscription_service, get_user_repo
from app.domain.constants import STORY_COMPANIONS, SUPPORTED_LANGUAGES, VOICE_PRESETS
from app.repositories.user_repository import UserRepository
from app.services.subscription_service import SubscriptionService

router = APIRouter(tags=['catalog'])


@router.get('/languages')
async def get_supported_languages() -> dict:
    return {
        'languages': [{'code': code, 'name': name} for code, name in SUPPORTED_LANGUAGES.items()],
        'voices': {code: next((v['name'] for v in VOICE_PRESETS.values() if v.get('language_code') == code), None) for code in SUPPORTED_LANGUAGES},
    }


@router.get('/voices')
async def get_voices(user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo), subscription_service: SubscriptionService = Depends(get_subscription_service)) -> dict:
    profile = user_repo.get_profile(user_id) or {}
    subscription = subscription_service.get_subscription(user_id, profile.get('email'))
    parent_voice_state = subscription_service.get_parent_voice_state(user_id)
    narrators = []
    for key, value in VOICE_PRESETS.items():
        item = {'id': key, **value}
        item['is_premium'] = bool(value.get('tier') == 'premium')
        item['requires_setup'] = bool(value.get('requires_setup', False))
        item['personality'] = value.get('description', '')
        item['is_ready'] = True
        if key == 'parent_voice':
            item['is_premium'] = False
            item['credits_remaining'] = parent_voice_state['credits']
            item['purchase_required_each_story'] = True
            item['price_eur'] = value.get('price_eur', 2.0)
            item['is_ready'] = bool(profile.get('parent_voice_id')) and profile.get('parent_voice_status', 'none') == 'ready'
        narrators.append(item)
    return {
        'narrators': narrators,
        'default_narrator': 'wise_owl',
        'has_parent_voice': bool(profile.get('parent_voice_id')) and profile.get('parent_voice_status', 'none') == 'ready',
        'parent_voice_story_credits': parent_voice_state['credits'],
        'parent_voice_story_price_eur': VOICE_PRESETS['parent_voice'].get('price_eur', 2.0),
        'parent_voice_story_bundle3_price_eur': VOICE_PRESETS['parent_voice'].get('bundle3_price_eur', 4.99),
        'parent_voice_intro_available': parent_voice_state['intro_available'],
    }


@router.get('/companions')
async def get_companions() -> dict:
    return {'companions': [{'id': key, **value} for key, value in STORY_COMPANIONS.items()]}
