from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user, get_story_repo, get_user_repo
from app.domain.constants import SUBSCRIPTION_TIERS, SUPPORTED_LANGUAGES, VOICE_PRESETS
from app.models.story import UserProfileResponse
from app.models.subscription import ParentVoiceCreditsRedeemRequest, ParentVoiceCreditsResponse
from app.repositories.story_repository import StoryRepository
from app.repositories.user_repository import UserRepository

router = APIRouter(prefix='/user', tags=['user'])


@router.get('/profile', response_model=UserProfileResponse)
async def get_user_profile(user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo), story_repo: StoryRepository = Depends(get_story_repo)) -> UserProfileResponse:
    profile = user_repo.get_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail='User profile not found')
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    stories_this_week = story_repo.count_since(user_id, week_ago)
    stories_saved = story_repo.count_all(user_id)
    plan = profile.get('subscription_status') or profile.get('plan') or 'free'
    tier = SUBSCRIPTION_TIERS['premium' if plan == 'premium' else 'free']
    weekly_story_limit = tier['weekly_story_limit']
    max_saved_stories = tier['max_saved_stories']
    return UserProfileResponse(id=user_id, email=profile.get('email', ''), plan=plan, preferred_language=profile.get('preferred_language', 'en'), stories_this_week=stories_this_week, stories_saved=stories_saved, can_generate=True if weekly_story_limit is None else stories_this_week < weekly_story_limit, can_save_more=True if max_saved_stories is None else stories_saved < max_saved_stories)


@router.get('/settings')
async def get_user_settings(user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo)) -> dict:
    profile = user_repo.get_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail='User not found')
    return {'preferred_language': profile.get('preferred_language', 'en'), 'bedtime_mode': profile.get('bedtime_mode', False), 'plan': profile.get('subscription_status') or profile.get('plan') or 'free'}


@router.put('/settings')
async def update_user_settings(preferred_language: Optional[str] = None, bedtime_mode: Optional[bool] = None, user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo)) -> dict:
    updates: Dict[str, Any] = {}
    if preferred_language is not None:
        if preferred_language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail='Unsupported language')
        updates['preferred_language'] = preferred_language
    if bedtime_mode is not None:
        updates['bedtime_mode'] = bedtime_mode
    if not updates:
        raise HTTPException(status_code=400, detail='No update data provided')
    settings_row = user_repo.update_profile(user_id, updates)
    return {'message': 'Settings updated successfully', 'settings': settings_row}


@router.get('/parent-voice-credits', response_model=ParentVoiceCreditsResponse)
async def get_parent_voice_credits(user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo)) -> ParentVoiceCreditsResponse:
    wallet = user_repo.get_parent_voice_wallet(user_id)
    return ParentVoiceCreditsResponse(
        credits=int(wallet.get('credits', 0)),
        price_eur=float(VOICE_PRESETS['parent_voice'].get('price_eur', 2.0)),
        currency='EUR',
        intro_offer_available=not bool(wallet.get('intro_used', False)),
        offers=[
            {'quantity': 1, 'price_eur': float(VOICE_PRESETS['parent_voice'].get('price_eur', 2.0)), 'label': '1 story'},
            {'quantity': 3, 'price_eur': float(VOICE_PRESETS['parent_voice'].get('bundle3_price_eur', 4.99)), 'label': '3 stories', 'best_value': True},
        ],
    )


@router.post('/parent-voice-credits/redeem', response_model=ParentVoiceCreditsResponse)
async def redeem_parent_voice_credits(payload: ParentVoiceCreditsRedeemRequest, user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo)) -> ParentVoiceCreditsResponse:
    wallet = user_repo.get_parent_voice_wallet(user_id)
    credits = int(wallet.get('credits', 0))
    intro_used = bool(wallet.get('intro_used', False))

    quantity = max(1, int(payload.quantity or 1))
    source = (payload.source or 'revenuecat_client').strip().lower()

    if source == 'parent_voice_intro_offer':
        if intro_used:
            raise HTTPException(status_code=409, detail={'error': 'intro_offer_already_used', 'message': 'The free Parent Voice intro offer has already been used.'})
        intro_used = True
        credits = max(credits, 0)
        saved = user_repo.save_parent_voice_wallet(user_id, credits=credits, intro_used=intro_used)
        message = 'Free Parent Voice story unlocked.'
    elif source == 'revenuecat_client':
        if quantity not in {1, 3}:
            raise HTTPException(status_code=400, detail={'error': 'invalid_credit_quantity', 'message': 'Only 1 or 3 story bundles are supported.'})

        # RevenueCat webhooks are the source of truth for paid Parent Voice credits.
        # This endpoint is retained as a client acknowledgement/refresh endpoint only.
        # Do not add credits here, otherwise purchases can be counted twice:
        # once by /revenuecat/webhook and once by this client-side redeem call.
        saved = wallet
        message = 'Parent Voice credit purchase acknowledged. Balance will refresh from RevenueCat webhook.'
    else:
        raise HTTPException(status_code=400, detail={'error': 'invalid_credit_source', 'message': 'Unsupported credit redemption source.'})

    return ParentVoiceCreditsResponse(
        credits=int(saved.get('credits', 0)),
        price_eur=float(VOICE_PRESETS['parent_voice'].get('price_eur', 2.0)),
        currency='EUR',
        source=source,
        message=message,
        intro_offer_available=not bool(saved.get('intro_used', False)),
        offers=[
            {'quantity': 1, 'price_eur': float(VOICE_PRESETS['parent_voice'].get('price_eur', 2.0)), 'label': '1 story'},
            {'quantity': 3, 'price_eur': float(VOICE_PRESETS['parent_voice'].get('bundle3_price_eur', 4.99)), 'label': '3 stories', 'best_value': True},
        ],
    )
