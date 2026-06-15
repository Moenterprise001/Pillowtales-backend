from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user, get_story_repo, get_user_repo
from app.domain.constants import SUBSCRIPTION_TIERS, SUPPORTED_LANGUAGES, VOICE_PRESETS
from app.models.story import UserProfileResponse
from app.models.subscription import ParentVoiceCreditsRedeemRequest, ParentVoiceCreditsResponse
from app.repositories.story_repository import StoryRepository
from app.repositories.user_repository import UserRepository

router = APIRouter(prefix='/user', tags=['user'])
logger = logging.getLogger(__name__)

_PARENT_VOICE_WALLET_PATH = Path('/tmp/pillowtales_parent_voice_wallets.json')
_PARENT_VOICE_PROFILE_META_PATH = Path('/tmp/pillowtales_parent_voice_profiles.json')


def _remove_json_key(path: Path, key: str) -> None:
    """Best-effort cleanup for temporary local stores used by launch-era services."""
    try:
        if not path.exists():
            return
        data = json.loads(path.read_text() or '{}')
        if key in data:
            del data[key]
            path.write_text(json.dumps(data))
    except Exception as exc:
        logger.warning('[ACCOUNT_DELETE] Could not clean local metadata file %s: %s', path, exc)


def _collect_storage_paths(storage_bucket, prefix: str) -> list[str]:
    """Recursively collect storage object paths under a prefix.

    Supabase storage list() returns folder-like entries for nested paths and file
    entries for actual objects. This helper is intentionally best-effort so account
    deletion still succeeds even if storage cleanup encounters a transient issue.
    """
    collected: list[str] = []

    try:
        entries = storage_bucket.list(prefix) or []
    except Exception as exc:
        logger.warning('[ACCOUNT_DELETE] Could not list storage prefix %s: %s', prefix, exc)
        return collected

    for entry in entries:
        name = entry.get('name') if isinstance(entry, dict) else None
        if not name:
            continue

        child_path = f'{prefix.rstrip("/")}/{name}' if prefix else name

        # Supabase folder entries usually have no id/metadata fields. Try listing
        # children first; if it has children, treat it as a folder.
        child_entries = []
        try:
            child_entries = storage_bucket.list(child_path) or []
        except Exception:
            child_entries = []

        if child_entries:
            collected.extend(_collect_storage_paths(storage_bucket, child_path))
        else:
            collected.append(child_path)

    return collected


def _delete_storage_prefix(user_repo: UserRepository, prefix: str) -> int:
    """Best-effort delete of storage objects under a prefix in story-audio."""
    try:
        bucket = user_repo.client.storage.from_('story-audio')
        paths = _collect_storage_paths(bucket, prefix)
        if paths:
            bucket.remove(paths)
        return len(paths)
    except Exception as exc:
        logger.warning('[ACCOUNT_DELETE] Could not delete storage prefix %s: %s', prefix, exc)
        return 0


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


@router.delete('/account')
async def delete_user_account(user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo)) -> dict:
    """Delete the authenticated user's PillowTales account and related data.

    Frontend Settings calls DELETE /api/user/account. This route must exist for
    App Store account-deletion compliance and for live Android users.
    """
    deleted_storage_objects = 0

    # Best-effort storage cleanup. Story audio is stored under:
    # story-audio/{user_id}/{story_id}/...
    # Parent Voice launch samples were stored under:
    # story-audio/parent-voice-samples/{user_id}/...
    deleted_storage_objects += _delete_storage_prefix(user_repo, user_id)
    deleted_storage_objects += _delete_storage_prefix(user_repo, f'parent-voice-samples/{user_id}')

    # Best-effort database cleanup before deleting the auth user.
    try:
        user_repo.client.table('stories').delete().eq('user_id', user_id).execute()
    except Exception as exc:
        logger.warning('[ACCOUNT_DELETE] Could not delete stories for user %s: %s', user_id, exc)

    try:
        user_repo.client.table('users_profile').delete().eq('id', user_id).execute()
    except Exception as exc:
        logger.warning('[ACCOUNT_DELETE] Could not delete profile for user %s: %s', user_id, exc)

    # Best-effort cleanup of temporary launch-era wallet/profile metadata.
    _remove_json_key(_PARENT_VOICE_WALLET_PATH, user_id)
    _remove_json_key(_PARENT_VOICE_PROFILE_META_PATH, user_id)

    # Delete the Supabase Auth user last. The service-role Supabase client is used
    # by the backend, so this does not rely on the user's token remaining valid
    # after profile/story cleanup.
    try:
        user_repo.client.auth.admin.delete_user(user_id)
    except Exception as exc:
        logger.warning('[ACCOUNT_DELETE] Could not delete auth user %s: %s', user_id, exc)
        raise HTTPException(status_code=500, detail='Account data was removed, but authentication account deletion failed. Please contact support.') from exc

    return {
        'status': 'deleted',
        'message': 'Account deleted successfully.',
        'deleted_storage_objects': deleted_storage_objects,
    }


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
