from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_current_user, get_subscription_service, get_user_repo
from app.repositories.user_repository import UserRepository
from app.services.subscription_service import SubscriptionService

router = APIRouter(prefix='/subscription', tags=['subscription'])


MONTHLY_PRODUCTS = {
    'com.pillowtales.monthly',
    'premium_monthly',
    'premium_monthly:monthly',
}

YEARLY_PRODUCTS = {
    'com.pillowtales.yearly',
    'premium_yearly',
    'premium_yearly:yearly',
}

YEARLY_PACKAGE_IDENTIFIERS = {
    '$rc_annual',
    '$rc_yearly',
    'annual',
    'yearly',
}


class SubscriptionSyncRequest(BaseModel):
    product_id: str | None = None
    package_identifier: str | None = None
    package_type: str | None = None
    source: str | None = 'revenuecat_client'


def _is_premium_profile(profile: dict) -> bool:
    plan = (profile.get('plan') or '').strip().lower()
    subscription_status = (profile.get('subscription_status') or '').strip().lower()
    return plan == 'premium' or subscription_status == 'premium'


def _is_yearly_sync(request: SubscriptionSyncRequest) -> bool:
    product_id = (request.product_id or '').strip().lower()
    package_identifier = (request.package_identifier or '').strip().lower()
    package_type = (request.package_type or '').strip().upper()

    return (
        product_id in YEARLY_PRODUCTS
        or package_identifier in YEARLY_PACKAGE_IDENTIFIERS
        or 'year' in product_id
        or 'annual' in package_identifier
        or 'yearly' in package_identifier
        or package_type == 'ANNUAL'
    )


def _is_subscription_sync(request: SubscriptionSyncRequest) -> bool:
    product_id = (request.product_id or '').strip().lower()
    package_identifier = (request.package_identifier or '').strip().lower()
    package_type = (request.package_type or '').strip().upper()

    source = (request.source or '').strip().lower()
    if not product_id and not package_identifier and source:
        return source in {'revenuecat_client', 'revenuecat_restore', 'revenuecat_native_paywall'}

    return (
        product_id in MONTHLY_PRODUCTS
        or product_id in YEARLY_PRODUCTS
        or package_identifier in YEARLY_PACKAGE_IDENTIFIERS
        or package_identifier in {'$rc_monthly', 'monthly'}
        or 'monthly' in product_id
        or 'yearly' in product_id
        or 'annual' in package_identifier
        or 'yearly' in package_identifier
        or package_type in {'MONTHLY', 'ANNUAL'}
    )


def _subscription_payload(
    user_id: str,
    user_repo: UserRepository,
    subscription_service: SubscriptionService,
) -> dict:
    profile = user_repo.get_profile(user_id) or {}
    subscription = subscription_service.get_subscription(user_id, profile.get('email'))
    payload = subscription_service.get_tier_payload(subscription)
    return {'subscription': subscription.model_dump(), **payload}


@router.get('')
async def get_subscription(
    user_id: str = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repo),
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> dict:
    return _subscription_payload(user_id, user_repo, subscription_service)


@router.post('/sync')
async def sync_subscription(
    request: SubscriptionSyncRequest,
    user_id: str = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repo),
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> dict:
    """Immediately mirror a successful RevenueCat client purchase into the backend profile.

    The webhook remains the source of truth/backstop, but it can arrive several seconds
    after the purchase. This endpoint is called only after RevenueCat reports a
    successful purchase/restore in the native app so the parent can create stories
    immediately without logging out and back in.
    """
    if not _is_subscription_sync(request):
        return {
            'status': 'ignored',
            'reason': 'not_a_subscription_product',
            **_subscription_payload(user_id, user_repo, subscription_service),
        }

    profile_before = user_repo.get_profile(user_id) or {}
    was_premium = _is_premium_profile(profile_before)

    user_repo.update_profile(user_id, {
        'plan': 'premium',
        'subscription_status': 'premium',
    })

    credits_added = 0
    if _is_yearly_sync(request) and not was_premium:
        wallet = user_repo.get_parent_voice_wallet(user_id)
        current_credits = int(wallet.get('credits', 0))
        credits_added = 3

        # Product rule:
        # The free Parent Voice intro offer is for users who have not upgraded.
        # A Yearly subscription includes 3 Parent Voice credits, so any unused
        # intro offer should be retired when those included credits are granted.
        user_repo.save_parent_voice_wallet(
            user_id,
            credits=current_credits + credits_added,
            intro_used=True,
        )

    return {
        'status': 'ok',
        'premium_synced': True,
        'credits_added': credits_added,
        **_subscription_payload(user_id, user_repo, subscription_service),
    }


@router.get('/check-feature')
async def check_feature(
    feature: str,
    item_id: str | None = None,
    user_id: str = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repo),
    subscription_service: SubscriptionService = Depends(get_subscription_service),
) -> dict:
    profile = user_repo.get_profile(user_id) or {}
    subscription = subscription_service.get_subscription(user_id, profile.get('email'))
    return subscription_service.feature_allowed(subscription, feature, item_id)
