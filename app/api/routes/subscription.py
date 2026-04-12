from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user, get_subscription_service, get_user_repo
from app.repositories.user_repository import UserRepository
from app.services.subscription_service import SubscriptionService

router = APIRouter(prefix='/subscription', tags=['subscription'])


@router.get('')
async def get_subscription(user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo), subscription_service: SubscriptionService = Depends(get_subscription_service)) -> dict:
    profile = user_repo.get_profile(user_id) or {}
    subscription = subscription_service.get_subscription(user_id, profile.get('email'))
    payload = subscription_service.get_tier_payload(subscription)
    return {'subscription': subscription.model_dump(), **payload}


@router.get('/check-feature')
async def check_feature(feature: str, item_id: str | None = None, user_id: str = Depends(get_current_user), user_repo: UserRepository = Depends(get_user_repo), subscription_service: SubscriptionService = Depends(get_subscription_service)) -> dict:
    profile = user_repo.get_profile(user_id) or {}
    subscription = subscription_service.get_subscription(user_id, profile.get('email'))
    return subscription_service.feature_allowed(subscription, feature, item_id)
