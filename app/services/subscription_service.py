from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from app.domain.constants import STORY_COMPANIONS, SUBSCRIPTION_TIERS, TESTER_EMAILS, VOICE_PRESETS
from app.models.subscription import SubscriptionResponse
from app.repositories.story_repository import StoryRepository
from app.repositories.user_repository import UserRepository


class SubscriptionService:
    def __init__(self, user_repo: UserRepository, story_repo: StoryRepository):
        self.user_repo = user_repo
        self.story_repo = story_repo

    def get_subscription(self, user_id: str, user_email: Optional[str]) -> SubscriptionResponse:
        is_tester = bool(user_email and user_email.lower() in TESTER_EMAILS)
        profile = self.user_repo.get_profile(user_id) or {}
        status_value = profile.get('subscription_status') or profile.get('plan') or 'free'
        is_premium = is_tester or status_value == 'premium'
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        narration_count = self.story_repo.count_narrations_since(user_id, week_ago)
        weekly_limit = None if is_premium else SUBSCRIPTION_TIERS['free']['weekly_narration_limit']
        can_narrate = is_premium or narration_count < (weekly_limit or 0)
        remaining = None if weekly_limit is None else max(0, weekly_limit - narration_count)
        return SubscriptionResponse(status='premium' if is_premium else 'free', is_premium=is_premium, is_tester=is_tester, weekly_narrations_used=narration_count, weekly_limit=weekly_limit, can_narrate=can_narrate, narrations_remaining=remaining)

    def get_tier_payload(self, subscription: SubscriptionResponse) -> dict:
        tier_key = 'premium' if subscription.is_premium else 'free'
        tier = SUBSCRIPTION_TIERS[tier_key]
        return {
            'tier': tier_key,
            'features': {
                'weekly_story_limit': tier['weekly_story_limit'],
                'weekly_narration_limit': tier['weekly_narration_limit'],
                'max_saved_stories': tier['max_saved_stories'],
                'parent_voice': tier['parent_voice'],
            },
            'available_narrators': [{'id': n, **VOICE_PRESETS[n]} for n in tier['narrators'] if n in VOICE_PRESETS],
            'available_companions': [{'id': c, **STORY_COMPANIONS[c]} for c in tier['companions'] if c in STORY_COMPANIONS],
        }

    def feature_allowed(self, subscription: SubscriptionResponse, feature: str, item_id: Optional[str] = None) -> dict:
        tier_key = 'premium' if subscription.is_premium else 'free'
        tier = SUBSCRIPTION_TIERS[tier_key]
        if feature == 'narrator' and item_id and item_id not in tier['narrators']:
            return {'allowed': False, 'upgrade_required': True, 'reason': 'premium_narrator'}
        if feature == 'companion' and item_id and item_id not in tier['companions']:
            return {'allowed': False, 'upgrade_required': True, 'reason': 'premium_companion'}
        if feature == 'parent_voice' and not tier['parent_voice']:
            return {'allowed': False, 'upgrade_required': True, 'reason': 'parent_voice_premium'}
        if feature == 'narration' and not subscription.can_narrate:
            return {'allowed': False, 'upgrade_required': True, 'reason': 'weekly_narration_limit', 'used': subscription.weekly_narrations_used, 'limit': subscription.weekly_limit}
        return {'allowed': True, 'upgrade_required': False, 'reason': None}
