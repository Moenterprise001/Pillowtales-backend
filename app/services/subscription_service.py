from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from app.domain.constants import STORY_COMPANIONS, SUBSCRIPTION_TIERS, PREMIUM_TESTER_EMAILS, QA_PARENT_VOICE_BYPASS_EMAILS, VOICE_PRESETS
from app.models.subscription import SubscriptionResponse
from app.repositories.story_repository import StoryRepository
from app.repositories.user_repository import UserRepository


class SubscriptionService:
    def __init__(self, user_repo: UserRepository, story_repo: StoryRepository):
        self.user_repo = user_repo
        self.story_repo = story_repo

    def get_parent_voice_state(self, user_id: str) -> dict:
        wallet = self.user_repo.get_parent_voice_wallet(user_id)
        return {
            'credits': int(wallet.get('credits', 0)),
            'intro_available': not bool(wallet.get('intro_used', False)),
            'profile': wallet.get('profile') or {},
        }

    def get_subscription(self, user_id: str, user_email: Optional[str]) -> SubscriptionResponse:
        normalized_email = (user_email or '').strip().lower()
        is_tester = bool(normalized_email and normalized_email in PREMIUM_TESTER_EMAILS)
        parent_voice_bypass = bool(normalized_email and normalized_email in QA_PARENT_VOICE_BYPASS_EMAILS)

        profile = self.user_repo.get_profile(user_id) or {}

        plan_value = (profile.get('plan') or '').strip().lower()
        subscription_value = (profile.get('subscription_status') or '').strip().lower()

        is_premium = (
            is_tester
            or plan_value == 'premium'
            or subscription_value == 'premium'
    	)

    	week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        narration_count = self.story_repo.count_narrations_since(user_id, week_ago)
        weekly_limit = None if is_premium else SUBSCRIPTION_TIERS['free']['weekly_narration_limit']
        can_narrate = is_premium or narration_count < (weekly_limit or 0)
        remaining = None if weekly_limit is None else max(0, weekly_limit - narration_count)
        parent_voice_state = self.get_parent_voice_state(user_id)
        return SubscriptionResponse(
            status='premium' if is_premium else 'free',
            is_premium=is_premium,
            is_tester=is_tester,
            weekly_narrations_used=narration_count,
            weekly_limit=weekly_limit,
            can_narrate=can_narrate,
            narrations_remaining=remaining,
            parent_voice_credits=parent_voice_state['credits'],
            parent_voice_intro_available=parent_voice_state['intro_available'],
            parent_voice_bypass=parent_voice_bypass,
        )

    def get_tier_payload(self, subscription: SubscriptionResponse) -> dict:
        tier_key = 'premium' if subscription.is_premium else 'free'
        tier = SUBSCRIPTION_TIERS[tier_key]
        return {
            'tier': tier_key,
            'features': {
                'weekly_story_limit': tier['weekly_story_limit'],
                'weekly_narration_limit': tier['weekly_narration_limit'],
                'max_saved_stories': tier['max_saved_stories'],
                'parent_voice': True,
            },
            'available_narrators': [{'id': n, **VOICE_PRESETS[n]} for n in tier['narrators'] if n in VOICE_PRESETS],
            'available_companions': [{'id': c, **STORY_COMPANIONS[c]} for c in tier['companions'] if c in STORY_COMPANIONS],
            'parent_voice_credits': subscription.parent_voice_credits,
            'parent_voice_intro_available': subscription.parent_voice_intro_available,
        }

    def feature_allowed(self, subscription: SubscriptionResponse, feature: str, item_id: Optional[str] = None) -> dict:
        tier_key = 'premium' if subscription.is_premium else 'free'
        tier = SUBSCRIPTION_TIERS[tier_key]
        if feature == 'narrator' and item_id and item_id not in tier['narrators']:
            if item_id == 'parent_voice':
                return {
                    'allowed': subscription.parent_voice_bypass or subscription.parent_voice_credits > 0 or subscription.parent_voice_intro_available,
                    'upgrade_required': not (subscription.parent_voice_bypass or subscription.parent_voice_credits > 0 or subscription.parent_voice_intro_available),
                    'reason': None if (subscription.parent_voice_bypass or subscription.parent_voice_credits > 0 or subscription.parent_voice_intro_available) else 'parent_voice_purchase_required',
                    'credits': subscription.parent_voice_credits,
                    'intro_offer_available': subscription.parent_voice_intro_available,
                }
            return {'allowed': False, 'upgrade_required': True, 'reason': 'premium_narrator'}
        if feature == 'companion' and item_id and item_id not in tier['companions']:
            return {'allowed': False, 'upgrade_required': True, 'reason': 'premium_companion'}
        if feature == 'parent_voice':
            # Parent Voice story generation must require an actual credit/bypass.
            # The free intro remains visible via parent_voice_intro_available, but it
            # must be explicitly redeemed first so parents do not consume it by accident.
            allowed = subscription.parent_voice_bypass or subscription.parent_voice_credits > 0
            return {
                'allowed': allowed,
                'upgrade_required': not allowed,
                'reason': None if allowed else 'parent_voice_credit_required',
                'credits': subscription.parent_voice_credits,
                'intro_offer_available': subscription.parent_voice_intro_available,
            }
        if feature == 'narration' and not subscription.can_narrate:
            return {'allowed': False, 'upgrade_required': True, 'reason': 'weekly_narration_limit', 'used': subscription.weekly_narrations_used, 'limit': subscription.weekly_limit}
        return {'allowed': True, 'upgrade_required': False, 'reason': None}
