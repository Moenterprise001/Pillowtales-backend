from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class SubscriptionResponse(BaseModel):
    status: str
    is_premium: bool
    is_tester: bool = False
    weekly_narrations_used: int = 0
    weekly_limit: Optional[int] = None
    can_narrate: bool = True
    narrations_remaining: Optional[int] = None
    parent_voice_credits: int = 0
    parent_voice_intro_available: bool = False



class ParentVoiceCreditsRedeemRequest(BaseModel):
    quantity: int = 1
    source: str = 'revenuecat_client'


class ParentVoiceCreditsResponse(BaseModel):
    credits: int
    price_eur: float = 2.0
    currency: str = 'EUR'
    source: Optional[str] = None
    message: Optional[str] = None
    intro_offer_available: bool = False
    offers: list[dict] = []
