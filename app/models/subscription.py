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
