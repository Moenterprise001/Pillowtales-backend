from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.api.deps import get_user_repo
from app.repositories.user_repository import UserRepository

router = APIRouter(prefix="/revenuecat", tags=["revenuecat"])

_PROCESSED_EVENTS_PATH = Path("/tmp/pillowtales_revenuecat_processed_events.json")

YEARLY_PRODUCTS = {
    "com.pillowtales.yearly",
    "premium_yearly:yearly",
}

PARENT_VOICE_CREDIT_PRODUCTS = {
    "parent_voice_1": 1,
    "parent_voice_3": 3,
}


def _read_processed_events() -> set[str]:
    try:
        if _PROCESSED_EVENTS_PATH.exists():
            return set(json.loads(_PROCESSED_EVENTS_PATH.read_text() or "[]"))
    except Exception:
        pass
    return set()


def _write_processed_events(events: set[str]) -> None:
    _PROCESSED_EVENTS_PATH.write_text(json.dumps(sorted(events)))


def _event_id(event: dict) -> str:
    return (
        str(event.get("id"))
        or str(event.get("transaction_id"))
        or str(event.get("original_transaction_id"))
        or str(event.get("event_timestamp_ms"))
    )


def _get_product_id(event: dict) -> str:
    return (
        event.get("product_id")
        or event.get("store_product_id")
        or event.get("new_product_id")
        or ""
    )


def _get_user_id(event: dict) -> str:
    return (
        event.get("app_user_id")
        or event.get("original_app_user_id")
        or ""
    )


@router.post("/webhook")
async def revenuecat_webhook(
    request: Request,
    authorization: str | None = Header(default=None),
    user_repo: UserRepository = Depends(get_user_repo),
) -> dict:
    webhook_secret = os.getenv("REVENUECAT_WEBHOOK_SECRET", "").strip()

    if webhook_secret:
        raw_secret = webhook_secret.strip()
        auth_value = (authorization or "").strip()

        valid_authorization_values = {
            raw_secret,
            f"Bearer {raw_secret}",
        }

        if auth_value not in valid_authorization_values:
            raise HTTPException(status_code=401, detail="Invalid RevenueCat webhook secret")

    payload = await request.json()
    event = payload.get("event") or payload

    event_id = _event_id(event)
    product_id = _get_product_id(event)
    user_id = _get_user_id(event)

    if not event_id:
        raise HTTPException(status_code=400, detail="Missing RevenueCat event id")

    processed = _read_processed_events()
    if event_id in processed:
        return {"status": "ignored", "reason": "already_processed"}

    if not user_id:
        raise HTTPException(status_code=400, detail="Missing RevenueCat app_user_id")

    credits_to_add = 0

    if product_id in YEARLY_PRODUCTS:
        credits_to_add = 3

    if product_id in PARENT_VOICE_CREDIT_PRODUCTS:
        credits_to_add = PARENT_VOICE_CREDIT_PRODUCTS[product_id]

    if credits_to_add <= 0:
        processed.add(event_id)
        _write_processed_events(processed)
        return {
            "status": "ignored",
            "reason": "no_credit_grant_required",
            "product_id": product_id,
        }

    wallet = user_repo.get_parent_voice_wallet(user_id)
    current_credits = int(wallet.get("credits", 0))
    intro_used = bool(wallet.get("intro_used", False))

    new_credits = current_credits + credits_to_add

    user_repo.save_parent_voice_wallet(
        user_id,
        credits=new_credits,
        intro_used=intro_used,
    )

    processed.add(event_id)
    _write_processed_events(processed)

    return {
        "status": "ok",
        "user_id": user_id,
        "product_id": product_id,
        "credits_added": credits_to_add,
        "new_credit_balance": new_credits,
    }