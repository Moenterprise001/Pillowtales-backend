from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.api.deps import get_user_repo
from app.repositories.user_repository import UserRepository

router = APIRouter(prefix="/revenuecat", tags=["revenuecat"])

_PROCESSED_EVENTS_PATH = Path("/tmp/pillowtales_revenuecat_processed_events.json")

MONTHLY_PRODUCTS = {
    "com.pillowtales.monthly",
    "premium_monthly",
    "premium_monthly:monthly",
}

YEARLY_PRODUCTS = {
    "com.pillowtales.yearly",
    "premium_yearly",
    "premium_yearly:yearly",
}

SUBSCRIPTION_PRODUCTS = MONTHLY_PRODUCTS | YEARLY_PRODUCTS

PARENT_VOICE_CREDIT_PRODUCTS = {
    "parent_voice_1": 1,
    "parent_voice_3": 3,
}

PURCHASE_EVENT_TYPES = {
    "INITIAL_PURCHASE",
    "RENEWAL",
    "PRODUCT_CHANGE",
    "UNCANCELLATION",
    "NON_RENEWING_PURCHASE",
}

IGNORED_LIFECYCLE_EVENT_TYPES = {
    "CANCELLATION",
    "EXPIRATION",
    "BILLING_ISSUE",
    "SUBSCRIPTION_PAUSED",
    "TRANSFER",
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


def _get_event_type(event: dict) -> str:
    return str(event.get("type") or event.get("event_type") or "").upper()


def _supabase_env() -> tuple[str, str]:
    supabase_url = (
        os.getenv("SUPABASE_URL")
        or os.getenv("SUPABASE_PROJECT_URL")
        or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        or ""
    ).rstrip("/")

    service_key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_KEY")
        or ""
    )

    return supabase_url, service_key


async def _set_user_plan_premium(user_id: str) -> bool:
    """Mirror RevenueCat premium subscription status into users_profile.

    Story generation and settings have historically read more than one profile
    field, so subscription webhooks must keep both fields aligned after
    monthly/yearly purchases or restores.
    """
    supabase_url, service_key = _supabase_env()

    if not supabase_url or not service_key:
        return False

    url = f"{supabase_url}/rest/v1/users_profile?id=eq.{user_id}"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    payload = {
        "plan": "premium",
        "subscription_status": "premium",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.patch(url, headers=headers, json=payload)
        response.raise_for_status()

    return True


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
    event_type = _get_event_type(event)

    if not event_id:
        raise HTTPException(status_code=400, detail="Missing RevenueCat event id")

    processed = _read_processed_events()
    if event_id in processed:
        return {"status": "ignored", "reason": "already_processed"}

    if not user_id:
        raise HTTPException(status_code=400, detail="Missing RevenueCat app_user_id")

    if event_type in IGNORED_LIFECYCLE_EVENT_TYPES:
        processed.add(event_id)
        _write_processed_events(processed)
        return {
            "status": "ignored",
            "reason": "lifecycle_event_no_credit_or_plan_change",
            "event_type": event_type,
            "product_id": product_id,
        }

    should_process_purchase = not event_type or event_type in PURCHASE_EVENT_TYPES

    if not should_process_purchase:
        processed.add(event_id)
        _write_processed_events(processed)
        return {
            "status": "ignored",
            "reason": "unsupported_event_type",
            "event_type": event_type,
            "product_id": product_id,
        }

    is_subscription_purchase = product_id in SUBSCRIPTION_PRODUCTS
    is_yearly_purchase = product_id in YEARLY_PRODUCTS

    plan_synced = False
    if is_subscription_purchase:
        try:
            plan_synced = await _set_user_plan_premium(user_id)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to sync premium plan for RevenueCat subscriber: {exc}",
            ) from exc

    credits_to_add = 0

    if is_yearly_purchase:
        credits_to_add += 3

    if product_id in PARENT_VOICE_CREDIT_PRODUCTS:
        credits_to_add += PARENT_VOICE_CREDIT_PRODUCTS[product_id]

    if credits_to_add > 0:
        wallet = user_repo.get_parent_voice_wallet(user_id)
        current_credits = int(wallet.get("credits", 0))
        intro_used = bool(wallet.get("intro_used", False))

        new_credits = current_credits + credits_to_add

        user_repo.save_parent_voice_wallet(
            user_id,
            credits=new_credits,
            intro_used=intro_used,
        )
    else:
        wallet = user_repo.get_parent_voice_wallet(user_id)
        new_credits = int(wallet.get("credits", 0))

    processed.add(event_id)
    _write_processed_events(processed)

    if not is_subscription_purchase and credits_to_add <= 0:
        return {
            "status": "ignored",
            "reason": "no_credit_or_plan_change_required",
            "event_type": event_type,
            "product_id": product_id,
        }

    return {
        "status": "ok",
        "event_type": event_type,
        "user_id": user_id,
        "product_id": product_id,
        "premium_plan_synced": plan_synced,
        "credits_added": credits_to_add,
        "new_credit_balance": new_credits,
    }
