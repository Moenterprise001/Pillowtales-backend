from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.api.deps import get_user_repo
from app.repositories.user_repository import UserRepository

router = APIRouter(prefix='/admin', tags=['admin-dashboard'])


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    try:
        # Supabase usually returns ISO strings with +00:00. Handle trailing Z too.
        normalised = value.replace('Z', '+00:00')
        parsed = datetime.fromisoformat(normalised)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except Exception:
        return default


def _pct(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round((part / whole) * 100, 1)


def _mask_email(email: str | None) -> str | None:
    if not email:
        return None
    if '@' not in email:
        return email[:2] + '***'
    local, domain = email.split('@', 1)
    if len(local) <= 2:
        masked_local = local[0] + '***' if local else '***'
    else:
        masked_local = f'{local[:2]}***{local[-1:]}'
    return f'{masked_local}@{domain}'


def _short_id(value: str | None) -> str | None:
    if not value:
        return None
    return f'{value[:8]}…{value[-4:]}' if len(value) > 14 else value


def _env_csv(name: str, default: str = '') -> set[str]:
    raw = os.getenv(name, default)
    return {item.strip().lower() for item in raw.split(',') if item.strip()}


def _verify_admin_key(x_admin_key: str | None) -> None:
    expected = (
        os.getenv('ADMIN_DASHBOARD_KEY')
        or os.getenv('DASHBOARD_ADMIN_KEY')
        or ''
    ).strip()

    if not expected:
        raise HTTPException(
            status_code=503,
            detail='Admin dashboard is not configured. Set ADMIN_DASHBOARD_KEY on the backend.',
        )

    if not x_admin_key or x_admin_key.strip() != expected:
        raise HTTPException(status_code=401, detail='Invalid admin dashboard key.')


def _fetch_table_rows(
    user_repo: UserRepository,
    table_name: str,
    select_columns: str = '*',
    *,
    order_column: str | None = None,
    desc: bool = True,
    max_rows: int = 5000,
) -> list[dict]:
    query = user_repo.client.table(table_name).select(select_columns)
    if order_column:
        query = query.order(order_column, desc=desc)

    result = query.range(0, max_rows - 1).execute()
    return result.data or []


def _safe_fetch_table_rows(
    user_repo: UserRepository,
    table_name: str,
    select_columns: str = '*',
    *,
    order_column: str | None = None,
    desc: bool = True,
    max_rows: int = 5000,
) -> tuple[list[dict], str | None]:
    try:
        return _fetch_table_rows(
            user_repo,
            table_name,
            select_columns,
            order_column=order_column,
            desc=desc,
            max_rows=max_rows,
        ), None
    except Exception as exc:
        return [], str(exc)


def _is_internal_profile(profile: dict, internal_domains: set[str], internal_emails: set[str], internal_user_ids: set[str]) -> bool:
    user_id = str(profile.get('id') or '').lower()
    email = str(profile.get('email') or '').strip().lower()

    if user_id and user_id in internal_user_ids:
        return True

    if email and email in internal_emails:
        return True

    domain = email.split('@', 1)[1] if '@' in email else ''
    if domain and domain in internal_domains:
        return True

    local = email.split('@', 1)[0] if '@' in email else email
    if local.startswith('qa-') or local.startswith('qa_') or local in {'qa', 'qaexpired', 'qa-expired'}:
        return True

    return False


def _count_by(rows: list[dict], key: str, *, fallback: str = 'unknown') -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = row.get(key) or fallback
        counter[str(value)] += 1
    return dict(counter.most_common())


def _count_stories_in_window(stories: list[dict], since: datetime, now: datetime) -> int:
    return sum(
        1 for story in stories
        if (created := _parse_dt(story.get('created_at'))) and since <= created <= now
    )


def _count_users_in_window(users: list[dict], since: datetime, now: datetime) -> int:
    return sum(
        1 for user in users
        if (created := _parse_dt(user.get('created_at'))) and since <= created <= now
    )


def _unique_story_users_in_window(stories: list[dict], since: datetime, now: datetime) -> int:
    user_ids = {
        story.get('user_id')
        for story in stories
        if story.get('user_id') and (created := _parse_dt(story.get('created_at'))) and since <= created <= now
    }
    return len(user_ids)


def _narrations_in_window(stories: list[dict], since: datetime, now: datetime) -> int:
    return sum(
        1 for story in stories
        if (created := _parse_dt(story.get('audio_created_at'))) and since <= created <= now
    )


def _build_periods(now: datetime) -> dict[str, datetime]:
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return {
        'today_utc': start_of_today,
        'last_24h': now - timedelta(hours=24),
        'last_7d': now - timedelta(days=7),
        'last_30d': now - timedelta(days=30),
    }


@router.get('/dashboard')
async def get_admin_dashboard(
    include_internal: bool = Query(False, description='Include PillowTales/test/internal accounts in headline metrics.'),
    x_admin_key: str | None = Header(default=None),
    user_repo: UserRepository = Depends(get_user_repo),
) -> dict:
    """Return a launch dashboard snapshot.

    This endpoint is intentionally read-only and protected by ADMIN_DASHBOARD_KEY.
    It does not touch reader, narration, story generation, subscriptions, or account
    mutation logic.
    """
    _verify_admin_key(x_admin_key)

    now = _utc_now()
    periods = _build_periods(now)
    max_rows = _safe_int(os.getenv('ADMIN_DASHBOARD_MAX_ROWS'), 5000)

    users_columns = (
        'id,email,preferred_language,plan,subscription_status,created_at,updated_at,'
        'trial_start,trial_end,trial_narrations_used,last_narration_reset,'
        'parent_voice_status,parent_voice_consent,parent_voice_created_at,parent_voice_deleted_at,'
        'parent_voice_story_credits,parent_voice_intro_used'
    )
    stories_columns = (
        'id,user_id,title,theme,moral,language,story_language_code,narration_language_code,'
        'audio_created_at,audio_status,created_at,generation_status,expected_pages,generation_error'
    )

    users, users_error = _safe_fetch_table_rows(
        user_repo,
        'users_profile',
        users_columns,
        order_column='created_at',
        desc=True,
        max_rows=max_rows,
    )
    stories, stories_error = _safe_fetch_table_rows(
        user_repo,
        'stories',
        stories_columns,
        order_column='created_at',
        desc=True,
        max_rows=max_rows,
    )
    subscriptions, subscriptions_error = _safe_fetch_table_rows(
        user_repo,
        'subscriptions',
        '*',
        order_column='created_at',
        desc=True,
        max_rows=max_rows,
    )

    internal_domains = _env_csv('ADMIN_INTERNAL_EMAIL_DOMAINS', 'pillowtales.co')
    internal_emails = _env_csv('ADMIN_INTERNAL_EMAILS')
    internal_user_ids = _env_csv('ADMIN_INTERNAL_USER_IDS')

    internal_user_id_set = {
        str(profile.get('id'))
        for profile in users
        if _is_internal_profile(profile, internal_domains, internal_emails, internal_user_ids)
    }

    active_users = users if include_internal else [u for u in users if str(u.get('id')) not in internal_user_id_set]
    active_user_ids = {str(u.get('id')) for u in active_users if u.get('id')}
    active_stories = stories if include_internal else [s for s in stories if str(s.get('user_id')) in active_user_ids]

    premium_users = [
        user for user in active_users
        if str(user.get('subscription_status') or user.get('plan') or '').lower() == 'premium'
    ]

    parent_voice_ready = [u for u in active_users if str(u.get('parent_voice_status') or '').lower() == 'ready']
    parent_voice_consented = [u for u in active_users if bool(u.get('parent_voice_consent'))]
    parent_voice_intro_used = [u for u in active_users if bool(u.get('parent_voice_intro_used'))]
    parent_voice_credits_total = sum(_safe_int(u.get('parent_voice_story_credits')) for u in active_users)

    narrated_stories = [s for s in active_stories if s.get('audio_created_at')]
    complete_stories = [
        s for s in active_stories
        if str(s.get('generation_status') or '').lower() == 'complete'
        or (_safe_int(s.get('expected_pages')) > 0)
    ]
    failed_stories = [s for s in active_stories if s.get('generation_error')]

    story_lang_rows = []
    for story in active_stories:
        story_lang = story.get('story_language_code') or story.get('language') or 'unknown'
        narration_lang = story.get('narration_language_code') or story.get('audio_language_code') or story_lang
        story_lang_rows.append({**story, '_story_lang': story_lang, '_narration_lang': narration_lang})

    language_counter: Counter[str] = Counter(str(row['_story_lang']) for row in story_lang_rows)
    narration_language_counter: Counter[str] = Counter(str(row['_narration_lang']) for row in story_lang_rows if row.get('audio_created_at') or row.get('_narration_lang'))

    recent_stories = []
    profile_by_id = {str(user.get('id')): user for user in users}
    for story in active_stories[:25]:
        user_id = str(story.get('user_id') or '')
        profile = profile_by_id.get(user_id, {})
        recent_stories.append({
            'story_id': _short_id(str(story.get('id') or '')),
            'user_id': _short_id(user_id),
            'user_email': _mask_email(profile.get('email')),
            'title': story.get('title'),
            'created_at': story.get('created_at'),
            'story_language': story.get('story_language_code') or story.get('language') or 'unknown',
            'narration_language': story.get('narration_language_code') or 'unknown',
            'theme': story.get('theme'),
            'moral': story.get('moral'),
            'has_narration': bool(story.get('audio_created_at')),
            'generation_status': story.get('generation_status'),
            'generation_error': bool(story.get('generation_error')),
        })

    def period_payload(since: datetime) -> dict:
        return {
            'new_users': _count_users_in_window(active_users, since, now),
            'active_story_users': _unique_story_users_in_window(active_stories, since, now),
            'stories_created': _count_stories_in_window(active_stories, since, now),
            'narrations_created': _narrations_in_window(active_stories, since, now),
        }

    headline = {
        'total_users': len(active_users),
        'total_users_all_including_internal': len(users),
        'internal_users_excluded': 0 if include_internal else len(internal_user_id_set),
        'premium_users': len(premium_users),
        'free_or_trial_users': max(len(active_users) - len(premium_users), 0),
        'stories_total': len(active_stories),
        'stories_with_narration': len(narrated_stories),
        'story_narration_rate_percent': _pct(len(narrated_stories), len(active_stories)),
        'story_generation_failures': len(failed_stories),
        'parent_voice_ready_users': len(parent_voice_ready),
        'parent_voice_consented_users': len(parent_voice_consented),
        'parent_voice_intro_used_users': len(parent_voice_intro_used),
        'parent_voice_credits_total': parent_voice_credits_total,
    }

    return {
        'status': 'ok',
        'generated_at': now.isoformat(),
        'scope': {
            'include_internal': include_internal,
            'max_rows_per_table': max_rows,
            'internal_domains_excluded_by_default': sorted(internal_domains),
            'internal_user_ids_excluded_count': 0 if include_internal else len(internal_user_id_set),
        },
        'headline': headline,
        'periods': {
            name: period_payload(since)
            for name, since in periods.items()
        },
        'users': {
            'by_preferred_language': _count_by(active_users, 'preferred_language'),
            'by_plan': _count_by(active_users, 'plan'),
            'by_subscription_status': _count_by(active_users, 'subscription_status'),
            'recent_new_users': [
                {
                    'user_id': _short_id(str(user.get('id') or '')),
                    'email': _mask_email(user.get('email')),
                    'created_at': user.get('created_at'),
                    'preferred_language': user.get('preferred_language'),
                    'plan': user.get('subscription_status') or user.get('plan') or 'free',
                    'parent_voice_status': user.get('parent_voice_status') or 'none',
                }
                for user in active_users[:20]
            ],
        },
        'stories': {
            'by_story_language': dict(language_counter.most_common()),
            'by_narration_language': dict(narration_language_counter.most_common()),
            'by_theme': _count_by(active_stories, 'theme'),
            'by_moral': _count_by(active_stories, 'moral'),
            'by_generation_status': _count_by(active_stories, 'generation_status'),
            'recent_stories': recent_stories,
        },
        'parent_voice': {
            'by_status': _count_by(active_users, 'parent_voice_status', fallback='none'),
            'ready_users': len(parent_voice_ready),
            'consented_users': len(parent_voice_consented),
            'intro_used_users': len(parent_voice_intro_used),
            'credits_total': parent_voice_credits_total,
            'users_with_credits': sum(1 for u in active_users if _safe_int(u.get('parent_voice_story_credits')) > 0),
        },
        'subscriptions': {
            'current_premium_users': len(premium_users),
            'subscriptions_table_rows': len(subscriptions),
            'note': 'RevenueCat purchase events are currently logged but not persisted to a purchases table; this dashboard can show current premium users, not full revenue history.',
        },
        'data_gaps': {
            'country': 'not stored yet',
            'platform': 'not stored yet',
            'app_version': 'not stored yet',
            'revenue_history': 'not persisted yet; RevenueCat webhooks are logged only',
        },
        'source_errors': {
            'users_profile': users_error,
            'stories': stories_error,
            'subscriptions': subscriptions_error,
        },
    }
