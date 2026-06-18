from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
from html import escape

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

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


def _fmt_metric(value: Any) -> str:
    if value is None:
        return '0'
    if isinstance(value, float):
        return f'{value:,.1f}'
    if isinstance(value, int):
        return f'{value:,}'
    return escape(str(value))


def _render_cards(items: list[tuple[str, Any, str]]) -> str:
    cards = []
    for label, value, hint in items:
        cards.append(
            '<div class="card">'
            f'<div class="metric-label">{escape(label)}</div>'
            f'<div class="metric-value">{_fmt_metric(value)}</div>'
            f'<div class="metric-hint">{escape(hint)}</div>'
            '</div>'
        )
    return ''.join(cards)


def _render_kv(title: str, data: dict[str, Any], *, limit: int = 10) -> str:
    rows = []
    items = list(data.items())[:limit]
    if not items:
        rows.append('<tr><td colspan="2" class="muted">No data yet</td></tr>')
    else:
        max_value = max((_safe_int(v) for _, v in items), default=0)
        for key, value in items:
            count = _safe_int(value)
            width = 0 if max_value <= 0 else max(4, round((count / max_value) * 100))
            rows.append(
                '<tr>'
                f'<td>{escape(str(key))}</td>'
                '<td class="bar-cell">'
                f'<span class="bar" style="width:{width}%"></span>'
                f'<span class="bar-value">{count}</span>'
                '</td>'
                '</tr>'
            )
    return (
        '<section class="panel">'
        f'<h2>{escape(title)}</h2>'
        '<table class="kv"><tbody>'
        + ''.join(rows) +
        '</tbody></table>'
        '</section>'
    )


def _render_period_table(periods: dict[str, dict[str, Any]]) -> str:
    labels = {
        'today_utc': 'Today UTC',
        'last_24h': 'Last 24h',
        'last_7d': 'Last 7d',
        'last_30d': 'Last 30d',
    }
    rows = []
    for key, label in labels.items():
        item = periods.get(key, {})
        rows.append(
            '<tr>'
            f'<td>{escape(label)}</td>'
            f'<td>{_fmt_metric(item.get("new_users", 0))}</td>'
            f'<td>{_fmt_metric(item.get("active_story_users", 0))}</td>'
            f'<td>{_fmt_metric(item.get("stories_created", 0))}</td>'
            f'<td>{_fmt_metric(item.get("narrations_created", 0))}</td>'
            '</tr>'
        )
    return (
        '<section class="panel wide">'
        '<h2>Activity by period</h2>'
        '<table><thead><tr><th>Period</th><th>New users</th><th>Story users</th><th>Stories</th><th>Narrations</th></tr></thead><tbody>'
        + ''.join(rows) +
        '</tbody></table></section>'
    )


def _render_recent_stories(stories: list[dict[str, Any]], *, limit: int = 12) -> str:
    rows = []
    for story in stories[:limit]:
        rows.append(
            '<tr>'
            f'<td>{escape(str(story.get("created_at") or ""))}</td>'
            f'<td>{escape(str(story.get("title") or "Untitled"))}</td>'
            f'<td>{escape(str(story.get("user_email") or ""))}</td>'
            f'<td>{escape(str(story.get("story_language") or ""))}</td>'
            f'<td>{escape(str(story.get("narration_language") or ""))}</td>'
            f'<td>{"Yes" if story.get("has_narration") else "No"}</td>'
            '</tr>'
        )
    if not rows:
        rows.append('<tr><td colspan="6" class="muted">No recent stories yet</td></tr>')
    return (
        '<section class="panel wide">'
        '<h2>Recent stories</h2>'
        '<table><thead><tr><th>Created</th><th>Title</th><th>User</th><th>Story lang</th><th>Narration lang</th><th>Narrated</th></tr></thead><tbody>'
        + ''.join(rows) +
        '</tbody></table></section>'
    )


def _render_recent_users(users: list[dict[str, Any]], *, limit: int = 12) -> str:
    rows = []
    for user in users[:limit]:
        rows.append(
            '<tr>'
            f'<td>{escape(str(user.get("created_at") or ""))}</td>'
            f'<td>{escape(str(user.get("email") or ""))}</td>'
            f'<td>{escape(str(user.get("preferred_language") or ""))}</td>'
            f'<td>{escape(str(user.get("plan") or ""))}</td>'
            f'<td>{escape(str(user.get("parent_voice_status") or "none"))}</td>'
            '</tr>'
        )
    if not rows:
        rows.append('<tr><td colspan="5" class="muted">No recent users yet</td></tr>')
    return (
        '<section class="panel wide">'
        '<h2>Recent users</h2>'
        '<table><thead><tr><th>Created</th><th>Email</th><th>Language</th><th>Plan</th><th>Parent Voice</th></tr></thead><tbody>'
        + ''.join(rows) +
        '</tbody></table></section>'
    )


def _dashboard_html(snapshot: dict[str, Any], include_internal: bool) -> str:
    headline = snapshot.get('headline', {})
    periods = snapshot.get('periods', {})
    users = snapshot.get('users', {})
    stories = snapshot.get('stories', {})
    parent_voice = snapshot.get('parent_voice', {})
    data_gaps = snapshot.get('data_gaps', {})
    generated_at = snapshot.get('generated_at', '')

    cards = _render_cards([
        ('Users', headline.get('total_users', 0), 'Excludes internal by default'),
        ('Premium', headline.get('premium_users', 0), 'Current premium users'),
        ('Stories', headline.get('stories_total', 0), 'Total story records'),
        ('Narrated', headline.get('stories_with_narration', 0), f'{headline.get("story_narration_rate_percent", 0)}% narration rate'),
        ('Parent Voice Ready', headline.get('parent_voice_ready_users', 0), 'Users with voice ready'),
        ('PV Intro Used', headline.get('parent_voice_intro_used_users', 0), 'Free intro consumed'),
    ])

    warning = ''
    source_errors = snapshot.get('source_errors', {}) or {}
    active_errors = {k: v for k, v in source_errors.items() if v}
    if active_errors:
        warning = '<div class="warning"><strong>Source errors:</strong> ' + escape(str(active_errors)) + '</div>'

    gaps = ', '.join(f'{k}: {v}' for k, v in data_gaps.items())
    internal_text = 'including internal users' if include_internal else 'excluding internal/test users'

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>PillowTales Dashboard</title>
<style>
:root {{ color-scheme: dark; --bg:#080b18; --panel:#12172a; --panel2:#171d33; --text:#eef1ff; --muted:#9ca3c7; --accent:#7c6cff; --accent2:#ffc857; --border:#252b44; --bad:#ff6b6b; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: radial-gradient(circle at top, #1a1f3d 0, var(--bg) 42%); color:var(--text); }}
main {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
header {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:24px; }}
h1 {{ margin:0; font-size:32px; letter-spacing:-0.03em; }}
.subtitle {{ color:var(--muted); margin-top:6px; }}
.badge {{ display:inline-block; border:1px solid var(--border); background:rgba(255,255,255,.04); color:var(--muted); padding:8px 10px; border-radius:999px; font-size:13px; }}
.grid {{ display:grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap:14px; }}
.card, .panel {{ background:linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.02)); border:1px solid var(--border); border-radius:18px; box-shadow:0 12px 32px rgba(0,0,0,.22); }}
.card {{ padding:18px; min-height:118px; }}
.metric-label {{ color:var(--muted); font-size:13px; }}
.metric-value {{ font-size:34px; font-weight:800; margin-top:10px; }}
.metric-hint {{ color:var(--muted); font-size:12px; margin-top:8px; }}
.panels {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:14px; margin-top:14px; }}
.panel {{ padding:18px; overflow:hidden; }}
.panel.wide {{ grid-column: 1 / -1; }}
h2 {{ font-size:18px; margin:0 0 14px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ text-align:left; padding:11px 10px; border-bottom:1px solid var(--border); vertical-align:middle; }}
th {{ color:var(--muted); font-weight:600; }}
tr:last-child td {{ border-bottom:0; }}
.muted {{ color:var(--muted); }}
.bar-cell {{ position:relative; min-width:140px; }}
.bar {{ display:inline-block; height:9px; border-radius:999px; background:linear-gradient(90deg, var(--accent), var(--accent2)); opacity:.9; margin-right:10px; vertical-align:middle; }}
.bar-value {{ color:var(--text); font-weight:700; }}
.warning {{ margin:14px 0; border:1px solid rgba(255,107,107,.5); color:#ffdede; background:rgba(255,107,107,.08); padding:12px 14px; border-radius:14px; }}
.footer {{ color:var(--muted); margin-top:18px; font-size:13px; line-height:1.5; }}
.controls {{ display:flex; gap:8px; justify-content:flex-end; flex-wrap:wrap; }}
a.button {{ color:var(--text); text-decoration:none; border:1px solid var(--border); padding:8px 12px; border-radius:12px; background:rgba(255,255,255,.05); }}
code {{ background:rgba(255,255,255,.08); padding:2px 5px; border-radius:6px; }}
@media (max-width: 1000px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .panels {{ grid-template-columns:1fr; }} header {{ flex-direction:column; }} }}
@media (max-width: 560px) {{ main {{ padding:16px; }} .grid {{ grid-template-columns:1fr; }} .metric-value {{ font-size:30px; }} table {{ font-size:12px; }} th,td {{ padding:9px 6px; }} }}
</style>
</head>
<body>
<main>
<header>
  <div>
    <h1>🌙 PillowTales Dashboard</h1>
    <div class="subtitle">Generated {escape(generated_at)} · {escape(internal_text)}</div>
  </div>
  <div class="controls"><span class="badge">Launch monitoring</span></div>
</header>
{warning}
<section class="grid">{cards}</section>
<section class="panels">
{_render_period_table(periods)}
{_render_kv('Story languages', stories.get('by_story_language', {}))}
{_render_kv('Narration languages', stories.get('by_narration_language', {}))}
{_render_kv('Themes', stories.get('by_theme', {}))}
{_render_kv('Morals', stories.get('by_moral', {}))}
{_render_kv('User preferred languages', users.get('by_preferred_language', {}))}
{_render_kv('Parent Voice status', parent_voice.get('by_status', {}))}
{_render_recent_stories(stories.get('recent_stories', []))}
{_render_recent_users(users.get('recent_new_users', []))}
</section>
<div class="footer">
  Data gaps: {escape(gaps)}<br />
  JSON endpoint remains available at <code>/api/admin/dashboard</code>. Browser view uses <code>/api/admin/dashboard-view?admin_key=...</code>.
</div>
</main>
</body>
</html>'''


@router.get('/dashboard-view', response_class=HTMLResponse)
async def get_admin_dashboard_view(
    include_internal: bool = Query(False, description='Include PillowTales/test/internal accounts in headline metrics.'),
    admin_key: str | None = Query(default=None, description='Admin key for browser access. Prefer header for API use.'),
    x_admin_key: str | None = Header(default=None),
    user_repo: UserRepository = Depends(get_user_repo),
) -> HTMLResponse:
    # Browsers cannot easily send custom headers, so this view accepts the same
    # key as a query parameter. Keep the URL private and do not share screenshots
    # containing the key.
    supplied_key = x_admin_key or admin_key
    snapshot = await get_admin_dashboard(
        include_internal=include_internal,
        x_admin_key=supplied_key,
        user_repo=user_repo,
    )
    return HTMLResponse(_dashboard_html(snapshot, include_internal=include_internal))
