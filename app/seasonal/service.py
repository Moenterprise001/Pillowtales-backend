"""Seasonal Stories catalogue and library service.

Rules (deliberately different from Story Worlds):
* Every text the app shows comes from the requested language. There is NO
  silent English fallback: an item without a published row in the requested
  language is left out and logged as ``[SEASONAL_MISSING_LANGUAGE]``, so a gap
  is visible in the logs instead of appearing in English inside a French app.
* Region, traditions and story credits are read per language
  (``ss_celebration_translations.display_labels`` and
  ``ss_canon_stories.author.credit_by_language``), never from English columns.
* Stories are frozen editorial content. A story is only served when its canon
  row is frozen and exactly one published edition matches the language and
  listener age. No language model is ever called.

This module has no FastAPI/Pydantic imports so it can be unit-tested on its own;
routes.py turns ``SeasonalError`` into HTTP errors.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger('pillowtales.seasonal')

SUPPORTED_LANGUAGES = ('en', 'es', 'fr', 'de', 'it', 'ja', 'ar')
AUDIENCES = ('child', 'teen', '16_plus')
LANES = ('canonical_traditional', 'cultural_narrative', 'pillowtales_original')


class SeasonalError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def normalise_language(code: Optional[str]) -> str:
    value = str(code or 'en').strip().lower().replace('_', '-').split('-', 1)[0]
    return value if value in SUPPORTED_LANGUAGES else 'en'


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).replace('Z', '+00:00')
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _is_frozen(story: dict) -> bool:
    rules = story.get('generation_rules') or {}
    return (
        isinstance(rules, dict)
        and rules.get('canon_frozen') is True
        and rules.get('master_status') == 'approved_frozen_master'
        and rules.get('runtime_story_generation_allowed') is False
    )


def _text(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ''
    return text or None


def _missing(kind: str, ref: str, language: str) -> None:
    logger.warning('[SEASONAL_MISSING_LANGUAGE] kind=%s ref=%s language=%s', kind, ref, language)


class SeasonalService:
    def __init__(self, repository, now: Optional[Callable[[], datetime]] = None):
        self.repository = repository
        self._now = now or (lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    def _window(self, celebration: dict) -> tuple[bool, bool]:
        """Return (live, coming_soon) for an enabled celebration."""
        now = self._now()
        starts = _parse_ts(celebration.get('available_from'))
        ends = _parse_ts(celebration.get('available_until'))
        published = bool(celebration.get('published'))
        in_window = (starts is None or starts <= now) and (ends is None or now <= ends)
        live = published and in_window
        upcoming = (published and starts is not None and now < starts) or (
            bool(celebration.get('coming_soon')) and not live and (ends is None or now <= ends)
        )
        return live, upcoming

    # ------------------------------------------------------------------
    # Celebrations
    # ------------------------------------------------------------------
    def list_celebrations(self, *, language: str = 'en', include_coming_soon: bool = False) -> dict:
        language = normalise_language(language)
        celebrations = self.repository.list_enabled_celebrations()
        translations = self._translations_by_celebration([c['id'] for c in celebrations])
        result = []
        for celebration in celebrations:
            live, upcoming = self._window(celebration)
            if not live and not (include_coming_soon and upcoming):
                continue
            item = self._celebration_model(celebration, translations.get(str(celebration['id']), {}),
                                           language, coming_soon=not live)
            if item:
                result.append(item)
        result.sort(key=lambda c: (c['presentation']['sortOrder'], c['slug']))
        return {'celebrations': result, 'count': len(result)}

    def get_celebration(self, slug: str, *, language: str = 'en') -> dict:
        language = normalise_language(language)
        celebration = self._live_celebration(slug)
        translations = self._translations_by_celebration([celebration['id']])
        item = self._celebration_model(celebration, translations.get(str(celebration['id']), {}),
                                       language, coming_soon=False)
        if not item:
            raise SeasonalError(404, 'Seasonal celebration is not available in the requested language')
        return item

    def _live_celebration(self, slug: str) -> dict:
        celebration = self.repository.get_enabled_celebration(str(slug or '').strip().lower())
        if not celebration or not self._window(celebration)[0]:
            raise SeasonalError(404, 'Seasonal celebration not found')
        return celebration

    def _translations_by_celebration(self, ids: list) -> dict[str, dict[str, dict]]:
        mapped: dict[str, dict[str, dict]] = {}
        for row in self.repository.celebration_translations(ids):
            lang = normalise_language(row.get('language_code'))
            if str(row.get('language_code') or '').lower().split('-')[0] != lang:
                continue
            current = mapped.setdefault(str(row['celebration_id']), {}).get(lang)
            if current is None or int(row.get('version') or 1) > int(current.get('version') or 1):
                mapped[str(row['celebration_id'])][lang] = row
        return mapped

    def _celebration_model(self, celebration: dict, by_language: dict[str, dict],
                           language: str, *, coming_soon: bool) -> Optional[dict]:
        translation = by_language.get(language)
        if not translation:
            _missing('celebration', str(celebration.get('slug')), language)
            return None
        labels = translation.get('display_labels') or {}
        if not isinstance(labels, dict):
            labels = {}
        traditions = labels.get('traditions')
        sections = []
        for index, section in enumerate(translation.get('discovery_sections') or []):
            if not isinstance(section, dict) or not _text(section.get('title')):
                continue
            sections.append({
                'slug': str(section.get('slug') or f'section-{index + 1}'),
                'title': str(section.get('title')),
                'body': str(section.get('body') or ''),
                'sortOrder': int(section.get('sort_order') or section.get('sortOrder') or (index + 1) * 10),
                'facts': [str(f) for f in (section.get('facts') or []) if _text(f)],
                'artwork': None,
            })
        return {
            'id': str(celebration['id']),
            'slug': str(celebration['slug']),
            'name': str(translation['name']),
            'shortDescription': str(translation.get('short_description') or ''),
            'description': str(translation.get('description') or ''),
            'originPlace': _text(translation.get('origin_place')),
            'originPeriod': _text(translation.get('origin_period')),
            'region': {
                'code': _text(celebration.get('region_code')),
                'name': _text(labels.get('region')),
            },
            'traditions': [str(t) for t in traditions if _text(t)] if isinstance(traditions, list) else [],
            'supportedLanguages': [lang for lang in SUPPORTED_LANGUAGES if lang in by_language],
            'artwork': {
                'coverUrl': _text(celebration.get('cover_url')),
                'thumbnailUrl': _text(celebration.get('thumbnail_url')),
                'iconUrl': _text(celebration.get('icon_url')),
            },
            'presentation': {
                'primaryColour': _text(celebration.get('primary_colour')),
                'secondaryColour': _text(celebration.get('secondary_colour')),
                'sortOrder': int(celebration.get('sort_order') or 100),
            },
            'availability': {
                'enabled': bool(celebration.get('enabled')),
                'published': bool(celebration.get('published')),
                'comingSoon': bool(coming_soon),
                'availableFrom': _text(celebration.get('available_from')),
                'availableUntil': _text(celebration.get('available_until')),
            },
            'discoverySections': sorted(sections, key=lambda s: s['sortOrder']),
            'edition': _text(celebration.get('edition')),
            'updatedAt': _text(translation.get('updated_at') or celebration.get('updated_at')),
        }

    # ------------------------------------------------------------------
    # Stories
    # ------------------------------------------------------------------
    def list_stories(self, slug: str, *, language: str = 'en', age: Optional[int] = None,
                     audience: str = 'child') -> dict:
        language = normalise_language(language)
        audience = audience if audience in AUDIENCES else 'child'
        if audience == 'child' and age is not None and not 1 <= int(age) <= 12:
            raise SeasonalError(422, 'age must be between 1 and 12')
        celebration = self._live_celebration(slug)
        stories = [s for s in self.repository.published_stories(celebration['id']) if _is_frozen(s)]
        if age is not None and audience == 'child':
            stories = [s for s in stories
                       if int(s.get('age_min') or 1) <= int(age) <= int(s.get('age_max') or 12)]
        ids = [s['id'] for s in stories]
        translations = self._single_by_story(self.repository.story_translations(ids, language), 'translation', language)
        editions = None
        if audience != 'child' or age is not None:
            editions = self._single_by_story(
                self.repository.adaptations(ids, language, audience, age if audience == 'child' else None),
                'edition', language)
        series_ids = {s['series_id'] for s in stories if s.get('series_id')}
        series = {str(r['id']): r for r in self.repository.series(series_ids)}
        series_titles = {str(r['series_id']): r for r in self.repository.series_translations(series.keys(), language)}

        result = []
        for story in stories:
            sid = str(story['id'])
            translation = translations.get(sid)
            if not translation:
                _missing('story', str(story.get('slug')), language)
                continue
            if editions is not None and sid not in editions:
                _missing('edition', f"{story.get('slug')}:{audience}:{age}", language)
                continue
            result.append(self._story_model(story, translation, language, series, series_titles))
        result.sort(key=lambda s: (s['sortOrder'], s['slug']))
        return {'celebrationSlug': str(celebration['slug']), 'stories': result, 'count': len(result)}

    @staticmethod
    def _single_by_story(rows: list[dict], kind: str, language: str) -> dict[str, dict]:
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(str(row['canon_story_id']), []).append(row)
        single = {}
        for story_id, items in grouped.items():
            if len(items) == 1:
                single[story_id] = items[0]
            else:
                logger.error('[SEASONAL_DUPLICATE_PUBLISHED] kind=%s story=%s language=%s count=%s',
                             kind, story_id, language, len(items))
        return single

    def _story_model(self, story: dict, translation: dict, language: str,
                     series: dict, series_titles: dict) -> dict:
        artwork = story.get('artwork') or {}
        scenes = sorted([s for s in (artwork.get('scenes') or []) if isinstance(s, dict) and s.get('url')],
                        key=lambda s: int(s.get('position') or 99))
        first = scenes[0]['url'] if scenes else None
        lane = story.get('lane') if story.get('lane') in LANES else 'canonical_traditional'
        series_model = None
        collection = None
        series_row = series.get(str(story.get('series_id'))) if story.get('series_id') else None
        if series_row:
            title_row = series_titles.get(str(series_row['id']))
            if title_row:
                series_model = {
                    'slug': str(series_row['slug']),
                    'title': str(title_row['title']),
                    'summary': _text(title_row.get('summary')),
                    'partNumber': int(story['part_number']) if story.get('part_number') else None,
                    'partCount': int(series_row.get('part_count') or 0) or None,
                }
                # The September frontend groups stories by ``collection``; a series
                # is presented as its own group so the parts appear together.
                collection = {
                    'slug': str(series_row['slug']),
                    'title': str(title_row['title']),
                    'description': _text(title_row.get('summary')),
                    'sortOrder': int(series_row.get('sort_order') or 100),
                    'artwork': {'coverUrl': first, 'thumbnailUrl': first, 'iconUrl': None},
                    'published': True,
                }
            else:
                _missing('series', str(series_row.get('slug')), language)
        return {
            'slug': str(story['slug']),
            'title': str(translation['title']),
            'subtitle': _text(translation.get('subtitle')),
            'summary': str(translation.get('summary') or ''),
            'lane': lane,
            'ageRange': {'min': int(story.get('age_min') or 1), 'max': int(story.get('age_max') or 12)},
            'artwork': {'coverUrl': first, 'thumbnailUrl': first, 'iconUrl': None},
            'collection': collection,
            'series': series_model,
            'author': self._author(story, language),
            'published': True,
            'sortOrder': int(story.get('sort_order') or 100),
        }

    @staticmethod
    def _author(story: dict, language: str) -> Optional[dict]:
        author = story.get('author') or {}
        if not isinstance(author, dict) or not author:
            return None
        by_language = author.get('credit_by_language') or {}
        credit = _text(by_language.get(language)) if isinstance(by_language, dict) else None
        if credit is None and language == 'en':
            credit = _text(author.get('credit'))
        if credit is None:
            _missing('credit', str(story.get('slug')), language)
        if credit is None and not _text(author.get('name')):
            return None
        return {'name': _text(author.get('name')), 'displayName': credit}

    # ------------------------------------------------------------------
    # Saving a story to the family's library
    # ------------------------------------------------------------------
    def build_library_record(self, slug: str, story_slug: str, *, user_id: str, language: str,
                             audience: str = 'child', age: Optional[int] = None) -> dict:
        language = normalise_language(language)
        if audience not in AUDIENCES:
            raise SeasonalError(422, 'audience must be child, teen or 16_plus')
        if audience == 'child':
            if age is None or not 1 <= int(age) <= 12:
                raise SeasonalError(422, 'age must be between 1 and 12 for a child listener')
            age = int(age)
        else:
            age = None
        celebration = self._live_celebration(slug)
        story_slug = str(story_slug or '').strip().lower()
        story = next((s for s in self.repository.published_stories(celebration['id'])
                      if str(s.get('slug')) == story_slug), None)
        if not story:
            raise SeasonalError(404, 'Seasonal story not found')
        if not _is_frozen(story):
            raise SeasonalError(409, 'This seasonal story is not available yet')
        translations = self.repository.story_translations([story['id']], language)
        editions = self.repository.adaptations([story['id']], language, audience, age, with_pages=True)
        if len(translations) != 1 or len(editions) != 1:
            logger.error('[SEASONAL_EDITION_UNAVAILABLE] story=%s language=%s audience=%s age=%s translations=%s editions=%s',
                         story_slug, language, audience, age, len(translations), len(editions))
            raise SeasonalError(409, 'This seasonal story is not available in the requested language')
        pages = editions[0].get('pages')
        if not isinstance(pages, list) or not pages or not all(isinstance(p, str) and p.strip() for p in pages):
            raise SeasonalError(409, 'This seasonal story has no valid pages')
        pages = [p.strip() for p in pages]
        translation = translations[0]
        now = self._now().isoformat()
        return {
            'user_id': user_id,
            'title': str(translation['title']),
            'child_name': None,
            'age': age,
            'theme': f"seasonal:{celebration['slug']}:{story['slug']}",
            'moral': None,
            'language': language,
            'story_language_code': language,
            'narration_language_code': language,
            'pages': pages,
            'full_text': '\n\n'.join(pages),
            'story_summary': str(translation.get('summary') or ''),
            'audio_url': None,
            'audio_status': 'none',
            'is_favorite': False,
            'generation_status': 'complete',
            'expected_pages': len(pages),
            'generation_error': None,
            'created_at': now,
        }

    # ------------------------------------------------------------------
    # Pictures for a saved library story (used by GET /stories/{id}/artwork)
    # ------------------------------------------------------------------
    @staticmethod
    def parse_library_theme(theme: Any) -> Optional[tuple[str, str]]:
        parts = str(theme or '').split(':')
        if len(parts) == 3 and parts[0] == 'seasonal' and parts[1] and parts[2]:
            return parts[1], parts[2]
        return None

    def artwork_for_saved_story(self, story: dict) -> Optional[dict]:
        parsed = self.parse_library_theme(story.get('theme'))
        if not parsed:
            return None
        artwork = self.repository.published_story_artwork(*parsed)
        if not isinstance(artwork, dict):
            return None
        language = normalise_language(story.get('story_language_code') or story.get('language'))
        localised = dict(artwork)
        scenes = []
        for scene in artwork.get('scenes') or []:
            if not isinstance(scene, dict):
                continue
            copy = dict(scene)
            by_language = scene.get('alt_text_by_language') or {}
            if isinstance(by_language, dict) and _text(by_language.get(language)):
                copy['alt_text'] = by_language[language]
            elif language != 'en':
                copy['alt_text'] = ''  # never read an English description inside another language
            copy.pop('alt_text_by_language', None)
            scenes.append(copy)
        localised['scenes'] = scenes
        alt = artwork.get('alt_text') if isinstance(artwork.get('alt_text'), dict) else {}
        localised['alt_text'] = {language: alt.get(language, '')}
        return localised
