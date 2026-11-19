"""Read/insert access for Seasonal Stories.

Only the Seasonal (ss_*) tables are ever read. The one shared write is the
family's library row in ``stories``, inserted through the existing
StoryRepository exactly like Story Worlds frozen stories.
"""
from __future__ import annotations

from typing import Iterable, Optional

ALLOWED_TABLES = frozenset({
    'ss_celebrations',
    'ss_celebration_translations',
    'ss_canon_stories',
    'ss_canon_story_translations',
    'ss_canon_story_adaptations',
    'ss_canon_series',
    'ss_canon_series_translations',
})


class SeasonalRepository:
    def __init__(self, client):
        self.client = client

    def _table(self, name: str):
        if name not in ALLOWED_TABLES:
            raise ValueError(f'Seasonal repository may not read {name}')
        return self.client.table(name)

    # -- celebrations -----------------------------------------------------
    def list_enabled_celebrations(self) -> list[dict]:
        result = (
            self._table('ss_celebrations')
            .select('*')
            .eq('enabled', True)
            .is_('archived_at', 'null')
            .order('sort_order')
            .execute()
        )
        return result.data or []

    def get_enabled_celebration(self, slug: str) -> Optional[dict]:
        result = (
            self._table('ss_celebrations')
            .select('*')
            .eq('slug', slug)
            .eq('enabled', True)
            .is_('archived_at', 'null')
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def celebration_translations(self, celebration_ids: Iterable[str]) -> list[dict]:
        """All published translations (every language) for these celebrations."""
        ids = [str(i) for i in celebration_ids]
        if not ids:
            return []
        result = (
            self._table('ss_celebration_translations')
            .select('*')
            .in_('celebration_id', ids)
            .eq('published', True)
            .execute()
        )
        return result.data or []

    # -- stories ----------------------------------------------------------
    def published_stories(self, celebration_id: str) -> list[dict]:
        result = (
            self._table('ss_canon_stories')
            .select('id,slug,title,summary,age_min,age_max,artwork,generation_rules,lane,author,'
                    'series_id,part_number,sort_order,active,published')
            .eq('celebration_id', str(celebration_id))
            .eq('active', True)
            .eq('published', True)
            .order('sort_order')
            .execute()
        )
        return result.data or []

    def story_translations(self, story_ids: Iterable[str], language: str) -> list[dict]:
        ids = [str(i) for i in story_ids]
        if not ids:
            return []
        result = (
            self._table('ss_canon_story_translations')
            .select('id,canon_story_id,language_code,title,subtitle,summary,version')
            .in_('canon_story_id', ids)
            .eq('language_code', language)
            .eq('published', True)
            .execute()
        )
        return result.data or []

    def adaptations(self, story_ids: Iterable[str], language: str, audience: str,
                    listener_age: Optional[int], with_pages: bool = False) -> list[dict]:
        ids = [str(i) for i in story_ids]
        if not ids:
            return []
        columns = 'id,canon_story_id,language_code,audience,listener_age,version'
        if with_pages:
            columns += ',pages'
        query = (
            self._table('ss_canon_story_adaptations')
            .select(columns)
            .in_('canon_story_id', ids)
            .eq('language_code', language)
            .eq('audience', audience)
            .eq('published', True)
        )
        if audience == 'child':
            query = query.eq('listener_age', int(listener_age))
        else:
            query = query.is_('listener_age', 'null')
        return query.execute().data or []

    # -- series -----------------------------------------------------------
    def series(self, series_ids: Iterable[str]) -> list[dict]:
        ids = [str(i) for i in series_ids if i]
        if not ids:
            return []
        result = (
            self._table('ss_canon_series')
            .select('id,slug,part_count,sort_order,artwork,active,published')
            .in_('id', ids)
            .eq('active', True)
            .eq('published', True)
            .execute()
        )
        return result.data or []

    def series_translations(self, series_ids: Iterable[str], language: str) -> list[dict]:
        ids = [str(i) for i in series_ids if i]
        if not ids:
            return []
        result = (
            self._table('ss_canon_series_translations')
            .select('series_id,language_code,title,summary')
            .in_('series_id', ids)
            .eq('language_code', language)
            .eq('published', True)
            .execute()
        )
        return result.data or []

    # -- artwork for a saved library story ----------------------------------
    def published_story_artwork(self, celebration_slug: str, story_slug: str) -> Optional[dict]:
        celebration = (
            self._table('ss_celebrations')
            .select('id')
            .eq('slug', celebration_slug)
            .limit(1)
            .execute()
        ).data
        if not celebration:
            return None
        result = (
            self._table('ss_canon_stories')
            .select('artwork')
            .eq('celebration_id', str(celebration[0]['id']))
            .eq('slug', story_slug)
            .eq('published', True)
            .limit(1)
            .execute()
        )
        if not result.data:
            return None
        artwork = result.data[0].get('artwork')
        return artwork if isinstance(artwork, dict) else None
