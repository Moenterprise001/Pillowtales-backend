from __future__ import annotations

from typing import Iterable, Optional

from supabase import Client


class StoryWorldRepository:
    """Data access for Story Worlds.

    This repository contains no knowledge of individual Story Worlds. All world
    identity, translations, availability and monthly weighting come from data.
    """

    def __init__(self, client: Client):
        self.client = client

    def list_published_worlds(
        self,
        *,
        region: Optional[str] = None,
        category: Optional[str] = None,
        age: Optional[int] = None,
    ) -> list[dict]:
        query = (
            self.client.table('story_worlds')
            .select('*')
            .eq('enabled', True)
            .eq('published', True)
            .eq('coming_soon', False)
            .is_('archived_at', 'null')
        )
        if region:
            query = query.eq('region_code', region.lower())
        if category:
            query = query.eq('category', category.lower())
        if age is not None:
            query = query.lte('age_min', age).gte('age_max', age)
        result = query.order('sort_order').order('slug').execute()
        return result.data or []

    def get_published_world(self, slug: str) -> Optional[dict]:
        result = (
            self.client.table('story_worlds')
            .select('*')
            .eq('slug', slug)
            .eq('enabled', True)
            .eq('published', True)
            .eq('coming_soon', False)
            .is_('archived_at', 'null')
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def get_translations(self, world_ids: Iterable[str]) -> list[dict]:
        ids = list(world_ids)
        if not ids:
            return []
        result = (
            self.client.table('story_world_translations')
            .select('*')
            .in_('story_world_id', ids)
            .eq('published', True)
            .execute()
        )
        return result.data or []

    def get_active_prompt_pack_languages(self, world_ids: Iterable[str]) -> list[dict]:
        ids = list(world_ids)
        if not ids:
            return []
        result = (
            self.client.table('story_world_prompt_packs')
            .select('story_world_id,language_code,version')
            .in_('story_world_id', ids)
            .eq('published', True)
            .eq('active', True)
            .execute()
        )
        return result.data or []

    def get_monthly_weights(self, world_ids: Iterable[str], month: int) -> list[dict]:
        ids = list(world_ids)
        if not ids:
            return []
        result = (
            self.client.table('story_world_monthly_weights')
            .select('story_world_id,month,weight')
            .in_('story_world_id', ids)
            .eq('month', month)
            .execute()
        )
        return result.data or []

    def get_global_pronunciations(self, language_code: str) -> list[dict]:
        """Return active global PillowTales pronunciation records for a language.

        Global pronunciation rows use story_world_id = NULL. Verification policy
        remains owned by StoryWorldPronunciationService so this repository stays
        a data-access layer only.
        """
        language = (
            str(language_code or 'en')
            .strip()
            .lower()
            .replace('_', '-')
            .split('-', 1)[0]
        )

        result = (
            self.client.table('story_world_pronunciations')
            .select('*')
            .is_('story_world_id', 'null')
            .eq('language_code', language)
            .eq('active', True)
            .order('display_text')
            .order('version', desc=True)
            .execute()
        )
        return result.data or []

    def get_pronunciations(self, world_id: str, language_code: str) -> list[dict]:
        """Return active pronunciation records for a Story World and language.

        Language matching uses the base language (for example en-GB -> en).
        Verification policy is intentionally owned by the pronunciation service,
        so this repository remains a data-access layer only.
        """
        world_key = str(world_id or '').strip()
        language = str(language_code or 'en').strip().lower().replace('_', '-').split('-', 1)[0]
        if not world_key:
            return []

        result = (
            self.client.table('story_world_pronunciations')
            .select('*')
            .eq('story_world_id', world_key)
            .eq('language_code', language)
            .eq('active', True)
            .order('display_text')
            .order('version', desc=True)
            .execute()
        )
        return result.data or []

    def list_original_folk_stories(
        self,
        slug: str,
        age: int,
        language_code: str = 'en',
    ) -> tuple[Optional[dict], list[dict]]:
        """Return published, age-eligible Canon stories with catalogue metadata.

        Each returned story row is enriched with internal catalogue keys:

        - ``_collection``
        - ``_series``
        - ``_collection_translation``
        - ``_series_translation``
        - ``_story_translation``

        The public API mapping remains the responsibility of the service layer.
        English is used as the translation fallback.
        """
        world = self.get_published_world(str(slug or '').strip().lower())
        if not world:
            return None, []

        world_id = str(world['id'])
        language = str(language_code or 'en').strip().lower().replace('_', '-').split('-', 1)[0]
        requested_languages = [language] if language == 'en' else [language, 'en']

        story_result = (
            self.client.table('story_world_canon_stories')
            .select('*')
            .eq('story_world_id', world_id)
            .eq('active', True)
            .eq('published', True)
            .lte('age_min', age)
            .gte('age_max', age)
            .order('sort_order')
            .order('title')
            .execute()
        )
        stories = story_result.data or []
        if not stories:
            return world, []

        collection_ids = sorted(
            {
                str(row['collection_id'])
                for row in stories
                if row.get('collection_id')
            }
        )
        series_ids = sorted(
            {
                str(row['series_id'])
                for row in stories
                if row.get('series_id')
            }
        )
        story_ids = [str(row['id']) for row in stories if row.get('id')]

        collections: list[dict] = []
        if collection_ids:
            collection_result = (
                self.client.table('story_world_canon_collections')
                .select('*')
                .in_('id', collection_ids)
                .eq('story_world_id', world_id)
                .eq('active', True)
                .eq('published', True)
                .order('sort_order')
                .order('title')
                .execute()
            )
            collections = collection_result.data or []

        series_rows: list[dict] = []
        if series_ids:
            series_result = (
                self.client.table('story_world_canon_series')
                .select('*')
                .in_('id', series_ids)
                .eq('story_world_id', world_id)
                .eq('active', True)
                .eq('published', True)
                .order('sort_order')
                .order('title')
                .execute()
            )
            series_rows = series_result.data or []

        collection_translations: list[dict] = []
        if collection_ids:
            collection_translation_result = (
                self.client.table('story_world_canon_collection_translations')
                .select('*')
                .in_('collection_id', collection_ids)
                .in_('language_code', requested_languages)
                .eq('published', True)
                .execute()
            )
            collection_translations = collection_translation_result.data or []

        series_translations: list[dict] = []
        if series_ids:
            series_translation_result = (
                self.client.table('story_world_canon_series_translations')
                .select('*')
                .in_('series_id', series_ids)
                .in_('language_code', requested_languages)
                .eq('published', True)
                .execute()
            )
            series_translations = series_translation_result.data or []

        story_translations: list[dict] = []
        if story_ids:
            story_translation_result = (
                self.client.table('story_world_canon_story_translations')
                .select('*')
                .in_('canon_story_id', story_ids)
                .in_('language_code', requested_languages)
                .eq('published', True)
                .order('version', desc=True)
                .execute()
            )
            story_translations = story_translation_result.data or []

        collection_by_id = {str(row['id']): row for row in collections}
        series_by_id = {str(row['id']): row for row in series_rows}

        def translation_map(rows: list[dict], id_field: str) -> dict[str, dict[str, dict]]:
            mapped: dict[str, dict[str, dict]] = {}
            for row in rows:
                parent_id = str(row.get(id_field) or '')
                row_language = str(row.get('language_code') or '').strip().lower().replace('_', '-').split('-', 1)[0]
                if not parent_id or not row_language:
                    continue
                mapped.setdefault(parent_id, {})
                # Rows are deterministic except story translations, which are
                # ordered newest version first. Keep the first row per language.
                mapped[parent_id].setdefault(row_language, row)
            return mapped

        collection_translation_map = translation_map(
            collection_translations,
            'collection_id',
        )
        series_translation_map = translation_map(
            series_translations,
            'series_id',
        )
        story_translation_map = translation_map(
            story_translations,
            'canon_story_id',
        )

        def choose_translation(
            mapped: dict[str, dict[str, dict]],
            parent_id: Optional[str],
        ) -> Optional[dict]:
            if not parent_id:
                return None
            by_language = mapped.get(str(parent_id), {})
            return by_language.get(language) or by_language.get('en')

        enriched: list[dict] = []
        for row in stories:
            item = dict(row)
            collection_id = str(row['collection_id']) if row.get('collection_id') else None
            series_id = str(row['series_id']) if row.get('series_id') else None
            story_id = str(row['id']) if row.get('id') else None

            item['_collection'] = collection_by_id.get(collection_id) if collection_id else None
            item['_series'] = series_by_id.get(series_id) if series_id else None
            item['_collection_translation'] = choose_translation(
                collection_translation_map,
                collection_id,
            )
            item['_series_translation'] = choose_translation(
                series_translation_map,
                series_id,
            )
            item['_story_translation'] = choose_translation(
                story_translation_map,
                story_id,
            )
            enriched.append(item)

        return world, enriched

    def list_folk_adventure_sources(self, slug: str, age: int) -> tuple[Optional[dict], list[dict]]:
        """Return published folklore sources eligible for Folk Adventure.

        No country-specific logic lives here. Eligibility comes entirely from
        Story World and canon-source data.
        """
        world = self.get_published_world(str(slug or '').strip().lower())
        if not world:
            return None, []

        result = (
            self.client.table('story_world_canon_stories')
            .select('*')
            .eq('story_world_id', str(world['id']))
            .eq('active', True)
            .eq('published', True)
            .eq('living_world_expansion_allowed', True)
            .lte('age_min', age)
            .gte('age_max', age)
            .order('title')
            .execute()
        )

        eligible: list[dict] = []
        for row in (result.data or []):
            generation_rules = row.get('generation_rules') or {}
            folk_rules = generation_rules.get('folk_adventure') if isinstance(generation_rules, dict) else None
            if isinstance(folk_rules, dict) and folk_rules.get('allowed') is False:
                continue
            eligible.append(row)
        return world, eligible

    def get_generation_context(self, slug: str, language_code: str, age: int, mode: str) -> Optional[dict]:
        """Resolve backend-only Story World generation data.

        Returns published world metadata plus the active prompt pack, Story DNA,
        editorial Bible, protected Source Canon, PillowTales Living World
        Continuity, and age-eligible canon stories.

        The repository remains world-agnostic: all Story Worlds are resolved
        from data only.
        """
        world = self.get_published_world(slug)
        if not world:
            return None

        world_id = str(world['id'])
        language = (language_code or 'en').strip().lower().split('-')[0]

        prompt_result = (
            self.client.table('story_world_prompt_packs')
            .select('*')
            .eq('story_world_id', world_id)
            .eq('language_code', language)
            .eq('active', True)
            .eq('published', True)
            .order('version', desc=True)
            .limit(1)
            .execute()
        )
        if not prompt_result.data and language != 'en':
            prompt_result = (
                self.client.table('story_world_prompt_packs')
                .select('*')
                .eq('story_world_id', world_id)
                .eq('language_code', 'en')
                .eq('active', True)
                .eq('published', True)
                .order('version', desc=True)
                .limit(1)
                .execute()
            )

        dna_result = (
            self.client.table('story_world_dna').select('*')
            .eq('story_world_id', world_id).eq('active', True).eq('published', True)
            .order('version', desc=True).limit(1).execute()
        )
        editorial_result = (
            self.client.table('story_world_editorial_bibles').select('*')
            .eq('story_world_id', world_id).eq('active', True).eq('published', True)
            .order('version', desc=True).limit(1).execute()
        )

        source_canon_result = (
            self.client.table('story_world_source_canon').select('*')
            .eq('story_world_id', world_id)
            .eq('active', True)
            .eq('published', True)
            .order('version', desc=True)
            .limit(1)
            .execute()
        )

        continuity_result = (
            self.client.table('story_world_continuity').select('*')
            .eq('story_world_id', world_id)
            .eq('active', True)
            .eq('published', True)
            .order('version', desc=True)
            .limit(1)
            .execute()
        )

        canon_result = (
            self.client.table('story_world_canon_stories').select('*')
            .eq('story_world_id', world_id).eq('active', True).eq('published', True)
            .lte('age_min', age).gte('age_max', age)
            .order('slug').execute()
        )
        canon_rows = canon_result.data or []

        # Generation needs the same published display-title translation already
        # used by the Story Worlds catalogue. Keep the authoritative Canon row
        # untouched and attach the selected translation separately.
        if canon_rows:
            canon_story_ids = [
                str(row['id'])
                for row in canon_rows
                if row.get('id')
            ]
            requested_languages = [language] if language == 'en' else [language, 'en']

            translation_result = (
                self.client.table('story_world_canon_story_translations')
                .select('*')
                .in_('canon_story_id', canon_story_ids)
                .in_('language_code', requested_languages)
                .eq('published', True)
                .order('version', desc=True)
                .execute()
            )

            translations_by_story: dict[str, dict[str, dict]] = {}
            for translation in (translation_result.data or []):
                story_id = str(translation.get('canon_story_id') or '')
                translation_language = (
                    str(translation.get('language_code') or '')
                    .strip()
                    .lower()
                    .replace('_', '-')
                    .split('-', 1)[0]
                )
                if not story_id or not translation_language:
                    continue
                translations_by_story.setdefault(story_id, {})
                # Newest version is first because the query is version DESC.
                translations_by_story[story_id].setdefault(
                    translation_language,
                    translation,
                )

            enriched_canon_rows: list[dict] = []
            for row in canon_rows:
                item = dict(row)
                story_id = str(row.get('id') or '')
                by_language = translations_by_story.get(story_id, {})
                item['_story_translation'] = (
                    by_language.get(language)
                    or by_language.get('en')
                )
                enriched_canon_rows.append(item)

            canon_rows = enriched_canon_rows

        # Random selection is done in the service so repository output remains deterministic.
        return {
            'world': world,
            'prompt_pack': (prompt_result.data or [None])[0],
            'story_dna': (dna_result.data or [None])[0],
            'editorial_bible': (editorial_result.data or [None])[0],
            'source_canon': (source_canon_result.data or [None])[0],
            'living_world_continuity': (continuity_result.data or [None])[0],
            'canon_stories': canon_rows,
            'mode': mode,
        }

