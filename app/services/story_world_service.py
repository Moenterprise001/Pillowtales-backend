from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

from app.models.story_world import (
    StoryWorldAgeRange,
    StoryWorldAdventureListResponse,
    StoryWorldAdventureSource,
    StoryWorldArtwork,
    StoryWorldCanonCollection,
    StoryWorldCanonCountry,
    StoryWorldCanonSeries,
    StoryWorldCanonStoryListResponse,
    StoryWorldCanonStorySource,
    StoryWorldAvailability,
    StoryWorldCountry,
    StoryWorldPresentation,
    StoryWorldPublic,
    StoryWorldRegion,
)
from app.repositories.story_world_repository import StoryWorldRepository

SUPPORTED_STORY_WORLD_LANGUAGES = ('en', 'es', 'fr', 'de', 'it', 'ja', 'ar')


class StoryWorldService:
    def __init__(self, repository: StoryWorldRepository):
        self.repository = repository

    def list_public(
        self,
        *,
        language: str = 'en',
        region: Optional[str] = None,
        category: Optional[str] = None,
        age: Optional[int] = None,
    ) -> list[StoryWorldPublic]:
        language = self._normalise_language(language)
        worlds = self.repository.list_published_worlds(region=region, category=category, age=age)
        return self._build_public_worlds(worlds, language)

    def get_public(self, slug: str, *, language: str = 'en') -> StoryWorldPublic:
        language = self._normalise_language(language)
        world = self.repository.get_published_world(slug.strip().lower())
        if not world:
            raise HTTPException(status_code=404, detail='Story World not found')
        public_worlds = self._build_public_worlds([world], language)
        if not public_worlds:
            raise HTTPException(
                status_code=404,
                detail='Story World is not available in the requested language',
            )
        return public_worlds[0]

    def list_original_stories(
        self,
        slug: str,
        *,
        age: int,
        language: str = 'en',
    ) -> StoryWorldCanonStoryListResponse:
        language = self._normalise_language(language)
        world, rows = self.repository.list_original_folk_stories(
            slug=str(slug or '').strip().lower(),
            age=age,
            language_code=language,
        )
        if not world:
            raise HTTPException(status_code=404, detail='Story World not found')

        # Country metadata is optional and data-driven. Worlds without country
        # grouping keep the existing flat catalogue behaviour unchanged.
        world_countries = [
            item for item in (world.get('countries') or [])
            if isinstance(item, dict) and item.get('code') and item.get('name')
        ]
        country_by_name = {
            str(item['name']).strip().casefold(): item
            for item in world_countries
        }

        stories: list[StoryWorldCanonStorySource] = []
        for row in rows:
            artwork_data = row.get('artwork') or {}
            if not isinstance(artwork_data, dict):
                artwork_data = {}

            story_translation = row.get('_story_translation') or {}
            if not isinstance(story_translation, dict):
                story_translation = {}

            collection_model = None
            collection = row.get('_collection')
            if isinstance(collection, dict):
                collection_translation = row.get('_collection_translation') or {}
                if not isinstance(collection_translation, dict):
                    collection_translation = {}
                collection_artwork = collection.get('artwork') or {}
                if not isinstance(collection_artwork, dict):
                    collection_artwork = {}

                collection_model = StoryWorldCanonCollection(
                    slug=str(collection['slug']),
                    title=str(collection_translation.get('title') or collection['title']),
                    description=(
                        str(
                            collection_translation.get('description')
                            or collection.get('description')
                            or ''
                        )
                        or None
                    ),
                    sortOrder=int(collection.get('sort_order') or 100),
                    artwork=StoryWorldArtwork(
                        coverUrl=collection_artwork.get('hero_url') or collection_artwork.get('cover_url'),
                        thumbnailUrl=collection_artwork.get('thumbnail_url'),
                        iconUrl=collection_artwork.get('icon_url'),
                    ),
                )

            series_model = None
            series = row.get('_series')
            if isinstance(series, dict):
                series_translation = row.get('_series_translation') or {}
                if not isinstance(series_translation, dict):
                    series_translation = {}
                series_artwork = series.get('artwork') or {}
                if not isinstance(series_artwork, dict):
                    series_artwork = {}

                series_model = StoryWorldCanonSeries(
                    slug=str(series['slug']),
                    title=str(series_translation.get('title') or series['title']),
                    summary=(
                        str(
                            series_translation.get('summary')
                            or series.get('summary')
                            or ''
                        )
                        or None
                    ),
                    partCount=int(series.get('part_count') or 0),
                    sortOrder=int(series.get('sort_order') or 100),
                    chronologyGroup=(
                        str(series.get('chronology_group'))
                        if series.get('chronology_group') is not None
                        else None
                    ),
                    chronologyOrder=(
                        int(series['chronology_order'])
                        if series.get('chronology_order') is not None
                        else None
                    ),
                    artwork=StoryWorldArtwork(
                        coverUrl=series_artwork.get('hero_url') or series_artwork.get('cover_url'),
                        thumbnailUrl=series_artwork.get('thumbnail_url'),
                        iconUrl=series_artwork.get('icon_url'),
                    ),
                )

            generation_rules = row.get('generation_rules') or {}
            country_name = (
                str(generation_rules.get('country') or '').strip()
                if isinstance(generation_rules, dict)
                else ''
            )
            country_data = country_by_name.get(country_name.casefold()) if country_name else None
            country_model = (
                StoryWorldCountry(
                    code=str(country_data.get('code') or ''),
                    name=str(country_data.get('name') or country_name),
                    heroUrl=(
                        str(country_data.get('hero_url'))
                        if country_data.get('hero_url')
                        else None
                    ),
                )
                if country_data
                else None
            )

            stories.append(
                StoryWorldCanonStorySource(
                    slug=str(row['slug']),
                    title=str(story_translation.get('title') or row['title']),
                    summary=str(story_translation.get('summary') or row.get('summary') or ''),
                    ageRange=StoryWorldAgeRange(min=int(row['age_min']), max=int(row['age_max'])),
                    artwork=StoryWorldArtwork(
                        coverUrl=artwork_data.get('hero_url') or artwork_data.get('cover_url'),
                        thumbnailUrl=artwork_data.get('thumbnail_url'),
                        iconUrl=artwork_data.get('icon_url'),
                    ),
                    coreValues=[str(item) for item in (row.get('core_values') or [])],
                    country=country_model,
                    collection=collection_model,
                    series=series_model,
                    partNumber=(
                        int(row['part_number'])
                        if row.get('part_number') is not None
                        else None
                    ),
                    partTitle=(
                        str(story_translation.get('subtitle') or row.get('part_title'))
                        if (story_translation.get('subtitle') or row.get('part_title')) is not None
                        else None
                    ),
                    sortOrder=int(row.get('sort_order') or 100),
                    chronologyGroup=(
                        str(row['chronology_group'])
                        if row.get('chronology_group') is not None
                        else None
                    ),
                    chronologyOrder=(
                        int(row['chronology_order'])
                        if row.get('chronology_order') is not None
                        else None
                    ),
                )
            )

        stories_by_country: dict[str, list[StoryWorldCanonStorySource]] = defaultdict(list)
        for story in stories:
            if story.country and story.country.name:
                stories_by_country[story.country.name.casefold()].append(story)

        country_groups: list[StoryWorldCanonCountry] = []
        for country_data in world_countries:
            country_name = str(country_data.get('name') or '').strip()
            grouped_stories = stories_by_country.get(country_name.casefold(), [])
            if not grouped_stories:
                continue
            country_groups.append(
                StoryWorldCanonCountry(
                    code=str(country_data.get('code') or ''),
                    name=country_name,
                    heroUrl=(
                        str(country_data.get('hero_url'))
                        if country_data.get('hero_url')
                        else None
                    ),
                    stories=grouped_stories,
                    count=len(grouped_stories),
                )
            )

        return StoryWorldCanonStoryListResponse(
            storyWorldSlug=str(world['slug']),
            stories=stories,
            countries=country_groups,
            count=len(stories),
        )

    def list_adventures(
        self,
        slug: str,
        *,
        age: int,
        language: str = 'en',
    ) -> StoryWorldAdventureListResponse:
        """Compatibility endpoint for Living World focus selection.

        The existing /adventures API and response model stay intact during the
        rollout so the current frontend does not break. Rows are now treated as
        optional source-canon/world-focus choices, not mandatory plot templates.
        """
        language = self._normalise_language(language)
        world, rows = self.repository.list_folk_adventure_sources(
            slug=str(slug or '').strip().lower(),
            age=age,
        )
        if not world:
            raise HTTPException(status_code=404, detail='Story World not found')

        adventures: list[StoryWorldAdventureSource] = []
        for row in rows:
            artwork_data = row.get('artwork') or {}
            if not isinstance(artwork_data, dict):
                artwork_data = {}
            generation_rules = row.get('generation_rules') or {}
            folk_rules = generation_rules.get('folk_adventure') if isinstance(generation_rules, dict) else {}
            presentation = folk_rules.get('presentation') if isinstance(folk_rules, dict) else {}
            if not isinstance(presentation, dict):
                presentation = {}
            language_presentation = presentation.get(language) or presentation.get('en') or {}
            if not isinstance(language_presentation, dict):
                language_presentation = {}

            adventures.append(
                StoryWorldAdventureSource(
                    slug=str(row['slug']),
                    title=str(language_presentation.get('title') or row['title']),
                    summary=str(
                        language_presentation.get('summary')
                        or 'A new PillowTales Living World story using this folklore as a possible world focus.'
                    ),
                    ageRange=StoryWorldAgeRange(min=int(row['age_min']), max=int(row['age_max'])),
                    artwork=StoryWorldArtwork(
                        coverUrl=artwork_data.get('hero_url') or artwork_data.get('cover_url'),
                        thumbnailUrl=artwork_data.get('thumbnail_url'),
                        iconUrl=artwork_data.get('icon_url'),
                    ),
                    coreValues=[str(item) for item in (row.get('core_values') or [])],
                )
            )

        return StoryWorldAdventureListResponse(
            storyWorldSlug=str(world['slug']),
            adventures=adventures,
            count=len(adventures),
        )

    def list_featured(
        self,
        *,
        language: str = 'en',
        month: Optional[int] = None,
        age: Optional[int] = None,
    ) -> list[StoryWorldPublic]:
        language = self._normalise_language(language)
        resolved_month = month or datetime.now(timezone.utc).month
        if not 1 <= resolved_month <= 12:
            raise HTTPException(status_code=422, detail='month must be between 1 and 12')

        worlds = self.repository.list_published_worlds(age=age)
        weights = self.repository.get_monthly_weights(
            [str(world['id']) for world in worlds],
            resolved_month,
        )
        weight_by_world = {str(row['story_world_id']): int(row['weight']) for row in weights}
        eligible = [world for world in worlds if weight_by_world.get(str(world['id']), 0) > 0]
        eligible.sort(
            key=lambda world: (
                -weight_by_world.get(str(world['id']), 0),
                int(world.get('sort_order') or 100),
                str(world.get('slug') or ''),
            )
        )
        return self._build_public_worlds(eligible, language)

    def _build_public_worlds(self, worlds: list[dict], language: str) -> list[StoryWorldPublic]:
        if not worlds:
            return []

        world_ids = [str(world['id']) for world in worlds]
        translations = self.repository.get_translations(world_ids)
        prompt_rows = self.repository.get_active_prompt_pack_languages(world_ids)

        translations_by_world: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in translations:
            translations_by_world[str(row['story_world_id'])][str(row['language_code']).lower()] = row

        prompt_languages_by_world: dict[str, set[str]] = defaultdict(set)
        for row in prompt_rows:
            prompt_languages_by_world[str(row['story_world_id'])].add(
                str(row['language_code']).lower()
            )

        response: list[StoryWorldPublic] = []
        for world in worlds:
            world_id = str(world['id'])
            prompt_languages = prompt_languages_by_world.get(world_id, set())
            selected_language = language if language in prompt_languages else 'en'
            if selected_language not in prompt_languages:
                continue

            world_translations = translations_by_world.get(world_id, {})
            translation = world_translations.get(selected_language) or world_translations.get('en')
            if not translation:
                continue

            response.append(self._to_public_model(world, translation, sorted(prompt_languages)))
        return response

    @staticmethod
    def _to_public_model(
        world: dict,
        translation: dict,
        supported_languages: list[str],
    ) -> StoryWorldPublic:
        countries = [
            StoryWorldCountry(
                code=str(item.get('code', '')),
                name=str(item.get('name', '')),
                heroUrl=(str(item.get('hero_url')) if item.get('hero_url') else None),
            )
            for item in (world.get('countries') or [])
            if isinstance(item, dict) and item.get('code') and item.get('name')
        ]
        return StoryWorldPublic(
            id=str(world['id']),
            slug=str(world['slug']),
            name=str(translation['name']),
            shortDescription=str(translation.get('short_description') or ''),
            description=str(translation.get('description') or ''),
            region=StoryWorldRegion(
                code=str(world['region_code']),
                name=str(world['region_name']),
            ),
            countries=countries,
            peoples=[str(item) for item in (world.get('peoples') or [])],
            traditions=[str(item) for item in (world.get('traditions') or [])],
            category=str(world['category']),
            worldType=str(world['world_type']),
            ageRange=StoryWorldAgeRange(
                min=int(world['age_min']),
                max=int(world['age_max']),
            ),
            artwork=StoryWorldArtwork(
                coverUrl=world.get('cover_url'),
                thumbnailUrl=world.get('thumbnail_url'),
                iconUrl=world.get('icon_url'),
            ),
            presentation=StoryWorldPresentation(
                primaryColour=world.get('primary_colour'),
                secondaryColour=world.get('secondary_colour'),
                sortOrder=int(world.get('sort_order') or 100),
            ),
            availability=StoryWorldAvailability(
                enabled=bool(world.get('enabled')),
                published=bool(world.get('published')),
                comingSoon=bool(world.get('coming_soon')),
            ),
            supportedLanguages=supported_languages,
            version=int(world.get('version') or 1),
            updatedAt=str(world['updated_at']),
        )

    @staticmethod
    def _normalise_language(language: str) -> str:
        code = (language or 'en').strip().lower().split('-')[0]
        return code if code in SUPPORTED_STORY_WORLD_LANGUAGES else 'en'
