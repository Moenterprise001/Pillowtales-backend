-- PillowTales Story Worlds
-- Migration: 20260805_003_seed_irish_legends_catalog.sql
-- Target: StoryWorlds_dev ONLY
--
-- Purpose:
--   1. Publish Irish Legends as the first catalogue-visible Story World.
--   2. Add EN/ES/FR/DE/IT presentation translations.
--   3. Add monthly weights used only by Tonight's Surprise Story.
--   4. Add initial internal editorial/DNA records.
--   5. Add small catalogue-ready language packs so the existing API exposes
--      the world in all five supported languages.
--
-- This migration does NOT:
--   - change story_service.py
--   - connect Story Worlds to generation
--   - add canon stories
--   - add unverified pronunciation records
--   - change narration, chunking, reader, feedback, share cards or frontend
--
-- Idempotent: safe to run again. Existing v1 rows are updated rather than duplicated.

begin;

-- ---------------------------------------------------------------------------
-- 1. Story World master record
-- ---------------------------------------------------------------------------

insert into public.story_worlds (
    slug,
    category,
    world_type,
    region_code,
    region_name,
    countries,
    peoples,
    traditions,
    age_min,
    age_max,
    living_world_weight,
    companion_weight,
    legendary_moment_weight,
    cover_url,
    thumbnail_url,
    icon_url,
    primary_colour,
    secondary_colour,
    sort_order,
    enabled,
    published,
    coming_soon,
    version,
    archived_at
)
values (
    'irish-legends',
    'folklore',
    'hybrid',
    'europe',
    'Europe',
    '[{"code":"IE","name":"Ireland"}]'::jsonb,
    '["Irish"]'::jsonb,
    '[
      "Irish mythology",
      "Irish folklore",
      "Fenian Cycle",
      "selected Mythological Cycle traditions"
    ]'::jsonb,
    1,
    12,
    70,
    20,
    10,
    null,
    null,
    null,
    '#2F6B4F',
    '#D6A94A',
    10,
    true,
    true,
    false,
    1,
    null
)
on conflict (slug) do update set
    category = excluded.category,
    world_type = excluded.world_type,
    region_code = excluded.region_code,
    region_name = excluded.region_name,
    countries = excluded.countries,
    peoples = excluded.peoples,
    traditions = excluded.traditions,
    age_min = excluded.age_min,
    age_max = excluded.age_max,
    living_world_weight = excluded.living_world_weight,
    companion_weight = excluded.companion_weight,
    legendary_moment_weight = excluded.legendary_moment_weight,
    primary_colour = excluded.primary_colour,
    secondary_colour = excluded.secondary_colour,
    sort_order = excluded.sort_order,
    enabled = excluded.enabled,
    published = excluded.published,
    coming_soon = excluded.coming_soon,
    version = excluded.version,
    archived_at = null,
    updated_at = now();

-- ---------------------------------------------------------------------------
-- 2. Presentation translations
-- ---------------------------------------------------------------------------

with world as (
    select id
    from public.story_worlds
    where slug = 'irish-legends'
)
insert into public.story_world_translations (
    story_world_id,
    language_code,
    name,
    short_description,
    description,
    display_labels,
    published,
    version
)
select
    world.id,
    values_to_insert.language_code,
    values_to_insert.name,
    values_to_insert.short_description,
    values_to_insert.description,
    values_to_insert.display_labels,
    true,
    1
from world
cross join (
    values
    (
        'en',
        'Irish Legends',
        'Magical adventures inspired by Ireland''s folklore, legends and myths.',
        'Journey through ancient forests, misty hills and legendary places in original bedtime adventures inspired by Ireland''s storytelling traditions.',
        '{"region":"Europe","country":"Ireland","collection":"Folklore, legends and myths"}'::jsonb
    ),
    (
        'es',
        'Leyendas de Irlanda',
        'Aventuras mágicas inspiradas en el folclore, las leyendas y los mitos de Irlanda.',
        'Viaja por bosques antiguos, colinas cubiertas de niebla y lugares legendarios en aventuras originales para dormir inspiradas en las tradiciones narrativas de Irlanda.',
        '{"region":"Europa","country":"Irlanda","collection":"Folclore, leyendas y mitos"}'::jsonb
    ),
    (
        'fr',
        'Légendes d''Irlande',
        'Des aventures magiques inspirées du folklore, des légendes et des mythes d''Irlande.',
        'Partez à travers des forêts anciennes, des collines brumeuses et des lieux légendaires dans des aventures originales du soir inspirées des traditions narratives irlandaises.',
        '{"region":"Europe","country":"Irlande","collection":"Folklore, légendes et mythes"}'::jsonb
    ),
    (
        'de',
        'Irische Legenden',
        'Magische Abenteuer, inspiriert von Irlands Folklore, Legenden und Mythen.',
        'Reise durch alte Wälder, neblige Hügel und sagenhafte Orte in neuen Gute-Nacht-Abenteuern, die von Irlands Erzähltraditionen inspiriert sind.',
        '{"region":"Europa","country":"Irland","collection":"Folklore, Legenden und Mythen"}'::jsonb
    ),
    (
        'it',
        'Leggende d''Irlanda',
        'Avventure magiche ispirate al folklore, alle leggende e ai miti d''Irlanda.',
        'Viaggia tra antiche foreste, colline avvolte nella nebbia e luoghi leggendari in originali avventure della buonanotte ispirate alle tradizioni narrative irlandesi.',
        '{"region":"Europa","country":"Irlanda","collection":"Folklore, leggende e miti"}'::jsonb
    )
) as values_to_insert(
    language_code,
    name,
    short_description,
    description,
    display_labels
)
on conflict (story_world_id, language_code, version) do update set
    name = excluded.name,
    short_description = excluded.short_description,
    description = excluded.description,
    display_labels = excluded.display_labels,
    published = excluded.published,
    updated_at = now();

-- ---------------------------------------------------------------------------
-- 3. Monthly weights
--
-- Discover Story Worlds ignores these weights; Ireland remains available all year.
-- These values influence only automatic/featured selection.
-- March receives the strongest weight because of St Patrick's Day.
-- ---------------------------------------------------------------------------

with world as (
    select id
    from public.story_worlds
    where slug = 'irish-legends'
)
insert into public.story_world_monthly_weights (
    story_world_id,
    month,
    weight
)
select
    world.id,
    month_weights.month,
    month_weights.weight
from world
cross join (
    values
        (1, 10),
        (2, 15),
        (3, 100),
        (4, 20),
        (5, 20),
        (6, 15),
        (7, 15),
        (8, 15),
        (9, 20),
        (10, 25),
        (11, 15),
        (12, 15)
) as month_weights(month, weight)
on conflict (story_world_id, month) do update set
    weight = excluded.weight,
    updated_at = now();

-- ---------------------------------------------------------------------------
-- 4. Initial internal Editorial Bible
--
-- This is intentionally concise. The full human Story World Bible remains in
-- the repository and will be compiled into a production prompt pack later.
-- ---------------------------------------------------------------------------

with world as (
    select id
    from public.story_worlds
    where slug = 'irish-legends'
)
insert into public.story_world_editorial_bibles (
    story_world_id,
    version,
    content,
    active,
    published
)
select
    world.id,
    1,
    '{
      "status": "draft_reference",
      "identity_statement": "Ireland is a land where stories are treasured, music travels on the wind, ancient places hold quiet magic, kindness and courage matter, and every hill, lake, forest and shore may hide another adventure.",
      "core_principle": "Respect the canon. Expand the world.",
      "tone": [
        "warm",
        "gentle",
        "curious",
        "magical",
        "hopeful",
        "bedtime-ready"
      ],
      "values": [
        "kindness",
        "hospitality",
        "courage",
        "wisdom",
        "friendship",
        "community",
        "storytelling",
        "music",
        "respect for nature"
      ],
      "narrative_rules": {
        "prior_knowledge_required": false,
        "child_replaces_legendary_hero": false,
        "direct_questions_to_child": false,
        "invented_material_claimed_as_canon": false
      },
      "cultural_boundaries": [
        "Do not reduce Ireland to leprechauns, shamrocks and pots of gold.",
        "Do not mock Irish names, accents or language.",
        "Do not turn political or historical conflict into bedtime content.",
        "Do not present invented additions as authentic canon."
      ],
      "generation_ready": false,
      "review_required": true
    }'::jsonb,
    true,
    false
from world
on conflict (story_world_id, version) do update set
    content = excluded.content,
    active = excluded.active,
    published = excluded.published,
    updated_at = now();

-- ---------------------------------------------------------------------------
-- 5. Initial internal Story DNA
-- ---------------------------------------------------------------------------

with world as (
    select id
    from public.story_worlds
    where slug = 'irish-legends'
)
insert into public.story_world_dna (
    story_world_id,
    version,
    content,
    active,
    published
)
select
    world.id,
    1,
    '{
      "status": "draft_reference",
      "story_mix": {
        "living_world": 70,
        "companion": 20,
        "canon_or_legendary_moment": 10
      },
      "openings": [
        "helping in a village",
        "preparing for a celebration",
        "walking beside a river",
        "hearing harp music",
        "receiving a letter",
        "seeing mist gather around an old place"
      ],
      "triggers": [
        "a lost object",
        "a creature asking for help",
        "a forgotten melody",
        "an unexpected path",
        "a glowing fairy ring",
        "a white horse appearing at dawn"
      ],
      "challenges": [
        "finding",
        "returning",
        "caring",
        "preparing",
        "listening",
        "guiding",
        "solving a gentle mystery"
      ],
      "endings": [
        "return home",
        "shared food",
        "quiet music",
        "gratitude",
        "firelight",
        "moonrise",
        "stars over the hills"
      ],
      "generation_ready": false,
      "review_required": true
    }'::jsonb,
    true,
    false
from world
on conflict (story_world_id, version) do update set
    content = excluded.content,
    active = excluded.active,
    published = excluded.published,
    updated_at = now();

-- ---------------------------------------------------------------------------
-- 6. Catalogue-ready language packs
--
-- The current API uses active, published prompt-pack rows to determine which
-- languages a Story World can be shown in. These compact rows make Ireland
-- visible in EN/ES/FR/DE/IT without connecting Story Worlds to generation.
--
-- "generation_ready": false is deliberate.
-- ---------------------------------------------------------------------------

with world as (
    select id
    from public.story_worlds
    where slug = 'irish-legends'
)
insert into public.story_world_prompt_packs (
    story_world_id,
    language_code,
    version,
    content,
    active,
    published
)
select
    world.id,
    language_codes.language_code,
    1,
    jsonb_build_object(
        'status', 'catalogue_ready',
        'generation_ready', false,
        'story_world_slug', 'irish-legends',
        'language_code', language_codes.language_code,
        'note', 'Catalogue availability record only. Full compiled generation prompt pack not yet approved.'
    ),
    true,
    true
from world
cross join (
    values ('en'), ('es'), ('fr'), ('de'), ('it')
) as language_codes(language_code)
on conflict (story_world_id, language_code, version) do update set
    content = excluded.content,
    active = excluded.active,
    published = excluded.published,
    updated_at = now();

commit;

-- ---------------------------------------------------------------------------
-- Verification query
-- ---------------------------------------------------------------------------

select
    sw.slug,
    sw.enabled,
    sw.published,
    sw.world_type,
    sw.living_world_weight,
    sw.companion_weight,
    sw.legendary_moment_weight,
    array_agg(distinct swt.language_code order by swt.language_code) as presentation_languages,
    array_agg(distinct swpp.language_code order by swpp.language_code) as catalogue_languages
from public.story_worlds sw
left join public.story_world_translations swt
    on swt.story_world_id = sw.id
   and swt.published = true
left join public.story_world_prompt_packs swpp
    on swpp.story_world_id = sw.id
   and swpp.active = true
   and swpp.published = true
where sw.slug = 'irish-legends'
group by
    sw.slug,
    sw.enabled,
    sw.published,
    sw.world_type,
    sw.living_world_weight,
    sw.companion_weight,
    sw.legendary_moment_weight;
