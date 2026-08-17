-- PillowTales Story Worlds content extensions v1.0
-- Additive only.
-- Adds canon and pronunciation data required by every Story World.
-- Does not modify story generation, narration, chunking, reader or playback.

create table if not exists public.story_world_canon_stories (
    id uuid primary key default gen_random_uuid(),
    story_world_id uuid not null references public.story_worlds(id) on delete cascade,
    slug text not null,
    version integer not null default 1 check (version > 0),
    title text not null,
    summary text not null,
    main_characters jsonb not null default '[]'::jsonb,
    locations jsonb not null default '[]'::jsonb,
    creatures jsonb not null default '[]'::jsonb,
    core_values jsonb not null default '[]'::jsonb,
    age_min integer not null default 3 check (age_min between 1 and 12),
    age_max integer not null default 12 check (age_max between 1 and 12 and age_max >= age_min),
    bedtime_adaptation text,
    companion_allowed boolean not null default true,
    living_world_expansion_allowed boolean not null default true,
    continuation_allowed boolean not null default false,
    active boolean not null default false,
    published boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (story_world_id, slug, version),
    constraint story_world_canon_main_characters_array
        check (jsonb_typeof(main_characters) = 'array'),
    constraint story_world_canon_locations_array
        check (jsonb_typeof(locations) = 'array'),
    constraint story_world_canon_creatures_array
        check (jsonb_typeof(creatures) = 'array'),
    constraint story_world_canon_core_values_array
        check (jsonb_typeof(core_values) = 'array')
);

create table if not exists public.story_world_pronunciations (
    id uuid primary key default gen_random_uuid(),
    story_world_id uuid not null references public.story_worlds(id) on delete cascade,
    language_code varchar(10) not null default 'en',
    display_text text not null,
    normalized_key text not null,
    ipa text,
    phonetic_hint text,
    provider_overrides jsonb not null default '{}'::jsonb,
    usage_notes text,
    verified_by text,
    verified_at timestamptz,
    active boolean not null default true,
    version integer not null default 1 check (version > 0),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (story_world_id, language_code, normalized_key, version),
    constraint story_world_pronunciation_provider_overrides_object
        check (jsonb_typeof(provider_overrides) = 'object')
);

create unique index if not exists uq_story_world_active_canon_story
    on public.story_world_canon_stories(story_world_id, slug)
    where active = true;

create unique index if not exists uq_story_world_active_pronunciation
    on public.story_world_pronunciations(story_world_id, language_code, normalized_key)
    where active = true;

create index if not exists idx_story_world_canon_public_lookup
    on public.story_world_canon_stories(story_world_id, active, published, age_min, age_max);

create index if not exists idx_story_world_pronunciation_lookup
    on public.story_world_pronunciations(story_world_id, language_code, active);

alter table public.story_world_canon_stories enable row level security;
alter table public.story_world_pronunciations enable row level security;

-- Canon and pronunciation content are backend/admin-managed.
drop policy if exists story_world_canon_service_role_all
    on public.story_world_canon_stories;
create policy story_world_canon_service_role_all
on public.story_world_canon_stories
for all to service_role
using (true)
with check (true);

drop policy if exists story_world_pronunciations_service_role_all
    on public.story_world_pronunciations;
create policy story_world_pronunciations_service_role_all
on public.story_world_pronunciations
for all to service_role
using (true)
with check (true);

grant all privileges
on public.story_world_canon_stories
to service_role;

grant all privileges
on public.story_world_pronunciations
to service_role;
