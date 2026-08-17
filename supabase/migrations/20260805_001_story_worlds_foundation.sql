-- PillowTales Story Worlds foundation v1.0
-- Additive only. Does not modify generation, narration, chunking, reader or playback.

create extension if not exists pgcrypto;

create table if not exists public.story_worlds (
    id uuid primary key default gen_random_uuid(),
    slug text not null unique,
    category text not null,
    world_type text not null check (world_type in ('living', 'legendary', 'seasonal', 'original', 'hybrid')),
    region_code text not null,
    region_name text not null,
    countries jsonb not null default '[]'::jsonb,
    peoples jsonb not null default '[]'::jsonb,
    traditions jsonb not null default '[]'::jsonb,
    age_min integer not null default 1 check (age_min between 1 and 12),
    age_max integer not null default 12 check (age_max between 1 and 12 and age_max >= age_min),
    living_world_weight integer not null default 70 check (living_world_weight between 0 and 100),
    companion_weight integer not null default 20 check (companion_weight between 0 and 100),
    legendary_moment_weight integer not null default 10 check (legendary_moment_weight between 0 and 100),
    cover_url text,
    thumbnail_url text,
    icon_url text,
    primary_colour text,
    secondary_colour text,
    sort_order integer not null default 100,
    enabled boolean not null default false,
    published boolean not null default false,
    coming_soon boolean not null default false,
    version integer not null default 1 check (version > 0),
    archived_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint story_world_story_mix_total check (
        living_world_weight + companion_weight + legendary_moment_weight = 100
    ),
    constraint story_world_countries_array check (jsonb_typeof(countries) = 'array'),
    constraint story_world_peoples_array check (jsonb_typeof(peoples) = 'array'),
    constraint story_world_traditions_array check (jsonb_typeof(traditions) = 'array')
);

create table if not exists public.story_world_translations (
    id uuid primary key default gen_random_uuid(),
    story_world_id uuid not null references public.story_worlds(id) on delete cascade,
    language_code varchar(10) not null,
    name text not null,
    short_description text not null,
    description text not null,
    display_labels jsonb not null default '{}'::jsonb,
    published boolean not null default false,
    version integer not null default 1 check (version > 0),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (story_world_id, language_code, version)
);

create table if not exists public.story_world_monthly_weights (
    id uuid primary key default gen_random_uuid(),
    story_world_id uuid not null references public.story_worlds(id) on delete cascade,
    month integer not null check (month between 1 and 12),
    weight integer not null default 0 check (weight between 0 and 1000),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (story_world_id, month)
);

create table if not exists public.story_world_editorial_bibles (
    id uuid primary key default gen_random_uuid(),
    story_world_id uuid not null references public.story_worlds(id) on delete cascade,
    version integer not null check (version > 0),
    content jsonb not null,
    active boolean not null default false,
    published boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (story_world_id, version)
);

create table if not exists public.story_world_dna (
    id uuid primary key default gen_random_uuid(),
    story_world_id uuid not null references public.story_worlds(id) on delete cascade,
    version integer not null check (version > 0),
    content jsonb not null,
    active boolean not null default false,
    published boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (story_world_id, version)
);

create table if not exists public.story_world_prompt_packs (
    id uuid primary key default gen_random_uuid(),
    story_world_id uuid not null references public.story_worlds(id) on delete cascade,
    language_code varchar(10) not null,
    version integer not null check (version > 0),
    content jsonb not null,
    active boolean not null default false,
    published boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (story_world_id, language_code, version)
);

create unique index if not exists uq_story_world_active_editorial_bible
    on public.story_world_editorial_bibles(story_world_id)
    where active = true;

create unique index if not exists uq_story_world_active_dna
    on public.story_world_dna(story_world_id)
    where active = true;

create unique index if not exists uq_story_world_active_prompt_pack_language
    on public.story_world_prompt_packs(story_world_id, language_code)
    where active = true;

create index if not exists idx_story_worlds_public_listing
    on public.story_worlds(enabled, published, coming_soon, region_code, category, sort_order);

create index if not exists idx_story_world_translations_lookup
    on public.story_world_translations(story_world_id, language_code, published);

create index if not exists idx_story_world_monthly_weights_lookup
    on public.story_world_monthly_weights(month, weight desc, story_world_id);

create index if not exists idx_story_world_prompt_packs_lookup
    on public.story_world_prompt_packs(story_world_id, language_code, active, published);

alter table public.story_worlds enable row level security;
alter table public.story_world_translations enable row level security;
alter table public.story_world_monthly_weights enable row level security;
alter table public.story_world_editorial_bibles enable row level security;
alter table public.story_world_dna enable row level security;
alter table public.story_world_prompt_packs enable row level security;

-- Public app clients may read only published presentation data.
drop policy if exists story_worlds_public_select on public.story_worlds;
create policy story_worlds_public_select on public.story_worlds
for select to public using (
    enabled = true and published = true and archived_at is null
);

drop policy if exists story_world_translations_public_select on public.story_world_translations;
create policy story_world_translations_public_select on public.story_world_translations
for select to public using (
    published = true
    and exists (
        select 1 from public.story_worlds world
        where world.id = story_world_translations.story_world_id
          and world.enabled = true
          and world.published = true
          and world.archived_at is null
    )
);

drop policy if exists story_world_weights_public_select on public.story_world_monthly_weights;
create policy story_world_weights_public_select on public.story_world_monthly_weights
for select to public using (
    exists (
        select 1 from public.story_worlds world
        where world.id = story_world_monthly_weights.story_world_id
          and world.enabled = true
          and world.published = true
          and world.archived_at is null
    )
);

-- Editorial Bible, Story DNA and prompt content remain backend/admin only.
drop policy if exists story_worlds_service_role_all on public.story_worlds;
create policy story_worlds_service_role_all on public.story_worlds
for all to service_role using (true) with check (true);

drop policy if exists story_world_translations_service_role_all on public.story_world_translations;
create policy story_world_translations_service_role_all on public.story_world_translations
for all to service_role using (true) with check (true);

drop policy if exists story_world_weights_service_role_all on public.story_world_monthly_weights;
create policy story_world_weights_service_role_all on public.story_world_monthly_weights
for all to service_role using (true) with check (true);

drop policy if exists story_world_editorial_service_role_all on public.story_world_editorial_bibles;
create policy story_world_editorial_service_role_all on public.story_world_editorial_bibles
for all to service_role using (true) with check (true);

drop policy if exists story_world_dna_service_role_all on public.story_world_dna;
create policy story_world_dna_service_role_all on public.story_world_dna
for all to service_role using (true) with check (true);

drop policy if exists story_world_prompt_packs_service_role_all on public.story_world_prompt_packs;
create policy story_world_prompt_packs_service_role_all on public.story_world_prompt_packs
for all to service_role using (true) with check (true);

grant usage on schema public to anon, authenticated, service_role;
grant select on public.story_worlds to anon, authenticated;
grant select on public.story_world_translations to anon, authenticated;
grant select on public.story_world_monthly_weights to anon, authenticated;
grant all privileges on public.story_worlds to service_role;
grant all privileges on public.story_world_translations to service_role;
grant all privileges on public.story_world_monthly_weights to service_role;
grant all privileges on public.story_world_editorial_bibles to service_role;
grant all privileges on public.story_world_dna to service_role;
grant all privileges on public.story_world_prompt_packs to service_role;
