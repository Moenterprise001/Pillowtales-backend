# Installation — Story Worlds Content Extensions

## What this step does

Adds two new backend-managed data structures:

- Canon story records
- Pronunciation records

It does not modify:

- `story_service.py`
- story generation
- narration generation
- chunking
- reader playback
- frontend
- existing stories

## Run the SQL

Run this file only in **StoryWorlds_dev**:

`supabase/migrations/20260805_002_story_world_content_extensions.sql`

Keep a copy in the backend repository under `supabase/migrations/`.

## Expected result

Supabase should create:

- `public.story_world_canon_stories`
- `public.story_world_pronunciations`

Ireland is not seeded or published by this migration.

The next step is the reviewed Irish content seed and public API verification.
