# Story Worlds Sprint 2 — Installation

This package adds the Story World foundation and public read APIs. It does not alter story generation or narration.

## 1. Run the SQL migration

In the **StoryWorlds_dev** Supabase SQL Editor, run:

`supabase/migrations/20260805_001_story_worlds_foundation.sql`

The migration creates only new Story World tables. It does not add Ireland or any other Story World yet.

## 2. Copy the backend files

### New files

- `app/models/story_world.py`
- `app/repositories/story_world_repository.py`
- `app/services/story_world_service.py`
- `app/api/routes/story_worlds.py`
- `supabase/migrations/20260805_001_story_worlds_foundation.sql`

### Existing files replaced with the supplied updated versions

- `app/api/deps.py`
- `app/main.py`

Those two existing files contain small additive changes only: dependency registration and router registration.

## 3. Restart the backend

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 4. Verify the API

Until Ireland is seeded, these should return an empty successful response:

- `GET /api/story-worlds`
- `GET /api/story-worlds?language=es`
- `GET /api/story-worlds/featured?month=3`

Expected shape:

```json
{
  "storyWorlds": [],
  "count": 0
}
```

This proves the Story World platform is installed without affecting the existing story flow.

## Not changed

- `story_service.py`
- `stories.py`
- narration code
- chunking
- reader/playback
- existing `stories` table
- frontend

Ireland is the next data/content step, not part of this foundation migration.
