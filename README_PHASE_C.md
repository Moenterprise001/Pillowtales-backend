# PillowTales Phase C multi-file backend

This package splits the cleaned backend into a maintainable structure.

## Structure
- `app/main.py` - FastAPI entrypoint
- `app/api/routes/` - thin API routes
- `app/services/` - business logic
- `app/repositories/` - Supabase data access
- `app/models/` - request/response models
- `app/domain/constants.py` - language, narrator, companion, and plan configuration
- `app/utils/story_text.py` - cleanup and preview helpers

## Why this helps
- narration becomes easier to speed up because the route now calls a dedicated `NarrationService`
- adding languages is easier because supported languages and narrator defaults are centralized
- story generation, auth, subscription, and persistence are separated so changes are safer

## Step-by-step rollout
1. Replace your current backend branch with this structure.
2. Set the same `.env` values you use today.
3. Start the app with `uvicorn app.main:app --reload` from the `backend/` folder.
4. Test login, story generation, story list/detail, settings, subscription, and preview.
5. Wire `NarrationService.enqueue_job()` to Celery/RQ/Cloud Tasks next.
6. Move TTS generation into a dedicated worker process.
7. Add Redis for queue state and caching once stable.

## Fast language expansion
1. Add the code/name in `app/domain/constants.py -> SUPPORTED_LANGUAGES`
2. Add a narrator preset in `VOICE_PRESETS`
3. Add a default mapping in `NarrationService.resolve_voice()`
4. Update prompt rules in `StoryService` if needed for locale tone

## Current status
- core story/auth/subscription flows are organized
- narration is queue-ready, not worker-complete
- this is the right foundation for the next speed/scaling step
