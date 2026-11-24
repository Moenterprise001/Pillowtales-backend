# PillowTales backend: rules for anyone changing this code

## Bedtime Stories are locked (owner's rule, 24 Nov 2026)

Bedtime Stories, Story Worlds and seasonal stories share code (`app/services/story_service.py`,
`story_quality.py`, `narration_service.py`, `text_cleaner.py` and others). Bedtime Stories have been
signed off by the owner, so **no Story Worlds, seasonal or other change may alter them without the
owner agreeing first.**

1. **Before you change any shared file, and again before you commit**, run both locks from the
   backend folder:

       python tests/test_bedtime_lock.py .
       python tests/test_bedtime_narration_lock.py .

2. **If either says FAILED, stop.** Don't deploy and don't "fix" the lock. Tell the owner, in plain
   English, which Bedtime stories or narration would change and why. Then wait for an answer.
   The owner can pause the change.
3. **Never run `--update` on a lock** unless the owner has signed off that exact Bedtime change. This
   is the only way the locks are meant to change.
4. Render runs both locks in its Build Command. A change that moves Bedtime fails the build. The
   live server then keeps running the signed-off version, and Render emails that the deploy failed.
   Don't remove the locks from the Build Command.

## Old Bedtime versions can't be switched on

Bedtime always runs the signed-off version, **v8e** (`bedtime_v8e=20261122`). The old Render
settings `BEDTIME_V8`, `BEDTIME_V8D`, `BEDTIME_V8E`, `BEDTIME_PROMPTS_V5`,
`BEDTIME_PLAN_CHECK_ENABLED`, `BEDTIME_DRAFT_REVISE`, `BEDTIME_STORY_PLAN_ENABLED` and
`BEDTIME_STORY_SCORE_ENABLED` are ignored. Pages 3-7 use `gemini-3.1-pro-preview` unless
`GEMINI_DRAFT_MODEL` names another model.

- Don't add new settings that can switch Bedtime back to older rules.
- The old code paths are still in the files, but servers can't reach them. Only the offline tests
  can, through `story_quality._ALLOW_OLD_VERSIONS`, which is set in `tests/harness.py`. Never set it
  in app code.

## Story types

- **Bedtime Stories**: ordinary stories, ages 3+ (v8e). Ages 0-2 have their own path (to be
  redesigned in the new year).
- **Story Worlds and seasonal stories** use the `canon` / `folk_adventure` modes and never use the
  Bedtime v8 code.
