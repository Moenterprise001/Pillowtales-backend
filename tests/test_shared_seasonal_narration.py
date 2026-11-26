"""Shared Seasonal narration (26 Nov 2026): offline checks.

    python tests/test_shared_seasonal_narration.py <backend folder>

Nothing is sent anywhere: OpenAI and Supabase storage are replaced by fakes.
"""
import asyncio, os, sys, types as pytypes
sys.path.insert(0, os.path.dirname(__file__))
try:
    import httpx  # noqa: F401
except ImportError:
    sys.modules["httpx"] = pytypes.ModuleType("httpx")
from harness import *  # noqa: E402,F401  (stubs fastapi / supabase / google)
sys.modules["fastapi"].BackgroundTasks = getattr(sys.modules["fastapi"], "BackgroundTasks", object)
from app.services import narration_service as N  # noqa: E402
from app.services import shared_narration as S  # noqa: E402

FAKE_MP3 = (bytes([0xFF, 0xF3, 0xE4, 0xC0]) + bytes(476)) * 10
THEME = "seasonal:christmas:the-little-fir"
PAGE = "The little fir tree shivered in the snow. Then the stars came out, one by one."


class FakeBucket:
    def __init__(self, store, fail_download=False):
        self.store, self.fail_download = store, fail_download

    def download(self, path):
        if self.fail_download:
            raise RuntimeError("storage down")
        if path not in self.store:
            raise RuntimeError("Object not found")
        return self.store[path]

    def upload(self, path, data, options=None):
        self.store[path] = bytes(data)


class FakeStorage:
    def __init__(self, fail_download=False):
        self.store, self.fail_download = {}, fail_download

    def from_(self, bucket):
        assert bucket == "story-audio"
        return FakeBucket(self.store, self.fail_download)


def make_service(storage):
    repo = pytypes.SimpleNamespace(client=pytypes.SimpleNamespace(storage=storage))
    svc = N.NarrationService(story_repo=repo, user_repo=None, subscription_service=None,
                             story_world_pronunciation_service=None)
    svc.tts_calls, svc.user_uploads = [], {}

    async def openai(text, voice, language_code=None, pronunciation_guide=None, bedtime_pacing=False):
        svc.tts_calls.append((voice, language_code, text, bedtime_pacing))
        return FAKE_MP3 + len(svc.tts_calls).to_bytes(2, "big")  # distinct bytes per generation

    async def eleven(text, voice_id, language_code):
        svc.tts_calls.append(("parent", voice_id, language_code, text))
        return FAKE_MP3

    async def upload(path, audio):
        svc.user_uploads[path] = audio

    async def translate(text, target, source=None):
        return text

    svc._generate_openai_tts, svc._generate_elevenlabs_tts = openai, eleven
    svc._upload_audio, svc._translate_text = upload, translate
    return svc


def play(svc, *, user="u1", story="s1", page=2, text=PAGE, voice="night_owl_english", lang="en-US",
         story_lang="en", theme=THEME, mode="standard", world_slug=None, child_name=None):
    return asyncio.run(svc._generate_page_audio(
        user_id=user, story_id=story, page=page, page_text=text, voice=voice, language_code=lang,
        voice_mode=mode, parent_voice_id=("pv1" if mode == "parent" else None), child_name=child_name,
        child_name_pronunciation=None, story_language_code=story_lang, story_world_id=None,
        story_world_slug=world_slug, story_theme=theme))


failures = 0


def check(ok, label):
    global failures
    failures += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + label)


def shared_keys(storage):
    return [k for k in storage.store if k.startswith("shared/seasonal/")]


# 1. First family generates and saves; second family reuses with no OpenAI call.
st = FakeStorage(); svc = make_service(st)
path_a, _ = play(svc, user="family-a", story="story-a")
check(len(svc.tts_calls) == 1, "first family: one OpenAI call")
check(len(shared_keys(st)) == 1, "first family: audio saved to the shared folder")
check(shared_keys(st)[0].startswith("shared/seasonal/christmas/the-little-fir/night_owl_english_en-us_"),
      "shared path is organised by festival / story / voice_language_version")
path_b, _ = play(svc, user="family-b", story="story-b")
check(len(svc.tts_calls) == 1, "second family: no OpenAI call (reused)")
check(svc.user_uploads[path_b] == svc.user_uploads[path_a], "second family gets identical audio")
check(path_b.startswith("family-b/story-b/chunked/"), "second family's copy goes in their own folder as before")

# 2. Anything that changes what OpenAI hears gets its own file.
def generates(label, **kw):
    n = len(svc.tts_calls)
    play(svc, **kw)
    check(len(svc.tts_calls) == n + 1, label)


generates("edited page text is generated again, never reused",
          user="family-c", story="story-c", text=PAGE + " Snow fell softly.")
generates("a different narrator / locale is a different file",
          user="family-e", story="story-e", lang="en-GB", voice="wise_owl")
generates("same words in another story are filed separately",
          user="family-f", story="story-f", theme="seasonal:christmas:another-story")
generates("another language is a different file",
          user="family-k", story="story-k", lang="es", voice="night_owl_spanish", story_lang="es")
before = len(shared_keys(st))


# 3. Never shared: they always generate and never touch the shared folder.
def never_shared(label, **kw):
    for who in ("x1", "x2"):  # twice: the second play must not reuse the first
        play(svc, user=who, story=who, **kw)
    check(len(shared_keys(st)) == before, label + ": nothing saved to the shared folder")


for label, kw in [("Bedtime", dict(theme=None)),
                  ("Story Worlds", dict(theme=None, world_slug="irish-legends")),
                  ("Parent Voice", dict(mode="parent")),
                  ("translated narration", dict(story_lang="es"))]:
    n = len(svc.tts_calls)
    never_shared(label, **kw)
    check(len(svc.tts_calls) == n + 2, label + ": generated both times")

# 4. Storage failures fall back to normal generation.
broken = FakeStorage(fail_download=True); svc2 = make_service(broken)
play(svc2); play(svc2, user="family-z", story="story-z")
check(len(svc2.tts_calls) == 2 and len(svc2.user_uploads) == 2, "storage down: still narrates normally")
nostore = make_service(None); nostore.story_repo = None
play(nostore)
check(len(nostore.tts_calls) == 1 and len(nostore.user_uploads) == 1, "no storage client: still narrates normally")

# 5. Pure helpers.
check(S.parse_seasonal_theme("seasonal:christmas:x") == ("christmas", "x"), "theme parsing")
check(S.parse_seasonal_theme("bedtime") is None and S.parse_seasonal_theme("seasonal:christmas") is None,
      "non-seasonal themes are rejected")
check(S.audio_fingerprint(tts_text="a", instructions="b") != S.audio_fingerprint(tts_text="a", instructions="c"),
      "changed voice directions change the fingerprint")

if failures:
    print(f"\nSHARED SEASONAL NARRATION: {failures} check(s) FAILED")
    sys.exit(1)
print("\nSHARED SEASONAL NARRATION OK")
