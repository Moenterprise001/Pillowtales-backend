"""BEDTIME NARRATION LOCK (24 Nov 2026). Fails if what a Bedtime story SOUNDS like changes.

Narration code is shared by Bedtime, Story Worlds and seasonal stories. This records exactly what a
Bedtime page sends to the voice (the final text and the voice directions) for every standard
narrator and for Parent Voice, and fails if any of it changes. Nothing is sent anywhere: the voice
calls are replaced by a recorder.

    python tests/test_bedtime_narration_lock.py <backend folder>           check (runs in the Render build)
    python tests/test_bedtime_narration_lock.py <backend folder> --update  ONLY after the owner signs off a
                                                                           Bedtime narration change
"""
import asyncio, os, sys, json, hashlib, types as pytypes
sys.path.insert(0, os.path.dirname(__file__))
try:
    import httpx  # noqa: F401  (installed on the server)
except ImportError:  # offline copies without httpx: the voice calls are replaced below anyway
    sys.modules["httpx"] = pytypes.ModuleType("httpx")
from harness import *  # noqa: E402  (stubs fastapi / supabase / google)
sys.modules["fastapi"].BackgroundTasks = getattr(sys.modules["fastapi"], "BackgroundTasks", object)
from app.domain.constants import VOICE_PRESETS  # noqa: E402
from app.services import narration_service as N  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(__file__), "bedtime_narration_golden.json")
PAGES = {  # one ordinary Bedtime page per language: dialogue, a dash, an ellipsis, a paragraph break
    "en": 'Emily tugged at Gobby\'s tail. "Up you get!" she puffed. The race had started... but Gobby '
          'just snored — loudly.\n\nThen the gong went CLANG. Gobby\'s eyes flew open!',
    "es": 'Emily tiró de la cola de Roque. —¡Arriba! —dijo. La carrera había empezado... pero Roque roncaba.'
          '\n\nDe pronto sonó el gong: ¡CLANG! Roque abrió los ojos.',
    "fr": 'Emily tira la queue de Gobby. « Debout ! » dit-elle. La course avait commencé... mais Gobby '
          'ronflait.\n\nSoudain, le gong fit CLANG ! Gobby ouvrit les yeux.',
    "de": 'Emily zog an Gobbys Schwanz. „Hoch mit dir!“, rief sie. Das Rennen hatte begonnen... aber Gobby '
          'schnarchte.\n\nPlötzlich machte der Gong KLANG! Gobby riss die Augen auf.',
    "it": 'Emily tirò la coda di Gobby. «Su, alzati!» disse. La corsa era iniziata... ma Gobby russava.'
          '\n\nAll’improvviso il gong fece CLANG! Gobby spalancò gli occhi.',
    "ja": 'エミリーはゴビーのしっぽを引っぱりました。「おきて！」\n\nそのとき、ドラがゴーンと鳴りました。',
    "ar": 'شدّت إميلي ذيل غوبي. «انهض!» قالت.\n\nثم دقّ الجرس فجأة! ففتح غوبي عينيه.',
}
LANG_OF_VOICE = {"night_owl_english": "en-US", "night_owl_spanish": "es", "night_owl_german": "de",
                 "night_owl_french": "fr", "night_owl_italian": "it", "night_owl_japanese": "ja",
                 "night_owl_arabic": "ar"}
CALLS = []
# Ten silent MP3 frames (MPEG-2 Layer III, 24 kHz mono, 160 kbit/s), standing in for the voice's audio.
FAKE_MP3 = (bytes([0xFF, 0xF3, 0xE4, 0xC0]) + bytes(476)) * 10


def recorder():
    svc = N.NarrationService(story_repo=None, user_repo=None, subscription_service=None,
                             story_world_pronunciation_service=None)

    async def openai(text, voice, language_code=None, pronunciation_guide=None, bedtime_pacing=False):
        CALLS.append(("openai", voice, language_code, text, svc._openai_tts_instructions(voice, language_code)
                      + svc._pronunciation_guide_instructions(pronunciation_guide)
                      + (getattr(N, "BEDTIME_PACING_DIRECTION", "") if bedtime_pacing else "")))
        return FAKE_MP3

    async def eleven(text, voice_id, language_code):
        CALLS.append(("parent", voice_id, language_code, text, ""))
        return FAKE_MP3

    async def upload(path, audio):
        CALLS.append(("audio_bytes", len(audio)))  # catches the page-end pause
        return None

    async def translate(text, target, source=None):
        return text

    svc._generate_openai_tts, svc._generate_elevenlabs_tts = openai, eleven
    svc._upload_audio, svc._translate_text = upload, translate
    return svc


def fingerprints(story_world=False):
    svc = recorder(); out = {}
    cases = []
    for voice, lang in LANG_OF_VOICE.items():
        if voice in VOICE_PRESETS:
            for page in (1, 3):
                cases.append((f"{voice}-p{page}", dict(voice=voice, language_code=lang, voice_mode="standard",
                                                       parent_voice_id=None, page=page)))
    cases.append(("en-GB-p3", dict(voice="night_owl_english", language_code="en-GB", voice_mode="standard",
                                   parent_voice_id=None, page=3)))
    for lang in ("en", "es", "fr"):
        cases.append((f"parent-{lang}-p3", dict(voice="night_owl_english", language_code=lang, voice_mode="parent",
                                                parent_voice_id="pv1", page=3)))
    for name, kw in cases:
        CALLS.clear()
        lang = kw["language_code"][:2]
        asyncio.run(svc._generate_page_audio(user_id="u1", story_id="s1", page_text=PAGES[lang],
                                             child_name="Emily", child_name_pronunciation=None,
                                             story_language_code=lang, story_world_id=None,
                                             story_world_slug="irish-legends" if story_world else None, **kw))
        out[name] = [hashlib.sha1(json.dumps(c, ensure_ascii=False).encode()).hexdigest()[:16] for c in CALLS]
    return out


if "--story-worlds" in sys.argv:  # parity helper: the same pages as a Story World
    print("JSON" + json.dumps(fingerprints(story_world=True))); sys.exit(0)
now = fingerprints()
if "--update" in sys.argv:
    json.dump(now, open(GOLDEN, "w"), indent=1); print("UPDATED bedtime_narration_golden.json"); sys.exit(0)
gold = json.load(open(GOLDEN))
failures = 0
for key in gold:
    same = gold[key] == now.get(key)
    failures += 0 if same else 1
    print(("PASS " if same else "FAIL ") + f"Bedtime narration unchanged for {key}")
# The Bedtime pace must stay out of Story Worlds and seasonal stories (owner, 25 Nov 2026).
svc = recorder()
for label, extra in [("Story Worlds", dict(story_world_slug="irish-legends")),
                     ("seasonal stories", dict(story_world_slug=None, story_theme="seasonal:christmas:the-little-fir"))]:
    leaked = False
    for voice, lang in LANG_OF_VOICE.items():
        CALLS.clear()
        asyncio.run(svc._generate_page_audio(user_id="u1", story_id="s1", page=3, page_text=PAGES[lang[:2]],
                                             voice=voice, language_code=lang, voice_mode="standard",
                                             parent_voice_id=None, child_name=None, child_name_pronunciation=None,
                                             story_language_code=lang[:2], story_world_id=None, **extra))
        direction = CALLS[0][4]
        pause_added = any(c[0] == "audio_bytes" and c[1] != len(FAKE_MP3) for c in CALLS)
        leaked = leaked or getattr(N, "BEDTIME_PACING_DIRECTION", "@@") in direction or pause_added
    failures += 1 if leaked else 0
    print(("FAIL " if leaked else "PASS ") + f"the Bedtime narration pace stays out of {label}")

if failures:
    print(f"\nBEDTIME NARRATION LOCK FAILED ({failures} check(s)). This change would alter how Bedtime Stories sound.")
    print("STOP: do not deploy. Tell the owner what changed and wait for sign-off.")
    sys.exit(1)
print("\nBEDTIME NARRATION LOCK OK: Bedtime narration is exactly as signed off.")
