"""BEDTIME LOCK (21 Nov 2026, guard added 24 Nov 2026). Fails if ANY Bedtime Stories prompt changes.

Run before every deploy, especially after Story Worlds, seasonal or legend work:
    python tests/test_bedtime_lock.py <backend folder>
On Render it runs automatically as part of the Build Command, so a change that moves Bedtime
FAILS THE BUILD: the deploy stops, the live server keeps running the signed-off version,
and Render emails that the deploy failed.

PASS = Bedtime Stories are exactly as signed off.
FAIL = something changed Bedtime. Stop and tell the owner before going any further.

Only when Bedtime is changed ON PURPOSE and the owner has signed it off, refresh the fingerprints:
    python tests/test_bedtime_lock.py <backend folder> --update

The lock is checked three ways:
  1. with no Render settings at all (a bare backend must still write the signed-off stories),
  2. with every old switch set to off (they must be ignored),
  3. the version label and the Pages 3-7 model must be the signed-off ones.
"""
import asyncio, os, sys, random, json, hashlib, collections
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
from app.services import story_v8 as V8

Q._ALLOW_OLD_VERSIONS = False  # the lock checks Bedtime exactly as a server runs it

GOLDEN = os.path.join(os.path.dirname(__file__), "bedtime_lock_golden.json")
EXPECTED_BUILD = "bedtime_v8e=20261122"
EXPECTED_DRAFT_MODEL = "gemini-3.1-pro-preview"
CHILDREN = [  # ages 2-12, boy/girl, moral/no moral, comfort, custom theme
    dict(childName="Ana", age=2, theme="animals", moral=""),
    dict(childName="Tom", age=3, theme="animals", moral=""),
    dict(childName="Jess", age=4, theme="princess", moral="kindness"),
    dict(childName="Emily", age=6, theme="dragons", moral="sharing"),
    dict(childName="Zoe", age=8, theme="space", moral="kindness", customTheme="my tummy hurts"),
    dict(childName="Liam", age=12, theme="adventure", moral=""),
]
SETTINGS = list(Q.BEDTIME_LOCKED_SWITCHES) + ["GEMINI_DRAFT_MODEL"]


def clear_settings():
    for name in SETTINGS:
        os.environ.pop(name, None)


def fingerprints():
    out = {}
    for kw in CHILDREN:
        random.seed(42); CALLS.clear()
        svc = StoryService(FakeRepo()); req = make_request(**kw)
        d = asyncio.run(svc.generate_story_first_page(req, None))
        asyncio.run(svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"],
                                                  current_pages=d["pages"], companion=None, expected_pages=svc._intended_page_count(req)))
        out[f'{kw["childName"]}-{kw["age"]}'] = sorted(f"{k}:{hashlib.sha1(p.encode()).hexdigest()[:16]}" for k, p in CALLS)
    return out


clear_settings()
bare = fingerprints()
if "--update" in sys.argv:
    json.dump(bare, open(GOLDEN, "w"), indent=1); print("UPDATED bedtime_lock_golden.json"); sys.exit(0)

gold = json.load(open(GOLDEN))
failures = 0


def compare(now, label):
    global failures
    for key in gold:
        same = collections.Counter(gold[key]) == collections.Counter(now.get(key, []))
        failures += 0 if same else 1
        print(("PASS " if same else "FAIL ") + f"Bedtime prompts unchanged for {key} ({label})")


compare(bare, "no settings")
clear_settings()
for name in Q.BEDTIME_LOCKED_SWITCHES:
    os.environ[name] = "0"
compare(fingerprints(), "old switches set to off")

for ok, what in [
    (V8.current_build() == EXPECTED_BUILD, f"version label is {EXPECTED_BUILD} (got {V8.current_build()})"),
    (Q.role_model("draft") == EXPECTED_DRAFT_MODEL, f"Pages 3-7 model is {EXPECTED_DRAFT_MODEL} (got {Q.role_model('draft')})"),
]:
    failures += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + what)
clear_settings()

if failures:
    print(f"\nBEDTIME LOCK FAILED ({failures} check(s)). This change would alter Bedtime Stories.")
    print("STOP: do not deploy. Tell the owner what changed and wait for sign-off.")
    sys.exit(1)
print("\nBEDTIME LOCK OK: Bedtime Stories are exactly as signed off.")
