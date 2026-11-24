"""v8d (21 Nov 2026): read-aloud excellence. Run: python tests/test_v8d.py <backend>"""
import asyncio, os, sys, random, re
sys.path.insert(0, os.path.dirname(__file__))
os.environ["BEDTIME_V8"] = "1"; os.environ.pop("BEDTIME_V8D", None)
os.environ["BEDTIME_V8E"] = "0"  # this file guards v8d; test_v8e.py covers v8e
from harness import *
from app.services import story_v8 as V8
from fixtures_readaloud import ZEBRA, JESS
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)

ok(V8.v8d_enabled() and V8.current_build() == "bedtime_v8d=20261121", "v8d is on by default with v8, and labelled bedtime_v8d")
os.environ["BEDTIME_V8D"] = "0"; ok(not V8.v8d_enabled() and V8.current_build() == "bedtime_v8c=20261113", "BEDTIME_V8D=0 gives back v8c (labelled v8c)")
del os.environ["BEDTIME_V8D"]

# ---- the measured read-aloud check, calibrated on real stories ----
zebra_flags = [V8.read_aloud_issue(p, 6) for p in ZEBRA.split("\n\n")]
ok(sum(1 for f in zebra_flags if f) >= 3, f"The Runaway Zebra Crossing (owner: hard to read at 6) has 3+ pages flagged: {[i+1 for i,f in enumerate(zebra_flags) if f]}")
ok(not any(V8.read_aloud_issue(p, 3) for p in JESS.split("\n\n")), "Jess and the Crown with Paws (owner's favourite) passes on every page")

def run(**kw):
    CALLS.clear(); svc = StoryService(FakeRepo()); req = make_request(**kw)
    d = asyncio.run(svc.generate_story_first_page(req, None))
    asyncio.run(svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7))
    by = {}
    for k, p in CALLS: by.setdefault(k, []).append(p)
    return req, by, svc

random.seed(5)
for age in (3, 6, 12):
    req, by, svc = run(childName="Emily", age=age, theme="animals", moral="")
    t = Q.get_trace(req)
    ok(t.get("v8_build") == "bedtime_v8d=20261121", f"age {age}: story labelled v8d")
    ok(all(p.count("READ-ALOUD STANDARD") == 1 for k in ("page1", "plan", "continuation") for p in by.get(k, [])), f"age {age}: read-aloud standard sent once to Page 1, planner and writer")
    s, u, l = V8.READ_ALOUD[V8.band(age)]
    ok(all(f"At most {s} sentences a page" in p and f"never more than {l}" in p for p in by["continuation"]), f"age {age}: the {s}-sentence / {l}-word limit for this age")
    ok("Read every page aloud as a parent would" in by["editor"][0], f"age {age}: the editor reads aloud first")
    ok("easy_to_read_aloud" in t["score"]["criteria"] and t["score"]["rubric_version"] == "v8d-20261121", f"age {age}: scorer marks ease of reading aloud (rubric v8d)")
    ok("Each beat is one clear picture" in by["plan"][0], f"age {age}: planner: one picture per beat, no repeated action")
    if by.get("page1_judge"):
        ok("Easy to picture and read aloud" in by["page1_judge"][0], f"age {age}: the judge prefers the easiest story to picture")
    ok("A painted stripe never becomes paper" in by["continuation"][0] or "a painted stripe never becomes paper" in by["continuation"][0], f"age {age}: the crazy thing never changes what it is")

svc = StoryService(FakeRepo())
for age, expect in ((5, (65, 85, 110)), (6, (65, 85, 110)), (7, (80, 100, 130)), (8, (90, 115, 150))):
    b = svc._standard_bedtime_page_budget(age, 3, "en")
    ok((b["target_min"], b["target_max"], b["hard_max"]) == expect, f"age {age}: shorter read-aloud pages {expect}")
for age in (3, 4, 9, 12):
    b = svc._standard_bedtime_page_budget(age, 3, "en")
    ok((b["target_min"], b["target_max"], b["hard_max"]) == Q.AGE_PAGE_BUDGETS[age][1], f"age {age}: page size unchanged")
totals = {a: sum(svc._standard_bedtime_page_budget(a, n, "en")["target_max"] for n in range(1, 8)) for a in (3, 5, 7, 8, 9, 11)}
ok(list(totals.values()) == sorted(totals.values()), f"length still grows with age {totals}")
os.environ["BEDTIME_V8D"] = "0"
ok(svc._standard_bedtime_page_budget(6, 3, "en")["target_max"] == Q.AGE_PAGE_BUDGETS[6][1][1], "BEDTIME_V8D=0: v8c page sizes")
del os.environ["BEDTIME_V8D"]

# deterministic rewrite: an overlong page is always sent back (fake draft pages are long)
req, by, svc = run(childName="Emily", age=6, theme="animals", moral="")
dr = Q.get_trace(req).get("draft_revise", {})
ok(isinstance(dr.get("read_aloud_pages", []), list) and len(dr.get("read_aloud_pages", [])) <= 3, f"read-aloud rewrites recorded and capped at 3: {dr.get('read_aloud_pages')}")
ok(all(n >= 3 for n in dr.get("read_aloud_pages", [])), "Page 2 (already on screen) is never rewritten")

# ages 0-2 untouched
random.seed(3); _, on, _ = run(childName="Ana", age=2, theme="animals", moral="")
os.environ["BEDTIME_V8D"] = "0"; random.seed(3); _, off, _ = run(childName="Ana", age=2, theme="animals", moral="")
del os.environ["BEDTIME_V8D"]
ok(sorted(p for ps in on.values() for p in ps) == sorted(p for ps in off.values() for p in ps), "ages 0-2: v8d changes nothing")
