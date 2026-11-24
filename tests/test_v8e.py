"""v8e (22 Nov 2026): the solution must really work, the child is in charge, the ending fits its time and place.
Run: python tests/test_v8e.py <backend>"""
import asyncio, os, sys, random
sys.path.insert(0, os.path.dirname(__file__))
os.environ["BEDTIME_V8"] = "1"; os.environ.pop("BEDTIME_V8D", None); os.environ.pop("BEDTIME_V8E", None)
from harness import *
from app.services import story_v8 as V8
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)

ok(V8.v8e_enabled() and V8.current_build() == "bedtime_v8e=20261122", "v8e is on by default, labelled bedtime_v8e")
os.environ["BEDTIME_V8E"] = "0"; ok(V8.current_build() == "bedtime_v8d=20261121", "BEDTIME_V8E=0 gives back v8d")
os.environ["BEDTIME_V8D"] = "0"; ok(V8.current_build() == "bedtime_v8c=20261113" and not V8.v8e_enabled(), "BEDTIME_V8D=0 gives back v8c (v8e off too)")
del os.environ["BEDTIME_V8E"]; del os.environ["BEDTIME_V8D"]

def run(**kw):
    CALLS.clear(); svc = StoryService(FakeRepo()); req = make_request(**kw)
    d = asyncio.run(svc.generate_story_first_page(req, None))
    asyncio.run(svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7))
    by = {}
    for k, p in CALLS: by.setdefault(k, []).append(p)
    return req, by

random.seed(9)
for age in (3, 6, 12):
    req, by = run(childName="Emily", age=age, theme="space", moral="patience")
    t = Q.get_trace(req)
    ok(t.get("v8_build") == "bedtime_v8e=20261122", f"age {age}: story labelled v8e")
    ok(all(p.count("THE CHILD IS IN CHARGE") == 1 for p in by["continuation"]), f"age {age}: writer: the child is in charge, from Page 2")
    ok(any("not among the stars" in p for p in by["continuation"]), f"age {age}: writer: the ending matches its time and place")
    plan = by["plan"][0]
    ok("A kind or brave gesture alone never fixes a physical problem" in plan and "1b. Would the decisive act really fix" in plan and "1c. Does the child do something" in plan,
       f"age {age}: planner: the solution must really work, and checks itself")
    ok("a sidekick never carries a beat alone" in plan, f"age {age}: planner: the child acts in every beat")
    pc = by["plan_check"][0]
    ok("- solution_really_works:" in pc and "- child_in_charge:" in pc and len([l for l in pc.split("CHECKS:")[1].splitlines() if l.startswith("- ")]) == 8,
       f"age {age}: plan check has 8 checks, including 'solution really works' and 'child in charge'")
    ok("Story logic (check right after reading aloud)" in by["editor"][0], f"age {age}: editor checks story logic")
    crit = dict(V8.score_criteria())
    ok(t["score"]["rubric_version"] == "v8e-20261122" and "sidekick doing the action" in crit["child_drives_climax"] and "would not really work" in crit["coherence"],
       f"age {age}: scorer marks down a passive child and a solution that doesn't work (rubric v8e)")
    ok(all(p.count("READ-ALOUD STANDARD") == 1 for p in by["continuation"]), f"age {age}: v8d read-aloud standard still there")

random.seed(3); _, on = run(childName="Ana", age=2, theme="animals", moral="")
os.environ["BEDTIME_V8E"] = "0"; random.seed(3); _, off = run(childName="Ana", age=2, theme="animals", moral="")
del os.environ["BEDTIME_V8E"]
ok(sorted(p for ps in on.values() for p in ps) == sorted(p for ps in off.values() for p in ps), "ages 0-2: v8e changes nothing")
