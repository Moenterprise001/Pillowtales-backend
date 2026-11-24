import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
fams = ["pirate_harbour", "dragon_market", "forest_school"]
rows = [{"title": f"Story {i}", "characters": [{"name": f"Name{i}"}], "seed_family": fams[i % 3] if i >= 12 and i != 15 else None,
         "generation_trace": {"seed": {"family": "glowing_attic"}} if i == 15 else {}} for i in range(30)]
h = Q.load_recent_history(FakeRepo(rows), "u1", "Emily")
ok({"pirate_harbour", "dragon_market", "forest_school", "glowing_attic"} <= h["families"], "openings from stories 11-30 are remembered (slim columns and full trace both read)")
ok(len(h["titles"]) == 30, "titles from all 30 stories")
ok(h["names"] == [f"Name{i}" for i in range(10)], "names to avoid only from the 10 most recent")
ok(h["family_order"][0] == "pirate_harbour" and len(h["family_order"]) == 4, f"recency order kept: {h['family_order']}")
# all openings for a theme used -> the one used longest ago
svc = StoryService(FakeRepo())
pool = [s for s in svc._seed_pool_for_age(8)]
allowed, _ = Q.resolve_theme_seed_families("dragons", None)
dragon_fams = sorted({s["family"] for s in pool if s.get("family") in allowed})
order = list(reversed(dragon_fams))  # dragon_fams[0] is the oldest
picked = set()
for _ in range(20):
    r = make_request(childName="Emily", age=8, theme="dragons")
    svc._recent_story_history = (lambda req, o=order: {"families": set(o), "family_order": o, "names": [], "titles": [], "loaded": True})
    picked.add(svc._select_opening_seed(r)["family"])
ok(picked == {dragon_fams[0]}, f"all {len(dragon_fams)} dragon openings used -> picks the oldest ({picked})")
