import asyncio, sys, os, json, re
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
import app.services.story_service as SS
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
new = SS.OPENING_SEED_FAMILIES_V5
ok(len(new) == 22 and all(all(s.get(k) for k in ("en", "es", "fr", "it", "de")) for s in new), "22 new openings, each in 5 languages")
ok(all("{childName}" in s[k] for s in new for k in ("en", "es", "fr", "it", "de")), "every translation names the child")
fams = {s["family"] for s in new}
ok(len(fams) == 22 and not (fams & {s["family"] for s in SS.OPENING_SEED_FAMILIES if not s.get("v5")}), "new family names are unique")
in_theme = set().union(*Q.THEME_SEED_ALLOWLIST.values())
ok(fams <= in_theme, "every new opening belongs to a theme")
ok(all(any(f in SS.AGE_SEED_FAMILY_ALLOWLIST[b] for b in ("5_6", "7_8", "9_10")) for f in fams), "every new opening is allowed for ages 5-10")
starts = sum(1 for s in new if s["en"].startswith("{childName}"))
ok(starts <= 6, f"most new openings do not start with the child's name ({starts}/22 do)")
svc = StoryService(FakeRepo())
for theme in ["dragons", "princess", "space", "animals", "adventure", "underwater", "dinosaurs", "magic", "superheroes", "emotions"]:
    allowed, _ = Q.resolve_theme_seed_families(theme, None)
    n = len([s for s in svc._seed_pool_for_age(8) if s["family"] in allowed])
    print(f"   age 8 {theme}: {n} openings")
allowed, label = Q.resolve_theme_seed_families("superheroes", None)
ok("hero_cape_repair" in allowed and "pirate_harbour" in allowed, f"superheroes get hero openings plus adventure ones ({label})")
os.environ["BEDTIME_PROMPTS_V5"] = "0"
ok(not any(s.get("v5") for s in svc._seed_pool_for_age(8)), "v5 off: new openings not used (old behaviour)")
del os.environ["BEDTIME_PROMPTS_V5"]
# best-of: three different openings
CALLS.clear(); repo = FakeRepo(); svc = StoryService(repo); req = make_request(childName="Emily", age=8, theme="dragons", moral="patience")
d = asyncio.run(svc.generate_story_first_page(req, None))
p1 = [p for k, p in CALLS if k == "page1"]
openings = [re.search(r'rewritten naturally: "([^"]+)"', p).group(1) for p in p1]
ok(len(p1) == 3 and len(set(openings)) == 3, "three Page 1 candidates, three different openings")
bo = Q.get_trace(req)["page1_best_of"]
ok(len({c["seed"] for c in bo["candidates"]}) == 3, f"candidate openings recorded: {[c['seed'] for c in bo['candidates']]}")
chosen = bo["candidates"][bo["chosen"] - 1]["seed"]
ok(Q.get_trace(req)["seed"]["family"] == chosen, f"the chosen candidate's opening is recorded for repeat-avoidance ({chosen})")
ok(all("never with the child's name followed by lived, loved or visited" in p for p in p1), "every candidate told not to start 'Emily lived/loved/visited'")
# custom theme: no seed sentence in prompt -> unchanged prompts
CALLS.clear(); req = make_request(childName="Emily", age=8, theme="custom", customTheme="Horse summer camp", moral="patience")
asyncio.run(StoryService(FakeRepo()).generate_story_first_page(req, None))
ok(len([k for k, _ in CALLS if k == "page1"]) == 3, "custom theme still gets 3 candidates (opening inside the parent's theme)")
