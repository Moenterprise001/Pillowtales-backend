import asyncio, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
def R(**kw): return make_request(**{"childName": "Emily", "age": 7, "theme": "custom", "moral": "kindness", **kw})
for theme, want in [("Daddy stopped the pain in my stomach", True), ("Scared of the dark", True), ("First day at new school", True),
                    ("Going to the dentist", True), ("Monster under my bed", True), ("Tummy ache", True),
                    ("Monster trucks", False), ("Dark forest adventure", False), ("Cold winter sledging", False), ("Mermaids", False), ("Soccer", False)]:
    ok(Q.comfort_theme(R(customTheme=theme)) == want, f"comfort theme {'on' if want else 'off'}: {theme!r}")
ok(not Q.comfort_theme(make_request(childName="Emily", age=7, theme="dragons")), "preset themes never switch comfort mode on")
# rules reach every stage
CALLS.clear()
class Repo(FakeRepo):
    def update(self, sid, uid, v): self.updates.append(json.loads(json.dumps(v, default=str))); return v
async def run(req):
    svc = StoryService(Repo()); d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
req = R(customTheme="Daddy stopped the pain in my stomach")
asyncio.run(run(req))
by = {}
for k, p in CALLS: by.setdefault(k, []).append(p)
ok(all("COMFORT THEME" in p for p in by["page1"]), "Page 1 options get the comfort rule")
ok("COMFORT THEME" in by["plan"][0], "plan gets the comfort rule")
ok(all("COMFORT THEME" in p and "MOMENTUM" in p for p in by["continuation"]), "every page-writing prompt gets momentum + comfort")
ok("COMFORT THEME" in by["editor"][0] and "Momentum (most important)" in by["editor"][0], "editor checks momentum and comfort")
ok("EVERY BEAT IS A NEW EVENT" in by["plan"][0] and "every_beat_is_an_event" in by["plan_check"][0], "plan and plan check require a new event in every beat")
ok("caps this score at 6" in by["score"][0], "scorer penalises pages where nothing happens")
CALLS.clear(); asyncio.run(run(make_request(childName="Emily", age=8, theme="dragons", moral="patience")))
ok(not any("COMFORT THEME" in p for _, p in CALLS) and any("MOMENTUM" in p for k, p in CALLS if k == "continuation"), "ordinary story: momentum yes, comfort no")
os.environ["BEDTIME_PROMPTS_V5"] = "0"
ok(Q.extra_page_rules(req) == "" and Q.comfort_theme_block(req) == "", "v5 off: no new rules")
del os.environ["BEDTIME_PROMPTS_V5"]
ok(Q.build_score_prompt(req, "t", ["a"], True, 7).count("") and "momentum=20261111" in Q.QUALITY_BUILD, "build tagged")
import app.services.story_service as SS
lb = [s for s in SS.OPENING_SEED_FAMILIES if s["family"] == "space_lunchbox_rocket"][0]
ok(lb["en"].startswith("Every Friday, {childName} packed the lunchboxes") and all("{childName}" in lb[k] for k in ("en","es","fr","it","de")), "lunchbox opening now gives the child a role, in all 5 languages")
ok("why the child is part of it" in Q.hook_block("job"), "Page 1 told to connect the child to any unusual custom")
ok("Page 2 opens with something happening in its FIRST sentence" in Q.MOMENTUM_BLOCK and "always a must-fix" in Q.EDITOR_CHECKLIST, "Page 2 opens with an event; editor must fix repeats")
