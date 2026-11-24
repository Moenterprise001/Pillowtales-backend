import asyncio, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
# shapes by age and recency
ok("cumulative_tale" in [k for k, lo, hi, _ in Q.PLOT_SHAPES if lo <= 4 <= hi] and "bargain" not in Q.choose_plot_shapes(4, [], 12), "cumulative tale for little ones; no bargain plot at 4")
recent = ["escalating_tries", "journey_home", "grumpy_won_over"]
picked = Q.choose_plot_shapes(9, recent)
ok(len(picked) == 3 and not set(picked) & set(recent), f"three shapes the child has not heard recently ({picked})")
ok(Q.choose_plot_shapes(3, Q.choose_plot_shapes(3, [], 12) )[0] in [k for k, lo, hi, _ in Q.PLOT_SHAPES if lo <= 3 <= hi], "when all are used, the oldest come back first")
# tropes
bad = {"stake_sign": "the lantern's glow fades and dims", "signature_delight": "Pip sneezes sparkly puffs", "beats": ["a"] * 6}
iss = [k for k, _ in Q.trope_issues(bad)]
ok(iss.count("worn_trope") == 2, "fading-glow stakes and sneeze-puff jokes are sent back")
ok(Q.trope_issues({"stake_sign": "the queue of guests grows longer", "signature_delight": "a grumpy dragon's catchphrase"}) == [], "fresh stakes and jokes pass")
# voice by age
v4 = Q.voice_block(make_request(age=4)); v7 = Q.voice_block(make_request(age=7)); v11 = Q.voice_block(make_request(age=11))
ok("REFRAIN" in v4 and "narrator" in v7 and "witty narrator" in v11, "voice changes with age: refrain (4), playful narrator (7), wit (11)")
ok(Q.voice_block(make_request(age=2)) == "", "ages 0-2 keep their own style")
# end to end
class Repo(FakeRepo):
    def update(self, sid, uid, v): self.updates.append(json.loads(json.dumps(v, default=str))); return v
async def run(req):
    repo = Repo(); svc = StoryService(repo); d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return repo
CALLS.clear(); req = make_request(childName="Emily", age=7, theme="dragons", moral="kindness"); repo = asyncio.run(run(req)); tr = Q.get_trace(req)
by = {}
for k, p in CALLS: by.setdefault(k, []).append(p)
ok("PLOT SHAPE" in by["plan"][0] and "WORN-OUT TROPES" in by["plan"][0] and "child_flaw" in by["plan"][0], "plan offers 3 plot shapes, bans tropes, asks for a child flaw and a comic trait")
ok(set(Q.PLAN_V5_EXTRA_FIELDS) >= {"plot_shape", "child_flaw", "comic_trait"}, "plan must return its shape, the child's flaw and a comic trait")
ok(tr["plan"]["plot_shape"] in tr["plot_shape_options"] and not tr["plan_check"].get("structural"), f"plan picked an offered shape ({tr['plan']['plot_shape']}) and passed the checks")
ok(all("VOICE" in p and "WORN-OUT TROPES" in p for p in by["page1"]), "Page 1 options get the voice and trope rules")
ok(all("VOICE" in p and "narrator" in p for p in by["continuation"]), "every page prompt gets the voice rules")
ok("Voice and laughs" in by["editor"][0], "editor checks voice, laughs and tropes")
ok("Generic description with one mild funny moment scores 7 at most" in by["score"][0], "scorer rewards voice and real laughs")
mem = repo.client.memory
ok(mem and mem[0].get("plot_shape") == tr["plan"]["plot_shape"], "plot shape saved to the child's memory")
h = Q.load_recent_history(FakeRepo([], memory=[{"story_id": "x", "created_at": "2026-11-12", "title": "T", "plot_shape": "swap", "names": []}]), "u1", "Emily")
ok(h.get("plot_shapes") == ["swap"], "next story knows which shapes were used")
# shape not offered is sent back
st = Q.get_state(req); st["plot_shape_options"] = ["swap", "bargain", "clue_trail"]
ok(any(k == "plot_shape" for k, _ in Q.structural_plan_issues(dict(tr["plan"], plot_shape="something_else"), req, "")), "a shape that was not offered is sent back")
os.environ["BEDTIME_PROMPTS_V5"] = "0"
ok(Q.story_craft_v7(req) == "" and Q.plot_shape_block(["swap"]) == "" and Q.trope_issues(bad) == [], "v5 off: none of this applies")
del os.environ["BEDTIME_PROMPTS_V5"]
