import asyncio, sys, os, json, importlib, types as pytypes
sys.path.insert(0, os.path.dirname(__file__))
import harness
from harness import *
import app.services.story_service as SS
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
class Repo(FakeRepo):
    def update(self, sid, uid, v): self.updates.append(json.loads(json.dumps(v, default=str))); return v
async def run(svc_cls=StoryService, **kw):
    CALLS.clear(); repo = Repo(); svc = svc_cls(repo); req = make_request(**{"childName": "Emily", "age": 8, "theme": "dragons", "moral": "patience", **kw})
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return req, repo, d

# --- 1. plan: solution, stakes, delight ---
ok(set(Q.PLAN_V5_EXTRA_FIELDS) <= set(Q.plan_response_schema()["required"]), "v5 plan must name solution_key, planted page, stake_for_child, stake_sign, signature_delight")
good = {"beats": ["Fern shows Emily her hairbrush habit and the queue of guests grows longer", "Emily brushes the dragon's mane; the queue of guests grows longer still",
                  "Her first idea fails and the queue of guests grows longer", "The setback: the queue of guests nearly leaves", "Emily uses the hairbrush to warm the nest", "Calm; the hairbrush on the pillow"],
        "solution_key": "the hairbrush", "solution_planted_page": 2, "stake_for_child": "the baby dragon will never hatch", "stake_sign": "the queue of guests grows longer", "signature_delight": "hairbrush"}
ok(Q.structural_plan_issues(good) == [], "a well-built plan passes the structural checks")
late = dict(good, beats=["A walk", "A chat", "The egg grows colder; it fails", "The egg grows colder", "Emily finds a hairbrush and uses it", "Calm"])
iss = Q.structural_plan_issues(late)
ok(any(k == "solution_planted" and "Page 3 beat" in m for k, m in iss), "solution first appearing on Page 6 is caught, fix says to plant it on Page 3")
nostake = dict(good, stake_sign="the lantern dims")
ok(any(k == "stake_felt" for k, _ in Q.structural_plan_issues(nostake)), "stake sign missing from Pages 2-3 and the setback is caught")
ok(Q.structural_plan_issues({**good, "solution_key": ""})[0][0] == "solution_planted", "missing solution_key is caught")
os.environ["BEDTIME_PROMPTS_V5"] = "0"
ok(Q.structural_plan_issues(late) == [] and Q.plan_response_schema() is Q.PLAN_RESPONSE_SCHEMA, "v5 off: no new checks, old schema")
del os.environ["BEDTIME_PROMPTS_V5"]
os.environ["FAKE_PLAN_NO_V5"] = "1"
req, repo, d = asyncio.run(run()); tr = Q.get_trace(req)
del os.environ["FAKE_PLAN_NO_V5"]
plan_p = [p for k, p in CALLS if k == "plan"][0]
ok("solution_key" in plan_p and "stake_for_child" in plan_p and "signature_delight" in plan_p and "want this story again" in plan_p, "plan prompt asks for the solution, the child's stake and a signature delight")
chk = [p for k, p in CALLS if k == "plan_check"][0]
ok("stake_felt_by_child" in chk, "plan checker judges whether the child feels the stake")
pc = tr["plan_check"]
ok(pc["status"] == "revised" and "solution_planted" in pc.get("structural", []), f"fake plan without a planted solution -> structural check fails -> revised ({pc['status']}, {pc.get('structural')})")
rv = [p for k, p in CALLS if k == "plan_revision"]
ok(rv and "Page 3 beat" in rv[0], "revision is told exactly where to plant the solution")
ok(len([u for u in repo.updates if u.get("generation_status") == "complete"][-1]["pages"]) == 7, "story completes")
ed = [p for k, p in CALLS if k == "editor"][0]
ok("shown clearly by Page 3" in ed and "signature delight" in ed, "editor checks the solution set-up and the delight")
req, repo, d = asyncio.run(run()); tr = Q.get_trace(req)
ok(not tr["plan_check"].get("structural"), f"a plan with the solution planted on Page 3 passes the structural checks ({tr['plan_check'].get('structural')})")

# --- 2. models: planner role moves plan/check/editor/Page 7 review; judge, scorer, Story Worlds stay ---
ss_mod = importlib.reload(SS); Real = ss_mod.StoryService
SEEN = []
def once(self, model, prompt, schema=None, max_tokens=None, thinking=None):
    ctx = Q.GEN_CTX.get(); SEEN.append((ctx[0] if ctx else None, model))
    r = harness.fake_generate(self, prompt, schema, max_tokens, thinking)
    r.usage_metadata = pytypes.SimpleNamespace(prompt_token_count=10, candidates_token_count=2, thoughts_token_count=0); return r
Real._generate_content_once_sync = once
os.environ["GEMINI_PLANNER_MODEL"] = "gemini-3.8-flash"; os.environ["GEMINI_PAGES_MODEL"] = "gemini-3.8-flash"
req, repo, d = asyncio.run(run(Real))
by = {}
for role, model in SEEN: by.setdefault(role, set()).add(model)
ok(by.get("planner") == {"gemini-3.8-flash"}, f"plan, plan check, editor and Page 7 review on 3.8 ({by.get('planner')})")
ok(by.get("judge") == {"gemini-3.8-flash"} and by.get("scorer") == {"gemini-3.8-flash"}, f"Page 1 judge and scorer unchanged ({by.get('judge')}, {by.get('scorer')})")
ok(by.get("page1") == {"gemini-3.8-flash"}, "Page 1 on the new default")
del os.environ["GEMINI_PLANNER_MODEL"]; del os.environ["GEMINI_PAGES_MODEL"]
src = open(SS.__file__).read()
ok(src.count('self._gen("canon_review", request)') == 2, "Story Worlds reviews keep their own setting (unchanged)")
