import asyncio, sys, os, json, importlib, types as pytypes
sys.path.insert(0, os.path.dirname(__file__))
import harness
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
acts = {
 "Rob slides his prized fleece mitten over the dragon's cold claws.": True,
 "Emily gives up her treasured lavender pouch to warm Fern.": True,
 "Jess sacrifices her own snack.": True,
 "Emily hums her favourite lullaby, which calms the dragonet.": False,
 "Paul twists the anchor-crank into the cliff to brace the balloon.": False,
 "Emily feeds out slack in rhythm with the gusts.": False,
}
for act, want in acts.items():
    got = any(k == "worn_trope" for k, _ in Q.trope_issues({"childs_decisive_act": act, "stake_sign": "a bell rings", "signature_delight": "a catchphrase"}))
    ok(got == want, f"{'flag' if want else 'allow'}: {act[:60]}")
ok("follows_plot_shape" in [k for k, _, _ in Q.PLAN_CHECK_KEYS] and "nothing_new_at_climax" in [k for k, _, _ in Q.PLAN_CHECK_KEYS], "plan check enforces the plot shape and nothing-new-at-the-climax")
pc = Q.build_plan_check_prompt(make_request(age=7), "p1", {"plot_shape": "swap", "beats": ["a"] * 6}, 7)
ok("PLOT SHAPE OF THIS PLAN: swap" in pc, "checker is told what the chosen shape means")
ok("Nothing new at the climax" in Q.EDITOR_CHECKLIST, "editor flags a solution that appears out of nowhere")
# Pro tuning
os.environ["GEMINI_DRAFT_MODEL"] = "gemini-3.1-pro-preview"
ok(Q.role_thinking_budget("draft", 0) == 2048, "Pro thinking capped at 2,048 by default")
os.environ["GEMINI_DRAFT_THINKING_BUDGET"] = "-1"; ok(Q.role_thinking_budget("draft", 0) == -1, "...the Render setting still overrides it"); del os.environ["GEMINI_DRAFT_THINKING_BUDGET"]
import app.services.story_service as SS
Real = importlib.reload(SS).StoryService
SEEN = []
def once(self, model, prompt, schema=None, max_tokens=None, thinking=None):
    ctx = Q.GEN_CTX.get(); SEEN.append((ctx[0] if ctx else None, model, prompt[:40]))
    r = harness.fake_generate(self, prompt, schema, max_tokens, thinking)
    r.usage_metadata = pytypes.SimpleNamespace(prompt_token_count=10, candidates_token_count=2, thoughts_token_count=0); return r
Real._generate_content_once_sync = once
svc = Real(FakeRepo()); req = make_request(childName="Emily", age=8, theme="dragons", moral="patience")
d = asyncio.run(svc.generate_story_first_page(req, None)); asyncio.run(svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7))
rewrite_models = {m for r, m, p in SEEN if p.startswith("Revise specific pages")}
draft_models = {m for r, m, p in SEEN if r == "draft"}
ok(draft_models == {"gemini-3.1-pro-preview"} and rewrite_models and "gemini-3.1-pro-preview" not in rewrite_models, f"Pro writes the draft, Flash does the small rewrites ({draft_models}, {rewrite_models})")
del os.environ["GEMINI_DRAFT_MODEL"]
