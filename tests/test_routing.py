import asyncio, sys, os, json, importlib, types as pytypes
sys.path.insert(0, os.path.dirname(__file__))
import harness
from harness import *
import app.services.story_service as ss_mod
ss_mod = importlib.reload(ss_mod)           # real _generate_content_sync again
Real = ss_mod.StoryService
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
SEEN = []
def once(self, model, prompt, schema=None, max_tokens=None, thinking=None):
    ctx = Q.GEN_CTX.get(); role = ctx[0] if ctx else None
    SEEN.append((role, model, thinking, max_tokens))
    if os.environ.get("FAIL_MODEL") == model: raise Exception("500 INTERNAL: upstream error")
    if os.environ.get("REJECT_THINKING") == model and thinking not in (-1,): raise Exception("400 INVALID_ARGUMENT: thinking_budget is not supported")
    r = harness.fake_generate(self, prompt, schema, max_tokens, thinking)
    r.usage_metadata = pytypes.SimpleNamespace(prompt_token_count=1000, candidates_token_count=200, thoughts_token_count=50 if thinking else 0)
    return r
Real._generate_content_once_sync = once
async def run():
    repo = FakeRepo(); svc = Real(repo); req = make_request(childName="Emily", age=8, theme="dragons", moral="patience")
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return req, repo
def models_for(role): return {m for r, m, _, _ in SEEN if r == role}

# 1) nothing set -> everything on GEMINI_MODEL default, as before
SEEN.clear(); req, repo = asyncio.run(run())
ok({m for _, m, _, _ in SEEN} == {"gemini-3.8-flash"}, f"no settings: every call uses gemini-3.8-flash ({ {m for _, m, _, _ in SEEN} })")
ok(Q.get_trace(req).get("final_status") == "complete", "story completes")
u = Q.get_trace(req).get("usage", {})
ok(set(u) >= {"page1", "pages", "planner"} and u["pages"]["calls"] >= (6 if os.environ.get("BEDTIME_DRAFT_REVISE") == "0" else 1) and u["pages"]["in"] > 0, f"token usage logged per role: { {k: (v['calls'], v['in'], v['out']) for k, v in u.items()} }")

# 2) GEMINI_PAGES_MODEL -> only Pages 2+ move
os.environ["GEMINI_PAGES_MODEL"] = "gemini-3.8-flash"; SEEN.clear(); req, repo = asyncio.run(run())
ok(models_for("pages") == {"gemini-3.8-flash"}, f"pages use the new model ({models_for('pages')})")
ok(models_for("page1") == {"gemini-3.8-flash"} and models_for("planner") == {"gemini-3.8-flash"}, "Page 1 and planner unchanged")
ok(Q.get_trace(req)["usage"]["pages"]["model"] == "gemini-3.8-flash", "log records the pages model")
ok(all(not t for r, _, t, _ in SEEN if r == "pages"), "pages thinking stays off by default")

# 3) new model fails -> falls back to default model, story still completes
os.environ["FAIL_MODEL"] = "gemini-3.8-flash"; SEEN.clear(); req, repo = asyncio.run(run())
tr = Q.get_trace(req)
ok(tr.get("final_status") == "complete" and tr["usage"]["pages"]["model"] == "gemini-3.6-flash" and tr["usage"]["pages"].get("fallback_from") == "gemini-3.8-flash", "failing pages model -> automatic fallback, story completes, fallback logged")
del os.environ["FAIL_MODEL"]

# 4) thinking budget for pages, and the output limit grows to make room
os.environ["GEMINI_PAGES_THINKING_BUDGET"] = "1024"; SEEN.clear(); req, repo = asyncio.run(run())
pages_calls = [(t, mx) for r, _, t, mx in SEEN if r == "pages"]
ok(all(t == 1024 for t, _ in pages_calls), "pages thinking = 1024 when set")
ok(all(mx is None or mx >= 1024 + 1000 for _, mx in pages_calls), f"page output never capped below thinking room ({sorted({str(mx) for _, mx in pages_calls})})")
ok(all(not t for r, _, t, _ in SEEN if r == "page1"), "Page 1 never thinks")
# 5) a model that rejects a thinking budget -> retried with model default (-1)
os.environ["REJECT_THINKING"] = "gemini-3.8-flash"; SEEN.clear(); req, repo = asyncio.run(run())
ok(any(r == "pages" and m == "gemini-3.8-flash" and t == -1 for r, m, t, _ in SEEN) and Q.get_trace(req)["usage"]["pages"]["model"] == "gemini-3.8-flash", "thinking rejected -> same model retried with its default thinking")
ok(Q.get_trace(req).get("final_status") == "complete", "story completes")
for k in ("GEMINI_PAGES_MODEL", "GEMINI_PAGES_THINKING_BUDGET", "REJECT_THINKING"): os.environ.pop(k, None)
