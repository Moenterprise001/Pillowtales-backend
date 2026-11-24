import asyncio, sys, os, json, importlib, types as pytypes
sys.path.insert(0, os.path.dirname(__file__))
import harness
from harness import *
import app.services.story_service as SS
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
Real = importlib.reload(SS).StoryService
SEEN = []
def once(self, model, prompt, schema=None, max_tokens=None, thinking=None):
    ctx = Q.GEN_CTX.get(); SEEN.append((ctx[0] if ctx else None, model, thinking))
    r = harness.fake_generate(self, prompt, schema, max_tokens, thinking)
    r.usage_metadata = pytypes.SimpleNamespace(prompt_token_count=10, candidates_token_count=2, thoughts_token_count=0); return r
Real._generate_content_once_sync = once
async def run():
    SEEN.clear(); svc = Real(FakeRepo()); req = make_request(childName="Emily", age=8, theme="dragons", moral="patience")
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return req
os.environ["GEMINI_PAGES_MODEL"] = "gemini-3.8-flash"; os.environ["GEMINI_PLANNER_MODEL"] = "gemini-3.8-flash"
os.environ["GEMINI_DRAFT_MODEL"] = "gemini-3.1-pro-preview"
req = asyncio.run(run()); tr = Q.get_trace(req)
by = {}
for role, model, th in SEEN: by.setdefault(role, set()).add(model)
ok(by.get("draft") == {"gemini-3.1-pro-preview"}, f"Pages 3-7 draft and rewrites on Pro ({by.get('draft')})")
ok(by.get("pages") == {"gemini-3.8-flash"}, f"Page 2 stays on Flash ({by.get('pages')})")
ok(by.get("page1") == {"gemini-3.8-flash"} and by.get("judge") == {"gemini-3.8-flash"} and by.get("scorer") == {"gemini-3.8-flash"}, f"Page 1, judge, scorer off the retiring 2.5 Flash ({by.get('page1')}, {by.get('judge')}, {by.get('scorer')})")
ok(all(th == 2048 for r, m, th in SEEN if r == "draft"), "Pro thinking capped at 2,048")
ok(Q.draft_timeout_seconds() == 120 and Q.rewrite_timeout_seconds() == 90, "longer time limits for the Pro draft")
ok("draft" in tr["usage"] and tr["usage"]["draft"]["model"] == "gemini-3.1-pro-preview", "Pro token use recorded separately (for the cost check)")
del os.environ["GEMINI_DRAFT_MODEL"]
req = asyncio.run(run())
by = {}
for role, model, th in SEEN: by.setdefault(role, set()).add(model)
ok(by.get("draft") == {"gemini-3.8-flash"} and Q.draft_timeout_seconds() == Q.DRAFT_TIMEOUT_SECONDS, "no draft model set: drafting uses the pages model, as before")
# Pro too slow -> draft fails -> page-by-page fallback still completes
os.environ["GEMINI_DRAFT_MODEL"] = "gemini-3.1-pro-preview"; os.environ["FAKE_DRAFT_EMPTY"] = "1"
repo = FakeRepo(); svc = Real(repo); r2 = make_request(childName="Emily", age=8, theme="dragons", moral="patience")
d = asyncio.run(svc.generate_story_first_page(r2, None)); asyncio.run(svc.complete_story_background(request=r2, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7))
ok(Q.get_trace(r2).get("final_status") == "complete", "if the Pro draft fails, the story still completes the normal way")
for k in ("GEMINI_DRAFT_MODEL", "FAKE_DRAFT_EMPTY", "GEMINI_PAGES_MODEL", "GEMINI_PLANNER_MODEL"): os.environ.pop(k, None)
