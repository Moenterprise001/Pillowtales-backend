import asyncio, sys, os, time, types as pytypes
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
import harness
deps = pytypes.ModuleType("app.api.deps")
for n in ["get_current_user","get_story_repo","get_story_service","get_subscription_service","get_user_repo"]:
    setattr(deps, n, lambda *a, **k: None)
sys.modules["app.api.deps"] = deps
from app.api.routes import stories as route
import json
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)

# 1) plan uses thinking on first attempt
async def run_bg(req=None):
    repo = FakeRepo(); svc = StoryService(repo); req = req or make_request()
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return req, repo
harness.THINKING.clear()
req, repo = asyncio.run(run_bg())
ok(harness.THINKING[:1] == [2048], f"plan call 1 uses thinking budget 2048 (got {harness.THINKING})")
ok(any("SELF-CHECK BEFORE ANSWERING" in p for k, p in CALLS if k == "plan"), "plan prompt has self-check")
ok(Q.get_trace(req).get("final_status") == "complete", "story still completes")

# 2) slow thinking plan falls back to fast plan
orig = StoryService._generate_content_sync
def slow_thinking(self, prompt, schema=None, max_tokens=None, thinking=None):
    if prompt.startswith("You are the story editor") and thinking:
        time.sleep(1.5)
    return orig(self, prompt, schema, max_tokens, thinking)
StoryService._generate_content_sync = slow_thinking
Q.PLAN_TIMEOUT_SECONDS = 0.5
req, repo = asyncio.run(run_bg())
tr = Q.get_trace(req)
ok(tr.get("plan_status") == "ok" and tr.get("plan_thinking_budget") == 0 and any(e["event"] == "plan_thinking_timeout" for e in tr["events"]), "slow thinking plan -> fast plan fallback")
StoryService._generate_content_sync = orig; Q.PLAN_TIMEOUT_SECONDS = 20

# 3) Page 1 'Barnaby' renamed before save, via route
def barnaby_page1(self, prompt, schema=None, max_tokens=None, thinking=None):
    if "Write Page 1 only" in prompt:
        return Resp(json.dumps({"title": "Emily and Barnaby the Bee", "pages": ["Emily met a bee named Barnaby. Barnaby looked sad, and his voice was squeaky."]}))
    return orig(self, prompt, schema, max_tokens, thinking)
StoryService._generate_content_sync = barnaby_page1
class UserRepo:
    def get_profile(self, uid): return {"email": "x"}
class SubSvc:
    def get_subscription(self, *a): return None
class StoryRepo(FakeRepo):
    def insert(self, rec): self.inserted = rec; return {"id": "s1", **rec}
async def call(**kw):
    sr = StoryRepo(); svc = StoryService(sr)
    svc.validate_story_limits = lambda *a: None
    svc.complete_story_background = lambda **k: asyncio.sleep(0)
    req = make_request(**kw)
    resp = await route.generate_story(req, user_id="u1", user_repo=UserRepo(), story_repo=sr, story_service=svc, subscription_service=SubSvc())
    return req, sr, resp
req, sr, resp = asyncio.run(call(childName="Emily", theme="animals"))
rec = sr.inserted
ok("Barnaby" not in rec["title"] and "Barnaby" not in rec["pages"][0] and "squeaky" in rec["pages"][0], f"Page 1 + title renamed before save: {rec['title']!r}")
ok(resp.pages[0] == rec["pages"][0] and resp.title == rec["title"], "app receives the renamed Page 1")
ok(Q.get_trace(req).get("page1_renamed", {}).get("Barnaby"), f"rename logged {Q.get_trace(req).get('page1_renamed')}")
# companion name is respected
req, sr, resp = asyncio.run(call(childName="Barnaby", theme="animals"))
ok("Barnaby" in sr.inserted["pages"][0], "child named Barnaby is never renamed")
StoryService._generate_content_sync = orig
