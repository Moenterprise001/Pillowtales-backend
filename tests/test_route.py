import asyncio, sys, os, types as pytypes
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
deps = pytypes.ModuleType("app.api.deps")
for n in ["get_current_user","get_story_repo","get_story_service","get_subscription_service","get_user_repo"]:
    setattr(deps, n, lambda *a, **k: None)
sys.modules["app.api.deps"] = deps
from app.api.routes import stories as route

# 1) plan failure must not stop the story
orig = StoryService._generate_content_sync
def failing_plan(self, prompt, schema=None, max_tokens=None, thinking=None):
    if prompt.startswith("You are the story editor"): raise RuntimeError("gemini 503")
    return orig(self, prompt, schema, max_tokens, thinking)
StoryService._generate_content_sync = failing_plan
async def run_bg():
    repo = FakeRepo(); svc = StoryService(repo); req = make_request()
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return req, repo
req, repo = asyncio.run(run_bg())
done = [u for u in repo.updates if u.get("generation_status") == "complete"]
print("PASS" if done and len(done[-1]["pages"]) == 7 else "FAIL", "story completes when plan call fails;", Q.get_trace(req)["plan_status"])
StoryService._generate_content_sync = orig

# 2) route: unsafe custom theme rejected before quota/generation
class UserRepo:
    def get_profile(self, uid): return {"email": "x@y.z"}
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
try:
    asyncio.run(call(theme="custom", customTheme="Nazi skinhead"))
    print("FAIL unsafe theme not rejected")
except Exception as e:
    print("PASS" if getattr(e, "status_code", None) == 400 else "FAIL", "unsafe custom theme rejected:", getattr(e, "detail", e))
try:
    asyncio.run(call(moral="other", customMoral="Erotic"))
    print("FAIL unsafe moral not rejected")
except Exception as e:
    print("PASS" if getattr(e, "status_code", None) == 400 else "FAIL", "unsafe custom moral rejected")

CALLS.clear()
req, sr, resp = asyncio.run(call(moral="other", customMoral="Teamwork", storyType="parent_selected"))
p1 = [p for k, p in CALLS if k == "page1"][0]
print("PASS" if "Moral: Teamwork" in p1 and "Moral: other" not in p1 else "FAIL", "custom moral reaches Page 1 prompt")
print("PASS" if sr.inserted["moral"] == "Teamwork" else "FAIL", "saved moral = Teamwork")
tr = [u["generation_trace"] for u in sr.updates if "generation_trace" in u][-1]
print("PASS" if tr["story_type"] == "parent_selected" and tr["inputs"]["moral_sent"] == "other" and tr["page1"]["source"] == "gemini_primary" else "FAIL",
      "initial trace saved:", tr["story_type"], tr["inputs"]["effective_moral"], tr["page1"]["source"], tr["seed"])
req, sr, resp = asyncio.run(call())
tr = [u["generation_trace"] for u in sr.updates if "generation_trace" in u][-1]
print("PASS" if tr["story_type"] == "unknown" else "FAIL", "older app (no storyType) still works")
