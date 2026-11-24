import asyncio, sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
os.environ["FAKE_CHECK_SLOW"] = "2"; os.environ["FAKE_PAGE_SLOW"] = "1"
class Repo(FakeRepo):
    def __init__(s): super().__init__(); s.t = {}
    def update(s, sid, uid, v):
        n = len(v.get("pages") or []); s.t.setdefault(n, time.time()); return v
async def go(on):
    os.environ["BEDTIME_DRAFT_REVISE"] = "1" if on else "0"
    repo = Repo(); svc = StoryService(repo); req = make_request(childName="Emily", age=8, theme="dragons", moral="patience")
    d = await svc.generate_story_first_page(req, None); t0 = time.time()
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return {n: round(t - t0, 1) for n, t in sorted(repo.t.items()) if n >= 2}
new, old = asyncio.run(go(True)), asyncio.run(go(False))
print("page shown at (s) new:", new); print("page shown at (s) old:", old)
print(("PASS" if new[2] <= old[2] + 0.3 else "FAIL"), "Page 2 no later than today's flow (check runs alongside it)")
