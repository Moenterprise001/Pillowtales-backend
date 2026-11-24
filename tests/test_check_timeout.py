import asyncio, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
async def run():
    repo = FakeRepo(); svc = StoryService(repo); req = make_request(childName="Emily", age=8, theme="dragons", moral="patience")
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return Q.get_trace(req)
Q.PLAN_FAST_TIMEOUT_SECONDS = 1; Q.PLAN_CHECK_TIMEOUT_SECONDS = 4; os.environ["FAKE_CHECK_SLOW"] = "2"
tr = asyncio.run(run())
ok(not str(tr["plan_check"]["status"]).startswith("error"), f"a plan check slower than the old limit still completes ({tr['plan_check']['status']}, {tr['plan_check']['ms']} ms)")
Q.PLAN_CHECK_TIMEOUT_SECONDS = 1
tr = asyncio.run(run())
ok(tr["plan_check"]["status"] == "error:TimeoutError" and tr.get("final_status") == "complete", f"a real timeout is named and the story still completes ({tr['plan_check']['status']})")
ok("fresh_title" in Q.PAGE1_JUDGE_SCHEMA["required"], "judge must always answer fresh_title (empty when not needed)")
