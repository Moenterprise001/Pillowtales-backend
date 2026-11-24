import asyncio, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
async def run(req):
    repo = FakeRepo(); svc = StoryService(repo)
    d = await svc.generate_story_first_page(req, None)
    svc.freshen_page_one_names(req, d)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
CALLS.clear(); req = make_request(childName="Emily", age=8, theme="adventure", moral="bravery"); asyncio.run(run(req)); tr = Q.get_trace(req)
p1 = [p for k, p in CALLS if k == "page1"][0]
ok("do not use these overused title words: Whispering, Secret, Lost" in p1, "Page 1 gets the overused title words")
ok("Never put the moral or a form of it in the title" in p1, "Page 1 told not to put the moral in the title")
chk = [p for k, p in CALLS if k == "plan_check"][0]
ok("threads_answered_by_page6" in chk, "plan checker checks thread answers are placed by Page 6")
ok(any("ANSWER EVERYTHING BY PAGE 6" in p for k, p in CALLS if k == "plan"), "plan prompt has the answer-by-Page-6 rule")
sc = [p for k, p in CALLS if k == "score"][0]
ok("bedtime-safe stake" in sc and "Do not reward danger or fear" in sc, "scorer uses bedtime-safe stakes")
ok(tr["score"]["rubric_version"] == "20261112", "new rubric version logged")
ok(tr.get("title_overused_words") == [], f"title words logged: {tr.get('title_overused_words')}")
ok(Q.overused_title_words_in("Emily and the Whispering Map") == ["Whispering"], "detects 'Whispering' in a title")
ok(tr.get("final_status") == "complete", "story completes")
