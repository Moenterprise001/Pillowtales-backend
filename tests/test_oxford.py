import asyncio, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
async def run(age):
    svc = StoryService(FakeRepo()); req = make_request(childName="Emily", age=age, theme="dragons", moral="patience")
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return req
for age, band in ((8, "7-8"), (5, "5-6"), (10, "9-10")):
    CALLS.clear(); req = asyncio.run(run(age))
    p1 = [p for k, p in CALLS if k == "page1"][0]; cont = [p for k, p in CALLS if k == "continuation"]; sc = [p for k, p in CALLS if k == "score"]
    ok(f"AGE VOCABULARY ENGINE — AGE {band}" in p1, f"age {age}: Page 1 gets the Oxford vocabulary engine ({band})")
    ok(all(f"OXFORD-INSPIRED AGE PROFILE — AGE {band}" in p and f"AGE VOCABULARY ENGINE — AGE {band}" in p for p in cont), f"age {age}: every page 2-7 prompt gets the Oxford profile + vocabulary ({band})")
    ok(not any("AGE CALIBRATION FOR PAGES 2-7" in p for p in cont), f"age {age}: the conflicting short age block is gone")
    ok(sc and f"AGE VOCABULARY ENGINE — AGE {band}" in sc[0], f"age {age}: scorer judges age fit against the same Oxford band")
    ok(Q.get_trace(req)["score"]["rubric_version"] == "20261112" and "oxford_vocabulary" in Q.get_trace(req)["rules"], f"age {age}: rubric version + Oxford fingerprints logged")
    if age == 8:
        words = [len(p.split()) for p in cont]; print(f"   age 8 page prompt size now {min(words)}-{max(words)} words")
