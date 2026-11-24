import sys, os, asyncio, random
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
OUT = sys.argv[2]
async def main():
    random.seed(7)
    svc = StoryService(FakeRepo()); req = make_request(childName="Emily", age=8, theme="princess", moral="patience")
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
asyncio.run(main())
for i, (k, p) in enumerate(CALLS):
    if k in ("page1", "continuation", "plan"):
        open(f"{OUT}/{i:02d}_{k}.txt", "w").write(p)
print(len(CALLS), "calls")
