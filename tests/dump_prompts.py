import sys, os, asyncio, json
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
OUT = sys.argv[2]
async def main():
    svc = StoryService(FakeRepo()); req = make_request(childName="Emily", age=8, theme="princess", moral="sharing")
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
asyncio.run(main())
seen = {}
for i, (k, p) in enumerate(CALLS):
    seen.setdefault(k, []).append(p)
for k, ps in seen.items():
    for j, p in enumerate(ps):
        open(f"{OUT}/{k}_{j+1}.txt", "w").write(p)
    print(k, len(ps), "calls; words per prompt:", [len(p.split()) for p in ps])
