import asyncio, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
async def main():
    repo = FakeRepo(); svc = StoryService(repo); req = make_request()
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    print([u["generation_trace"]["final_status"] for u in repo.updates if "generation_trace" in u])
asyncio.run(main())
