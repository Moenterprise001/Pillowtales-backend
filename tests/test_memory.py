import asyncio, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
class Repo(FakeRepo):
    def update(self, sid, uid, v): self.updates.append(json.loads(json.dumps(v, default=str))); return v
async def make(repo, sid="s1", **kw):
    svc = StoryService(repo); req = make_request(**{"childName": "Emily", "age": 8, "theme": "dragons", "moral": "patience", **kw})
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id=sid, title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return req
repo = Repo()
req = asyncio.run(make(repo))
mem = repo.client.memory
ok(len(mem) == 1 and mem[0]["story_id"] == "s1" and mem[0]["child_name"] == "Emily" and mem[0]["seed_family"] == Q.get_trace(req)["seed"]["family"] and mem[0]["page1_hook"], f"finished story remembered: {mem[0] if mem else None}")
ok(set(mem[0]) - {"created_at"} == {"user_id", "child_name", "story_id", "title", "seed_family", "page1_hook", "plot_shape", "names"}, "only labels are kept, never story text")
# story leaves the library: memory still counts
old = [{"story_id": f"gone{i}", "created_at": f"2026-11-0{i}T20:00:00", "title": f"Emily and the Gone Story {i}", "seed_family": fam, "page1_hook": "job", "names": [f"Gone{i}"]}
       for i, fam in enumerate(["dragon_market", "dragon_post_office", "dragon_nursery"], start=1)]
h = Q.load_recent_history(FakeRepo([], memory=old).client.__class__ and FakeRepo([], memory=old), "u1", "Emily")
ok({"dragon_market", "dragon_post_office", "dragon_nursery"} <= h["families"] and "Emily and the Gone Story 1" in h["titles"] and "Gone1" in h["names"], "stories no longer in the library still block repeats (openings, titles, names)")
# no double counting when the story is still in the library
lib = [{"id": "s9", "created_at": "2026-11-10T20:00:00", "title": "Emily and the Kept Story", "characters": [{"name": "Kept"}], "seed_family": "dragon_bakery"}]
h2 = Q.load_recent_history(FakeRepo(lib, memory=[{"story_id": "s9", "created_at": "2026-11-10T20:00:00", "title": "Emily and the Kept Story", "seed_family": "dragon_bakery", "names": ["Kept"]}]), "u1", "Emily")
ok(h2["titles"].count("Emily and the Kept Story") == 1, "a story in both the library and memory is counted once")
# newest first, capped at 30 across both
many = [{"story_id": f"m{i}", "created_at": f"2026-10-{(i % 28) + 1:02d}T10:{i:02d}:00", "title": f"T{i}", "seed_family": None, "names": []} for i in range(40)]
h3 = Q.load_recent_history(FakeRepo(lib, memory=many), "u1", "Emily")
ok(len(h3["titles"]) == 30 and h3["titles"][0] == "Emily and the Kept Story", "30 most recent across library + memory, newest first")
# fails open before the table exists
os.environ["NO_MEMORY_TABLE"] = "1"
repo = Repo(); req = asyncio.run(make(repo))
tr = Q.get_trace(req)
ok(tr.get("final_status") == "complete" and str(tr.get("story_memory_saved", "")).startswith("failed"), "missing table never breaks a story")
del os.environ["NO_MEMORY_TABLE"]
# Story Worlds are not remembered here
repo = Repo()
r = make_request(childName="Emily", age=8, theme="dragons", storyWorldSlug="irish-myths")
StoryService(repo)._remember_story(r, "sw1", "u1", "x", None)
ok(repo.client.memory == [], "Story Worlds stories are not written to Bedtime memory")
