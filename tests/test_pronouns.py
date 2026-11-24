import asyncio, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
for g, want in [("boy", "he"), ("Boy", "he"), ("male", "he"), ("girl", "she"), ("Female", "she"), ("neutral", None), ("", None), ("other", None)]:
    got = Q.child_pronouns(make_request(childName="Alex", age=7, gender=g))
    ok((got[0] if got else None) == want, f"gender {g!r} -> {got}")
alex = make_request(childName="Alex", age=7, gender="girl", theme="dragons", moral="kindness")
ok("Alex is a girl; always use she/her/her for Alex" in Q.pronoun_block(alex) and "must not change the kind of story" in Q.pronoun_block(alex), "pronoun rule, pronouns only")
CALLS.clear()
class Repo(FakeRepo):
    def update(self, sid, uid, v): self.updates.append(v); return v
async def run(req):
    svc = StoryService(Repo()); d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
asyncio.run(run(alex))
by = {}
for k, p in CALLS: by.setdefault(k, []).append(p)
ok(all("Alex is a girl" in p for p in by["page1"]), "every Page 1 option gets the pronoun rule")
ok("Alex is a girl" in by["plan"][0] and all("Alex is a girl" in p for p in by["continuation"]), "plan and every page prompt get it")
CALLS.clear(); asyncio.run(run(make_request(childName="Sam", age=7, theme="dragons", moral="kindness")))
ok(not any("PRONOUNS:" in p for _, p in CALLS), "gender not set (neutral): no pronoun rule, as before")
os.environ["BEDTIME_PROMPTS_V5"] = "0"; ok(Q.pronoun_block(alex) == "", "v5 off: unchanged"); del os.environ["BEDTIME_PROMPTS_V5"]
