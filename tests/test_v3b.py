import asyncio, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
import harness
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
orig = StoryService._generate_content_sync
def sparkle_pages(self, prompt, schema=None, max_tokens=None, thinking=None):
    r = orig(self, prompt, schema, max_tokens, thinking)
    if CALLS and CALLS[-1][0] == "continuation":
        d = json.loads(r.text); d["pages"] = [p.replace("Curiosity bubbled", "A pony named Sparkle trotted by. Sparkle winked. Curiosity bubbled").replace("Aifric sat by", "Sparkle neighed. Aifric sat by") for p in d["pages"]]
        return Resp(json.dumps(d))
    return r
StoryService._generate_content_sync = sparkle_pages
async def run(req):
    repo = FakeRepo(); svc = StoryService(repo)
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return repo
req = make_request(); repo = asyncio.run(run(req))
pages = [u for u in repo.updates if u.get("generation_status") == "complete"][-1]["pages"]
tr = Q.get_trace(req)
new = tr.get("continuation_renamed", {}).get("Sparkle")
ok(new and not any("Sparkle" in p for p in pages), f"Sparkle renamed on later pages -> {new}")
ok(sum(p.count(new or "@@") for p in pages) >= 3, "same replacement name used consistently across pages")
ok(tr.get("banned_names_in_story") == ["Pip"] or "Sparkle" not in (tr.get("banned_names_in_story") or []), f"final banned list {tr.get('banned_names_in_story')}")
ok(len(pages) == 7 and tr.get("final_status") == "complete", "story completes")
StoryService._generate_content_sync = orig

# custom theme opens directly in the theme
svc = StoryService(FakeRepo())
req = make_request(childName="Emily", age=8, theme="custom", customTheme="Horse summer camp", moral="bravery")
p = svc._build_first_page_prompt(req, None)
ok('Open directly inside the parent\'s chosen theme: "Horse summer camp"' in p and "theme-matched opening idea" not in p, "custom theme: Page 1 opens inside the theme")
ok(Q.get_trace(req)["seed"].get("prompt_opening") == "custom_theme_direct", "logged as custom_theme_direct")
req2 = make_request(theme="dragons")
p2 = svc._build_first_page_prompt(req2, None)
ok("theme-matched opening idea" in p2, "preset theme keeps curated opening")
