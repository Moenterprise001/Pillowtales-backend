import asyncio, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
# role words
ok(Q.clean_character_name("Princess Lily") == "Lily" and Q.clean_character_name("The Princess") == "" and Q.clean_character_name("Princess") == "" and Q.clean_character_name("Captain Thistle") == "Thistle" and Q.clean_character_name("Old Nell") == "Nell" and Q.clean_character_name("Pipit") == "Pipit", "role words stripped from saved character names")
r = make_request(childName="Emily", theme="princess")
banned = Q.avoid_names_for(r, {"names": ["The Princess", "Princess", "Princess Lily", "Captain Thistle", "Pipit"]}, None)
ok("Princess" not in banned and "The Princess" not in banned and "Lily" in banned and "Thistle" in banned and "Pipit" in banned, "'Princess' no longer banned; real names still are")
# plan rename instead of re-plan
CALLS.clear()
rows = [{"title": "Emily and the Whispering Map", "characters": [{"name": "Barnaby"}, {"name": "The Princess"}], "generation_trace": {}},
        {"title": "Emily and the Forest Feast", "characters": [], "generation_trace": {}}]
class Repo(FakeRepo):
    def update(self, sid, uid, v): self.updates.append(json.loads(json.dumps(v, default=str))); return v
async def run(rows, **kw):
    CALLS.clear(); repo = Repo(rows); svc = StoryService(repo); req = make_request(**{"childName": "Emily", "age": 8, "theme": "dragons", "moral": "patience", **kw})
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return req, repo, d
req, repo, d = asyncio.run(run(rows)); tr = Q.get_trace(req)
plans = [k for k, _ in CALLS if k == "plan"]
ok(len(plans) == 1 and tr.get("plan_renamed", {}).get("Barnaby") and "Barnaby" not in json.dumps(tr["plan"]), f"banned name in plan swapped, no second plan call ({len(plans)} plan calls, {tr.get('plan_renamed')})")
fin = [u for u in repo.updates if u.get("generation_status") == "complete"][-1]["pages"]
ok(not any("Barnaby" in p for p in fin), "banned name never reaches the story")
# titles: flag repeats and overused words; judge's fresh title used
ok(Q.title_flags("Emily and the Whispering Map", ["Emily and the Whispering Map"], "Emily") == ["repeats a recent title", "overused word 'Whispering'"], "repeat + overused word flagged")
ok(Q.title_flags("Emily and the Forest Feast", ["Emily's Forest Feast"], "Emily") == ["repeats a recent title"], "near-identical repeat flagged")
ok(Q.title_flags("Emily and the Ribbon Mail", ["Emily and the Forest Feast"], "Emily") == [], "fresh title not flagged")
rows2 = [{"title": "Aifric and the Dream Dragon (strange)", "characters": [], "generation_trace": {}}] + [{"title": f"Aifric and the Dream Dragon ({w})", "characters": [], "generation_trace": {}} for w in ["small","job","mistake","misunderstanding","rule","challenge","promise","visitor","found","funny","strange","unexpected","tiny"]]
os.environ["FAKE_FRESH_TITLE"] = "Aifric and the Hiccuping Egg"; os.environ["FAKE_P1_TITLE"] = "Aifric and the Whispering Map"; rows2 = [{"title": "Aifric and the Whispering Map", "characters": [], "generation_trace": {}}]
req, repo, d, = asyncio.run(run(rows2, childName="Aifric"))[:3]; tr = Q.get_trace(req)
jp = [p for k, p in CALLS if k == "page1_judge"][0]
ok("TITLE PROBLEM: repeats a recent title" in jp, "judge sees the title problem")
ok(d["title"] == "Aifric and the Hiccuping Egg" and tr["page1_best_of"].get("title_replaced"), f"repeated title replaced by the judge's fresh title ({d['title']!r})")
os.environ["FAKE_FRESH_TITLE"] = "Aifric and the Secret Egg"
req, repo, d, = asyncio.run(run(rows2, childName="Aifric"))[:3]; tr = Q.get_trace(req)
ok(d["title"] != "Aifric and the Secret Egg" and not tr["page1_best_of"].get("title_replaced"), "a fresh title with its own problem is not used")
del os.environ["FAKE_FRESH_TITLE"]; del os.environ["FAKE_P1_TITLE"]
req, repo, d = asyncio.run(run([])); ok(not Q.get_trace(req)["page1_best_of"].get("title_replaced"), "clean title left alone")
