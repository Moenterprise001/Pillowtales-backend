import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
svc = StoryService(FakeRepo())
totals = {}
for age in range(0, 13):
    b = [svc._standard_bedtime_page_budget(age, n, "en") for n in range(1, 8)]
    totals[age] = sum((x["target_min"] + x["target_max"]) / 2 for x in b)
print("   words per story by age:", {a: int(t) for a, t in totals.items()})
ok(all(totals[a] <= totals[a + 1] for a in range(3, 12)), "story length grows (or holds) with every year from 3 to 12")
ok(350 <= totals[3] <= 450 and 450 <= totals[4] <= 600 and 600 <= totals[6] <= 750 and 800 <= totals[8] <= 900 and 900 <= totals[10] <= 1000, "ages 3, 4, 6, 8, 10 land in the planned ranges")
ok(svc._standard_bedtime_page_budget(10, 3, "en")["target_max"] == 155, "ages 9-10 unchanged")
def sentences(n_words, n_sent):
    per = max(1, n_words // n_sent)
    return " ".join(("Rob ran to see the duck. " * 0) + " ".join(["word"] * (per - 1)) + "." for _ in range(n_sent))
page60 = sentences(60, 6)
r3, r9 = make_request(age=3), make_request(age=9)
ok(bool(svc._valid_generated_pages([page60], 1, request=r3, start_page_number=3)), "a 60-word page is a full page at age 3")
ok(not svc._valid_generated_pages([page60], 1, request=r9, start_page_number=3), "...but too thin at age 9")
page120 = sentences(120, 10)
ok(not svc._valid_generated_pages([page120], 1, request=r3, start_page_number=3), "a 120-word page is too long at age 3")
p1 = svc._build_first_page_prompt(make_request(childName="Rob", age=3, theme="animals"), None)
ok("Target 35-50 words for Page 1" in p1, "Page 1 prompt asks a 3-year-old for 35-50 words")
os.environ["BEDTIME_PROMPTS_V5"] = "0"
ok(svc._standard_bedtime_page_budget(3, 3, "en")["target_max"] == 155 and not svc._valid_generated_pages([page60], 1, request=r3, start_page_number=3), "v5 off: old lengths and old thin-page rule")
del os.environ["BEDTIME_PROMPTS_V5"]
ok(svc._standard_bedtime_page_budget(3, 3, "ja")["unit_label"].startswith("English-equivalent"), "Japanese keeps its own sizing model")
import asyncio, json
class Repo(FakeRepo):
    def update(self, sid, uid, v): self.updates.append(json.loads(json.dumps(v, default=str))); return v
async def run(age):
    repo = Repo(); s2 = StoryService(repo); req = make_request(childName="Rob", age=age, theme="dragons", moral="kindness")
    d = await s2.generate_story_first_page(req, None)
    await s2.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    done = [u for u in repo.updates if u.get("generation_status") == "complete"]
    return done[-1]["pages"] if done else None
for age in (3, 4, 6, 8, 10, 12):
    pages = asyncio.run(run(age))
    words = [len(p.split()) for p in pages] if pages else None
    ok(pages and len(pages) == 7, f"age {age}: story completes, words per page {words}")
