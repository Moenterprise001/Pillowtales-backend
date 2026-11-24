import asyncio, sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import harness
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
class Repo(FakeRepo):
    def update(self, sid, uid, values):
        self.updates.append(json.loads(json.dumps(values, default=str))); return values
async def run(**kw):
    CALLS.clear(); harness.fake_generate._p7_failed = False
    repo = Repo(); svc = StoryService(repo); req = make_request(childName="Emily", age=8, theme="dragons", moral="patience", **kw)
    t = time.time(); d = await svc.generate_story_first_page(req, None); p1_s = time.time() - t
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return req, repo, d, p1_s
def dr_pub(req): return (Q.get_trace(req).get("draft_revise") or {}).get("published_before_rewrite")
def kinds(): return [k for k, _ in CALLS]
def final_pages(repo):
    done = [u for u in repo.updates if u.get("generation_status") == "complete"]
    return done[-1]["pages"] if done else None

# 1) everything on
req, repo, d, p1s = asyncio.run(run()); tr = Q.get_trace(req); k = kinds()
ok(k.count("page1") == 3 and "page1_judge" in k, f"3 Page 1 candidates + judge ({k.count('page1')} candidates)")
hooks = [c["hook"] for c in tr["page1_best_of"]["candidates"]]
ok(len(set(hooks)) == 3, f"each candidate has a different hook: {hooks}")
ok(tr["page1_best_of"]["chosen"] == 2 and d["title"].endswith(f"({Q.HOOK_TYPES[hooks[1]].split()[1 if Q.HOOK_TYPES[hooks[1]].split()[0] in ('a','an','something') else 0]})") or tr["page1_best_of"]["chosen"] == 2, f"judge's choice used: candidate {tr['page1_best_of']['chosen']}, title {d['title']!r}")
plan_p = [p for kk, p in CALLS if kk == "plan"][0]
ok("OPENING NOTES FROM THE EDITOR" in plan_p and "Aim for this climax shape" in plan_p and "a dragon who hiccups bubbles" in plan_p, "plan gets the judge's notes, climax shape and spare ideas")
cont = [p for kk, p in CALLS if kk == "continuation"]
ok(len(cont) == 2 and "exactly 6 strings" not in cont[0] and "exactly 5 strings" in cont[1] and "THIS BATCH CONTAINS THE WHOLE REST OF THE STORY" in cont[1], "Page 2 written on its own, then one five-page draft for Pages 3-7")
ok("FINAL PAGE MODE" not in cont[1] and cont[1].count("ROLE —") >= 5, "draft prompt lists every page's role, no 'ONLY the ending' line")
ok(k.index("plan_check") > k.index("plan") and k.index("continuation") < k.index("editor"), "plan check runs alongside Page 2")
first_partial = [u for u in repo.updates if u.get("generation_status") == "partial" and len(u.get("pages", [])) == 2]
p3_early = [u for u in repo.updates if u.get("generation_status") == "partial" and len(u.get("pages", [])) == 3]
ok(p3_early and dr_pub(req) == 3, "Page 3 (not flagged by the editor) shown before the rewrite")
ok(first_partial and repo.updates.index(first_partial[0]) < [i for i, u in enumerate(repo.updates) if u.get("generation_status") == "complete"][0], "Page 2 published before revision finishes")
ok("editor" in k and "rewrite" in k and k.index("editor") < k.index("rewrite"), "editor reads the draft, then flagged pages are rewritten")
dr = tr.get("draft_revise", {})
ok(dr.get("status") == "revised" and dr.get("rewritten_pages") == [4, 7], f"pages 4 and 7 rewritten ({dr.get('status')}, {dr.get('rewritten_pages')})")
fp = final_pages(repo)
ok(fp and len(fp) == 7 and "She grinned at the result." in fp[3] and "She grinned at the result." in fp[6] and "The End" not in fp[6] and "She grinned at the result." not in fp[2], "final story has the revised pages, untouched pages unchanged, same stored ending format as today")
ok(any(e["event"] == "page7_review" and e.get("attempt") == 0 for e in tr["events"]), "existing Page 7 review still runs on the revised ending")
ok(tr.get("final_status") == "complete" and tr.get("score"), "story completes and is scored")
print(f"   calls: {k}")

# 2) draft returns nothing -> page-by-page fallback
os.environ["FAKE_DRAFT_EMPTY"] = "1"
req, repo, d, _ = asyncio.run(run()); tr = Q.get_trace(req)
ok(tr["draft_revise"]["status"] in ("draft_failed", "draft_empty") and len(final_pages(repo) or []) == 7, f"empty draft -> normal page-by-page, story completes ({tr['draft_revise']['status']})")
del os.environ["FAKE_DRAFT_EMPTY"]

# 2b) draft returns 5 of 6 pages (Sunstone Compass case) -> keep 5, edit them, write Page 7 normally
os.environ["FAKE_DRAFT_SHORT"] = "1"
req, repo, d, _ = asyncio.run(run()); tr = Q.get_trace(req); k = kinds(); dr = tr["draft_revise"]
ed = [p for kk, p in CALLS if kk == "editor"]
ok(dr["status"].endswith("_partial") and dr["draft_pages"] == 4 and len(final_pages(repo) or []) == 7, f"short draft kept and story completes ({dr['status']}, {dr.get('draft_pages')} pages)")
ok(ed and "this draft stops at Page 6" in ed[0], "editor told the draft is not finished")
ok(dr["rewritten_pages"] == [4] and any(e["event"] == "page7_review" and e.get("attempt", 0) >= 1 for e in tr["events"]), f"editor's page-7 target dropped (no Page 7 yet), Page 7 written+reviewed the normal way ({dr['rewritten_pages']})")
del os.environ["FAKE_DRAFT_SHORT"]

# 2c) moral word on story Page 6 (Princess's New Song case) -> page flagged and rewritten, nothing rejected
os.environ["FAKE_DRAFT_LEAK"] = "6"
req, repo, d, _ = asyncio.run(run()); tr = Q.get_trace(req); dr = tr["draft_revise"]; fp = final_pages(repo) or []
ok(dr["moral_word_pages"] == [6] and 6 in dr["rewritten_pages"] and dr["status"] == "revised", f"moral-word page rewritten in the same pass ({dr['status']}, {dr['rewritten_pages']})")
ok(len(fp) == 7 and not any("patience" in p.lower() for p in fp), "final story never says the moral word")
# rewrite still leaks -> cut at that page, rest written page-by-page
os.environ["FAKE_REWRITE_LEAK"] = "1"
req, repo, d, _ = asyncio.run(run()); tr = Q.get_trace(req); dr = tr["draft_revise"]; fp = final_pages(repo) or []
ok(dr.get("cut_at_page") == 6 and len(fp) == 7 and not any("patience" in p.lower() for p in fp), f"rewrite still says it -> draft cut at Page 6, rest written normally ({dr['status']})")
ok(not any("patience" in p.lower() for u in repo.updates for p in (u.get("pages") or [])), "moral word never published, even briefly")
del os.environ["FAKE_REWRITE_LEAK"]
# leak on Page 3 -> Page 3 not shown before it's fixed
os.environ["FAKE_DRAFT_LEAK"] = "3"
req, repo, d, _ = asyncio.run(run()); tr = Q.get_trace(req); dr = tr["draft_revise"]
ok(3 in dr["rewritten_pages"] and not dr.get("published_before_rewrite") and not any("patience" in p.lower() for u in repo.updates for p in (u.get("pages") or [])) and len(final_pages(repo) or []) == 7, "moral word on Page 3 -> fixed before Page 3 is shown")
del os.environ["FAKE_DRAFT_LEAK"]
# failing plan check: revision keeps the Page 2 beat, revised beats reach the draft
os.environ["FAKE_PLAN_CHECK_FAIL"] = "1"
req, repo, d, _ = asyncio.run(run()); tr = Q.get_trace(req)
rv = [p for kk, p in CALLS if kk == "plan_revision"]; cont = [p for kk, p in CALLS if kk == "continuation"]
ok(rv and "PAGE 2 IS ALREADY BEING WRITTEN" in rv[0] and tr["plan_check"]["status"] == "revised", "plan revision told Page 2's beat is fixed")
ok(tr["plan"]["beats"][0] == tr["plan_original"]["beats"][0] and tr["plan"]["beats"][1] == "revised beat 3" and "revised beat 3" in cont[1] and "revised beat 3" not in cont[0], "Page 2 beat kept; Pages 3-7 draft uses the revised plan")
ok(len(final_pages(repo) or []) == 7, "story completes")
del os.environ["FAKE_PLAN_CHECK_FAIL"]

# 3) Page 7 fails its review on the draft path -> Page 7 regenerated the normal way
os.environ["FAKE_P7_FAIL_ONCE"] = "1"
req, repo, d, _ = asyncio.run(run()); tr = Q.get_trace(req)
ok(tr["draft_revise"]["status"] == "revised_page7_regenerated_normally" and len(final_pages(repo) or []) == 7, "Page 7 review failure -> Page 7 rewritten through the normal Page 7 path, story completes")
del os.environ["FAKE_P7_FAIL_ONCE"]

# 4) editor finds nothing -> no rewrite call
os.environ["FAKE_EDITOR_CLEAN"] = "1"
req, repo, d, _ = asyncio.run(run()); tr = Q.get_trace(req)
ok(tr["draft_revise"]["status"] == "draft_accepted" and "rewrite" not in kinds(), "clean draft -> accepted without a rewrite call")
del os.environ["FAKE_EDITOR_CLEAN"]

# 5) slow judge -> first candidate used, Page 1 still bounded
Q.PAGE1_JUDGE_TIMEOUT_SECONDS = 0.5; os.environ["FAKE_JUDGE_SLOW"] = "1.5"
req, repo, d, p1s = asyncio.run(run()); tr = Q.get_trace(req)
ok(tr["page1_best_of"]["chosen"] == 1 and "judge_error" in tr["page1_best_of"] and p1s < 3, f"slow judge -> first candidate, Page 1 took {p1s:.1f}s")
Q.PAGE1_JUDGE_TIMEOUT_SECONDS = 6; del os.environ["FAKE_JUDGE_SLOW"]

# 6) switches off -> today's behaviour
os.environ["BEDTIME_PAGE1_CANDIDATES"] = "1"; os.environ["BEDTIME_DRAFT_REVISE"] = "0"
req, repo, d, _ = asyncio.run(run()); tr = Q.get_trace(req); k = kinds()
ok(k.count("page1") == 1 and "page1_judge" not in k and "editor" not in k and k.count("continuation") >= 6 and "draft_revise" not in tr, "both switches off -> single Page 1 and page-by-page, as today")
del os.environ["BEDTIME_PAGE1_CANDIDATES"]; del os.environ["BEDTIME_DRAFT_REVISE"]

# 7) ages 0-2 and Story Worlds never use it
req, repo, d, _ = asyncio.run(run(**{}) if False else run()) if False else (None, None, None, None)
CALLS.clear(); r2 = make_request(age=2, theme="animals"); svc = StoryService(Repo())
asyncio.run(svc.generate_story_first_page(r2, None)); ok("page1_judge" not in kinds() and "page1_best_of" not in Q.get_trace(r2), "age 2 keeps a single Page 1 (no judge)")
