import asyncio, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from harness import *  # noqa

failures = []
def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info and not cond else ""))
    if not cond: failures.append(name)

# --- Fix 4: safety ---------------------------------------------------------
for bad in ["Sexy", "Erotic", "Nazi skinhead", "Prostitution", "Violence", "violencia"]:
    check(f"unsafe blocked: {bad}", Q.unsafe_custom_text(bad) is not None)
for ok in ["Soccer", "Rabbit🐰", "Super jacob", "Teamwork", "Honesty", "kindness", "cultural differences", "Mermaids", "skills", "Sussex"]:
    check(f"safe allowed: {ok}", Q.unsafe_custom_text(ok) is None, Q.unsafe_custom_text(ok))

# --- Fix 4: custom theme seed matching (age 9 real pool) --------------------
svc = StoryService(FakeRepo())
for theme, custom, expect_prefix in [
    ("dragons", None, "theme_filtered"), ("animals", None, "theme_filtered"),
    ("custom", "Soccer", "custom_keyword:adventure"), ("custom", "Mermaids", "custom_keyword:underwater"),
    ("custom", "Rabbit🐰", "custom_keyword:animals"), ("custom", "Farm", "custom_keyword:animals"),
    ("custom", "Super jacob", "custom_keyword:adventure"), ("custom", "cultural differences", "custom_neutral"),
    ("superheroes", None, "theme_filtered"), ("emotions", None, "theme_filtered"),
]:
    fams = set()
    for _ in range(40):
        r = make_request(theme=theme, customTheme=custom)
        svc._select_opening_seed(r)
        tr = Q.get_trace(r)["seed"]; fams.add(tr["family"])
    check(f"seed pool {theme}/{custom} -> {tr['pool']}", tr["pool"].startswith(expect_prefix), tr["pool"])
    allowed, _ = Q.resolve_theme_seed_families(theme, custom)
    check(f"seed families within theme for {theme}/{custom}", allowed is None or fams <= allowed, fams)
    if theme == "animals": print("   animals age-9 families:", sorted(fams))

# Cached: Page 1 prompt and fallback use same seed
r = make_request()
a = svc._select_opening_seed(r); b = svc._select_opening_seed(r)
check("seed cached per request", a is b)

# --- Fix 3: no repeats -------------------------------------------------------
rows = [{"title": "Emily and the Whispering Woods", "characters": [{"name": "Barnaby"}, {"name": "Squeaky"}, {"name": "Emily"}],
         "generation_trace": {"seed": {"family": "forest_school"}}}]
svc2 = StoryService(FakeRepo(rows))
seen = set()
for _ in range(60):
    r = make_request(childName="Emily", theme="animals", moral="bravery")
    seen.add(svc2._select_opening_seed(r)["family"])
check("recent seed family excluded", "forest_school" not in seen, seen)
r = make_request(childName="Emily", theme="animals", moral="bravery")
prompt = svc2._build_first_page_prompt(r, None)
check("page1 prompt has freshness block", "FRESHNESS" in prompt and "Emily and the Whispering Woods" in prompt and "Barnaby" in prompt)
check("child's own name not in avoid list", "names for any new side character" in prompt and ", Emily" not in prompt.split("names for any new side character")[1].split("\n")[0])
r2 = make_request(childName="Emily", companionId="luna_owl")
check("companion name kept usable", "Luna" not in Q.avoid_names_for(r2, {"names": []}, {"short_name": "Luna", "name": "Luna the Moon Owl"}))
check("rule fingerprints recorded", len(Q.get_trace(r).get("rules", {})) >= 11 and "oxford_vocabulary" in Q.get_trace(r)["rules"])

os.environ["NO_TRACE_COL"] = "1"
svc3 = StoryService(FakeRepo(rows))
h = svc3._recent_story_history(make_request(childName="Emily"))
check("history works before migration", h["loaded"] and h["titles"])
svc3.persist_generation_trace(make_request(), "s1", "u1")  # must not raise
check("trace persist fails open before migration", True)
del os.environ["NO_TRACE_COL"]

# --- Fix 1/2/5: end-to-end background with fake Gemini ----------------------
async def e2e(age=9, theme="dragons", moral="kindness", custom_moral=None):
    CALLS.clear()
    repo = FakeRepo()
    svc = StoryService(repo)
    req = make_request(age=age, theme=theme, moral=moral, customMoral=custom_moral, storyType="parent_selected")
    data = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=data["title"],
                                        current_pages=data["pages"], companion=None, expected_pages=7)
    return req, repo, data

req, repo, data = asyncio.run(e2e())
kinds = [k for k, _ in CALLS]
print("   call order:", kinds)
trace = Q.get_trace(req)
final = [u for u in repo.updates if u.get("generation_status") == "complete"]
check("story completed with 7 pages", final and len(final[-1]["pages"]) == 7, repo.updates[-1:])
check("plan generated before first continuation", "plan" in kinds and kinds.index("plan") < kinds.index("continuation"))
check("plan status ok", trace.get("plan_status") == "ok", trace.get("plan_status"))
check("plan with banned name was renamed", trace.get("plan_attempts") == 1 and trace.get("plan_renamed") and "Barnaby" not in json.dumps(trace.get("plan")) and not trace.get("plan_banned_names_kept"), (trace.get("plan_attempts"), trace.get("plan_banned_names_kept")))
check("plan prompt has new stakes/reveal rules", any("HEAR the stakes" in p and "REVEALED ALOUD" in p and "cost the child something" in p for k, p in CALLS if k == "plan"))
check("banned names on later pages renamed (none left in story)", trace.get("banned_names_in_story") == [] and set(trace.get("continuation_renamed", {})) | set(trace.get("plan_renamed", {})) == {"Barnaby", "Pip"}, (trace.get("banned_names_in_story"), trace.get("continuation_renamed")))
cont_prompts = [p for k, p in CALLS if k == "continuation"]
check("continuation prompts carry locked plan", all("LOCKED STORY PLAN" in p for p in cont_prompts))
check("continuation prompts carry page beat", all("MUST deliver this beat" in p for p in cont_prompts))
check("page 7 prompt says final page", any("This is the final page" in p for p in cont_prompts))
review_prompts = [p for k, p in CALLS if k == "review"]
check("page7 reviewer sees the plan", review_prompts and "LOCKED STORY PLAN" in review_prompts[0])
check("score recorded", (trace.get("score") or {}).get("overall") == 8.5, trace.get("score_status"))
check("score runs after pages saved", kinds[-1] == "score" or kinds.index("score") > max(i for i, k in enumerate(kinds) if k == "review"))
persisted = [u["generation_trace"] for u in repo.updates if "generation_trace" in u]
check("trace persisted", persisted and persisted[-1].get("final_status") == "complete")
check("trace has seed/rules/page7 review", all(k in persisted[-1] for k in ("seed", "rules", "plan", "score")) and any(e["event"] == "page7_review" for e in persisted[-1]["events"]))
json.dumps(persisted[-1])  # must be JSON serialisable
check("trace JSON serialisable", True)

# custom moral reaches prompts (route sets request.moral before service)
req, repo, data = asyncio.run(e2e(moral="Teamwork"))
check("custom moral in plan prompt", any("Private moral target: Teamwork" in p for k, p in CALLS if k == "plan"))
check("literal 'other' never in prompts", not any("Moral: other" in p or "moral: other" in p for _, p in CALLS))

# very young: no plan/score
req, repo, data = asyncio.run(e2e(age=2, theme="animals"))
check("age 2: plan not applicable", Q.get_trace(req).get("plan_status") == "not_applicable")

# plan disabled flag
os.environ["BEDTIME_STORY_PLAN_ENABLED"] = "0"
req, repo, data = asyncio.run(e2e())
check("plan kill-switch works", Q.get_trace(req).get("plan_status") == "disabled")
del os.environ["BEDTIME_STORY_PLAN_ENABLED"]

print("\nFAILURES:", failures if failures else "none")
