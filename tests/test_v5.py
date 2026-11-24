import asyncio, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
async def run(req):
    repo = FakeRepo(); svc = StoryService(repo)
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return repo
# --- v5 on (default) ---
CALLS.clear(); req = make_request(childName="Emily", age=8, theme="princess", moral="patience"); repo = asyncio.run(run(req))
tr = Q.get_trace(req); cont = [p for k, p in CALLS if k == "continuation"]
ok(tr.get("prompt_version") == "v5", "v5 prompts used by default")
ok(tr.get("final_status") == "complete", "story completes on v5")
ok(all("STORY CRAFT — EACH RULE STATED ONCE" in p for p in cont), "every page prompt has the merged craft list")
ok(not any("Include at least one memorable magical detail that affects the story" in p for p in cont), "no 'magic on every page' rule")
ok(not any(p.count("- Do not use frightening danger or an unresolved cliffhanger.") for p in cont), "bare 'no frightening danger' line replaced")
ok(all("bedtime-safe is not stake-free" in p for p in cont), "stakes definition present")
ok(not any("PAGE 1 RULES" in p for p in cont) and all("OXFORD-INSPIRED AGE PROFILE" in p for p in cont), "Pages 2-7 get the Oxford age profile, not Page 1 rules")
ok(all('{"pages":["new page text","new page text"]}' in p and '{{"pages"' not in p for p in cont), "JSON example uses valid single braces")
ok(all("follow the page role and adjust the beat" in p for p in cont), "plan/page-role precedence line present")
ok(any("PAGE 6 COMPLETES THE SOLUTION" in p for k, p in CALLS if k == "plan"), "plan prompt has v5 plan rules")
ok(tr.get("plan_check", {}).get("status") == "passed", f"plan check ran: {tr.get('plan_check')}")
ok("v5_craft" in tr.get("rules", {}), "v5 craft block fingerprinted")
for flag in ("CHILDREN'S AUTHOR VOICE", "ELITE QUALITY GUARD", "NATURAL READ-ALOUD CADENCE", "STRICT LANGUAGE CLEANUP", "BEDTIME PROSE POLISH"):
    ok(not any(flag in p for p in cont), f"old block removed: {flag}")
ok(all("BEDTIME STORY FORWARD PROGRESSION — HARD REQUIREMENT" in p for p in cont), "forward-progression block kept verbatim")
v5_words = [len(p.split()) for p in cont]
# --- failing plan check -> revision ---
os.environ["FAKE_PLAN_CHECK_FAIL"] = "1"; CALLS.clear()
req = make_request(childName="Emily", age=8, theme="princess", moral="patience"); repo = asyncio.run(run(req)); tr = Q.get_trace(req)
pc = tr.get("plan_check", {})
ok(pc.get("status") == "revised" and "child_causes_solution" in pc.get("failed_critical", []), f"failed check -> plan revised ({pc.get('status')})")
ok(tr["plan"]["beats"][1] == "revised beat 3" and tr["plan"]["beats"][0] == tr["plan_original"]["beats"][0] and tr.get("plan_original"), "revised plan used (Page 2 beat locked), original kept in log")
ok(any("revised beat 3" in p for k, p in CALLS if k == "continuation"), "pages follow the revised plan")
rev = [p for k, p in CALLS if k == "plan_revision"][0]
ok("The princess heals the wing on Page 7." in rev, "revision prompt lists the specific issue")
del os.environ["FAKE_PLAN_CHECK_FAIL"]
# --- switches off -> v4 ---
os.environ["BEDTIME_PROMPTS_V5"] = "0"; os.environ["BEDTIME_PLAN_CHECK_ENABLED"] = "0"; CALLS.clear()
req = make_request(childName="Emily", age=8, theme="princess", moral="patience"); repo = asyncio.run(run(req)); tr = Q.get_trace(req)
cont4 = [p for k, p in CALLS if k == "continuation"]
ok(tr.get("prompt_version") == "v4" and all("CHILDREN'S AUTHOR VOICE" in p for p in cont4), "BEDTIME_PROMPTS_V5=0 -> v4 prompts")
ok(tr.get("plan_check", {}).get("status") == "disabled" and not any(k == "plan_check" for k, _ in CALLS), "BEDTIME_PLAN_CHECK_ENABLED=0 -> no check call")
ok(not any("PAGE 6 COMPLETES THE SOLUTION" in p for k, p in CALLS if k == "plan"), "plan prompt back to v4 rules")
v4_words = [len(p.split()) for p in cont4]
del os.environ["BEDTIME_PROMPTS_V5"]; del os.environ["BEDTIME_PLAN_CHECK_ENABLED"]
# --- ages 0-2 never use v5 ---
CALLS.clear(); req = make_request(age=2, theme="animals"); repo = asyncio.run(run(req))
ok(Q.get_trace(req).get("prompt_version") == "v4", "age 2 keeps v4 prompts")
print(f"\nPage prompt size  v4: {min(v4_words)}-{max(v4_words)} words   v5: {min(v5_words)}-{max(v5_words)} words   "
      f"({round(100 * (1 - (sum(v5_words)/len(v5_words)) / (sum(v4_words)/len(v4_words))))}% shorter)")
