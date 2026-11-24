"""v8 (12 Nov 2026): outline-first openings, story design by age, one rule per idea, 5-check plan check,
peril by age with the wild-animal rule, a happy 'ask again tomorrow' ending. Run: python tests/test_v8.py <backend>"""
import asyncio, os, sys, random, re
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
from app.services import story_v8 as V8
os.environ["BEDTIME_V8D"] = "0"  # this file guards the signed-off v8c; test_v8d.py covers v8d
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)

# ---- the switch -------------------------------------------------------------
os.environ.pop("BEDTIME_V8", None)
ok(not V8.v8_enabled(), "v8 is off unless BEDTIME_V8=1 (deploying the zip changes nothing by itself)")
os.environ["BEDTIME_V8"] = "1"
ok(V8.v8_enabled(), "BEDTIME_V8=1 turns v8 on")
os.environ["BEDTIME_PROMPTS_V5"] = "0"
ok(not V8.v8_enabled(), "BEDTIME_PROMPTS_V5=0 still means the v4 prompts exactly (v8 stays off)")
del os.environ["BEDTIME_PROMPTS_V5"]


def run(**kw):
    CALLS.clear()
    svc = StoryService(FakeRepo()); req = make_request(**kw)
    d = asyncio.run(svc.generate_story_first_page(req, None))
    asyncio.run(svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"],
                                              current_pages=d["pages"], companion=None, expected_pages=7))
    by = {}
    for k, p in CALLS:
        by.setdefault(k, []).append(p)
    return req, by, svc


def words(p): return len(p.split())


def rules_only(prompt):
    """The prompt without the story's own data (the fake plan and pages), so only the rules are scanned."""
    out, skip = [], False
    for line in prompt.splitlines():
        if line.startswith(("LOCKED STORY PLAN", "PLAN THE STORY WAS WRITTEN FROM", "PLAN:", "PLAN TO REVISE", "THIS PAGE'S JOB FROM THE LOCKED PLAN")):
            skip = True
            continue
        if skip and (not line.strip() or line[:1].isupper() and line.rstrip().endswith(":") and not line.startswith("- ")):
            skip = False
        if skip or line.startswith(("{", "Page ", "PAGE 1 (FIXED)")) or line.startswith("  Emily") or line.startswith("  Tom"):
            continue
        out.append(line)
    return "\n".join(out)


# Phrases from the v7b contradictions (see project doc v8-rule-review) that must not appear in any v8 prompt.
BANNED = [
    "No real danger", "no fear, injury", "Do not reward danger", "giving up something they love",
    "egg grows cooler", "lantern dims", "lamp flickered", "egg grows colder", "evening or night",
    "It is answered on Page 7", "Page 7 must answer", "ALWAYS choose the lower reading level", "Sleepy Dragon",
    "PHASE 12 ENDING ENGINE", "FINAL ENDING CHECKLIST", "FORWARD PROGRESSION", "OXFORD-INSPIRED",
    "Sentence rhythm:", "calm, settled", "calm bedtime landing", "Age 0:", "twinkle,", "deliver the planned answer",
    "at some cost to themselves", "even though it costs them something",
]
WRITER_KINDS = ("page1", "plan", "continuation", "rewrite")

random.seed(7)
stories = {
    3: run(childName="Tom", age=3, theme="animals", moral=""),
    6: run(childName="Emily", age=6, theme="dragons", moral="patience"),
    9: run(childName="Noor", age=9, theme="space", moral=""),
    12: run(childName="Liam", age=12, theme="adventure", moral=""),
}
for age, (req, by, _svc) in stories.items():
    trace = Q.get_trace(req)
    ok(trace.get("prompt_version") == "v8", f"age {age}: the story is written with the v8 prompts")
    allp = [(k, rules_only(p)) for k, ps in by.items() for p in ps if k != "metadata"]
    hits = sorted({(k, b) for k, p in allp for b in BANNED if b in p})
    ok(not hits, f"age {age}: none of the old contradictions are in any prompt {hits[:4]}")
    for kind in WRITER_KINDS:
        ps = by.get(kind) or []
        if ps:
            ok(all("dangerous wild animals" in p for p in ps), f"age {age}: {kind} prompt has the wild-animal rule")
    ok(all("ADVENTURE AND SAFETY" in p for p in by["continuation"]), f"age {age}: adventure is asked for, by age")
    ok(any("ask for PillowTales again tomorrow" in p for p in by["continuation"]), f"age {age}: the ending aims for 'again tomorrow'")
    ok(any("does\n  not need to be bedtime or sleepy" in p or "not need to be bedtime or sleepy" in p for p in by["continuation"]),
       f"age {age}: the ending is at the story's own time of day, not forced to bedtime")
    ok(all(f"STORY DESIGN FOR {V8.BAND_LABEL[V8.band(age)]}" in p for p in by["continuation"] + by["plan"]),
       f"age {age}: the design for this age only ({V8.BAND_LABEL[V8.band(age)]})")
    other = [lbl for b, lbl in V8.BAND_LABEL.items() if b != V8.band(age)]
    ok(not any(f"STORY DESIGN FOR {lbl}" in p for p in by["continuation"] for lbl in other), f"age {age}: no other age's design")
    pc = by["plan_check"][0]
    ok(len(re.findall(r"^- \w+: ", pc.split("CHECKS:")[1], flags=re.M)) == 6 and "- few_things:" in pc and "FEW THINGS BUDGET FOR THIS AGE" in pc,
       f"age {age}: the plan check has 6 checks, including the few-things budget")
    ok("only when the plan clearly breaks a check" in pc and "whenever you are in doubt" not in pc, f"age {age}: the plan check fails real problems only")
    ok(trace.get("score", {}).get("rubric_version") == V8.RUBRIC_VERSION_V8 and "want_again" in trace["score"]["criteria"],
       f"age {age}: the scorer uses the v8 rubric, including 'would a child ask for it again'")
    ok(any("ends on a happy, satisfying final moment" in p for p in by["review"]), f"age {age}: the Page 7 review matches the v8 ending")
    ok(all(p.count("WORN-OUT TROPES") == 1 for p in by["continuation"]), f"age {age}: tropes listed once in the draft prompt")
    ok("Excitement: the story has real adventure" in by["editor"][0], f"age {age}: the editor flags a flat story too")

# ---- a little bit crazy (owner, 12 Nov) and the two fixes ------------------------
for age, (req, by, _svc) in stories.items():
    ok(all(p.count("A LITTLE BIT CRAZY") == 1 for k in ("page1", "plan", "continuation") for p in by.get(k, [])),
       f"age {age}: the 'little bit crazy' block reaches Page 1, the planner and the writer, once each")
    ok("child logic" in by["editor"][0] and "bedtime_landing" not in by["score"][0] and "read at bedtime" not in by["score"][0],
       f"age {age}: editor checks child logic; scorer has no bedtime bias")
    if age >= 9:
        ok(all("never words to put in" in p and "- Use upper-child vocabulary" not in p and "- Use richer middle-grade vocabulary" not in p
               for p in by["continuation"]), f"age {age}: the fancy-word list is a level guide, not words to insert")
    ok(all("Never add a word just to show vocabulary" in p and "Word budget:" not in p for p in by["continuation"]), f"age {age}: no forced impressive words, no rich-word quota")
    ok(all("A world a child would design" in p and "hot chocolate and" in p for p in by["continuation"]),
       f"age {age}: the writer is asked for a world a child would design (wish-come-true details)")
    ok(all("FEW THINGS, USED WELL" in p and V8._BUDGET[V8.band(age)] in p for p in by["continuation"] + by["plan"] + by["page1"]),
       f"age {age}: Page 1, planner and writer get this age's few-things budget")
    ok(all("never by adding more" in p for p in by["continuation"]), f"age {age}: new events come from what is already there")
    ok("Too much going on (check this first)" in by["editor"][0] and "A busy story" in by["score"][0], f"age {age}: editor and scorer check for clutter")
    ok("ONE wish-come-true detail that belongs to the big idea" in by["continuation"][0] and "use two or three" not in by["continuation"][0],
       f"age {age}: one wish-come-true detail, part of the big idea (not two or three extras)")
    ok(Q.get_trace(req).get("v8_build") == "bedtime_v8c=20261113", f"age {age}: every story records its version (v8c) in the trace")
req12 = stories[12][0]
ok("big idea" in Q.get_state(req12)["v8_outline"]["big_idea"] or Q.get_state(req12)["v8_outline"].get("big_idea"), "the outline carries the big idea")
ok("big idea: a dragon egg that hums" in stories[12][1]["page1_judge"][0] and "Prefer a little bit crazy over sensible" in stories[12][1]["page1_judge"][0],
   "the judge sees each big idea and prefers a little crazy over sensible")
ok("The big idea (keep it at the centre of every page): a dragon egg that hums" in stories[12][1]["plan"][0]
   and "The big idea (keep it at the centre of every page)" in stories[12][1]["continuation"][-1],
   "the big idea reaches the planner and every page")
ok("satisfying_ending" in Q.get_trace(req12)["score"]["criteria"] and "bedtime_landing" not in Q.get_trace(req12)["score"]["criteria"],
   "the scorer's ending criterion is 'satisfying_ending'")

ok("puns" in V8.imagination_block(12) and "backwards or wrong" in V8.imagination_block(3) and "slapstick" in V8.imagination_block(6),
   "humour matches each age: backwards/wrong at 3-4, slapstick at 5-7, wordplay at 8+")

# ---- moral only when there is one ------------------------------------------
cont_nomoral = stories[12][1]["continuation"][-1]
ok("MORAL:" not in cont_nomoral and "Requested moral" not in cont_nomoral and cont_nomoral.count("No moral was requested") == 1,
   "no moral: one line says so, and no moral rules are sent")
cont_moral = stories[6][1]["continuation"][-1]
ok("MORAL: patience" in cont_moral, "with a moral: one moral rule, shown through action")

# ---- outline first ------------------------------------------------------------
req, by, _ = stories[12]
state = Q.get_state(req)
judge = by.get("page1_judge", [""])[0]
ok(judge.count("Outline: big idea:") >= 2, "the judge sees each candidate's outline (whole stories, not just hooks)")
shapes = re.findall(r"\| shape: (\w+)", judge)
ok(len(set(shapes)) == len(shapes) >= 2, f"each candidate has its own plot shape {shapes}")
ok(isinstance(state.get("v8_outline"), dict) and state["v8_outline"].get("want"), "the chosen outline is kept for the planner")
plan = by["plan"][0]
ok("OUTLINE CHOSEN WITH PAGE 1" in plan and state["v8_outline"]["want"] in plan, "the planner expands the chosen outline")
ok(state.get("plot_shape_options") == [state["v8_outline"]["plot_shape"]], "the planner keeps the outline's plot shape")
ok("Answered on Page 6" in by["continuation"][-1], "the plan says questions are answered on Page 6")
p1 = by["page1"][0]
ok("OUTLINE (private" in p1 and "PLOT SHAPE FOR THIS STORY" in p1, "Page 1 asks for the outline in the same call")
ok(Q.get_trace(req).get("v8_outline") not in (None, "missing"), "the outline is recorded in the trace")

# single candidate path still gets the outline
os.environ["BEDTIME_PAGE1_CANDIDATES"] = "1"
r1, b1, _ = run(childName="Noor", age=10, theme="space", moral="")
ok(isinstance(Q.get_state(r1).get("v8_outline"), dict) and "OUTLINE CHOSEN WITH PAGE 1" in b1["plan"][0], "one candidate: the outline still reaches the planner")
del os.environ["BEDTIME_PAGE1_CANDIDATES"]

# no outline returned: the story still completes, the planner chooses a shape as before
os.environ["FAKE_NO_OUTLINE"] = "1"
r2, b2, svc2 = run(childName="Noor", age=10, theme="space", moral="")
ok("OUTLINE CHOSEN" not in b2["plan"][0] and "choose ONE of these" in b2["plan"][0], "no outline: the planner falls back to choosing a plot shape")
pages_done = (svc2.story_repo.updates[-1] if svc2.story_repo.updates else {})
ok(str(Q.get_trace(r2).get("draft_revise", {}).get("status", "")).startswith(("revised", "draft_accepted")) and "score" in Q.get_trace(r2),
   "no outline: the story is still finished and scored")
del os.environ["FAKE_NO_OUTLINE"]

# ---- smaller prompts ----------------------------------------------------------
os.environ["BEDTIME_V8"] = "0"
random.seed(7)
old = {age: run(childName="Liam", age=age, theme="adventure", moral="")[1] for age in (3, 6, 12)}
os.environ["BEDTIME_V8"] = "1"
for age in (3, 6, 12):
    o = max(words(p) for p in old[age]["continuation"]); n = max(words(p) for p in stories[age][1]["continuation"])
    ok(n < 0.6 * o, f"age {age}: the Pages 3-7 prompt is much shorter ({o} -> {n} words)")
    o1 = words(old[age]["page1"][0]); n1 = words(stories[age][1]["page1"][0])
    ok(n1 < 0.8 * o1, f"age {age}: the Page 1 prompt is shorter ({o1} -> {n1} words)")

# ---- age bands ----------------------------------------------------------------
young = V8.choose_shapes(3, [], 3)
ok(set(young) <= set(V8._YOUNG_SHAPES) and len(young) == 3, f"ages 3-4 get only patterned plot shapes {young}")
ok("two different places" in V8.shape_text("journey_home", 3) and "three different places" in V8.shape_text("journey_home", 9),
   "journey home has 2 stops at ages 3-4, 3 when older")
ok("twinkle" not in V8.clean_vocabulary("- Introduce words such as cosy, twinkle, sparkle, whisper, surprise, or adventure.\n- Sentence rhythm: short."),
   "the age-3 vocabulary no longer suggests words the voice rules ban")

# ---- structural plan check ------------------------------------------------------
plan_ok = {"beats": ["a fox arrives", "the bell rings", "c", "d", "Tom rings the bell", "f"], "solution_key": "the bell",
           "stake_sign": "the queue gets longer", "childs_decisive_act": "Tom rings the bell", "signature_delight": "x"}
issues = V8.structural_plan_issues(plan_ok, None, page_one="Tom found a bell. The queue gets longer.")
ok(not issues, f"v8 accepts a solution seen on Page 1 {issues}")
issues = V8.structural_plan_issues({**plan_ok, "beats": ["a", "b", "c", "d", "e", "f"]}, None, page_one="Tom woke up.")
ok(any(k == "solution_planted" for k, _ in issues), "v8 still flags a solution that is never planted")

# ---- ages 0-2 are untouched by v8 ------------------------------------------------
random.seed(3); _, a_on, _ = run(childName="Ana", age=2, theme="animals", moral="")
os.environ["BEDTIME_V8"] = "0"; random.seed(3); _, a_off, _ = run(childName="Ana", age=2, theme="animals", moral="")
os.environ["BEDTIME_V8"] = "1"
ok(sorted(p for ps in a_on.values() for p in ps) == sorted(p for ps in a_off.values() for p in ps), "ages 0-2: v8 changes nothing")
del os.environ["BEDTIME_V8"]
