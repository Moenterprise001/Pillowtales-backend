"""Bedtime Stories v8 (12 Nov 2026): fix causes, not symptoms.

Switch: BEDTIME_V8=1 turns v8 on (default off, so deploying this changes nothing
until the switch is set). v8 also needs v5 (BEDTIME_PROMPTS_V5 on); with
BEDTIME_PROMPTS_V5=0 the v4 prompts are used exactly, as before.

What v8 changes (see the project doc "v8-rule-review"):
1. Outline first. Each Page 1 candidate also returns a short outline of the whole
   story (want, solution object, big moment, ending, time of day, plot shape) in
   the same call. The judge picks the best whole story; the planner expands it.
2. Story design by age: 3-4, 5-7 and 8+ get different designs.
3. One instruction per idea. The rules are merged, contradictions are removed, and
   the checkers check only what the writer was told.
4. Owner decisions (12 Nov): adventure and excitement are wanted at every age
   (peril by age); the ending is a happy, satisfying finish that makes the child ask
   for another story, not a sleepy wind-down; Page 6 answers every question and
   Page 7 shows the result; the child never touches dangerous wild animals.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional

from app.services import story_quality as Q

V8_BUILD = "bedtime_v8c=20261113"
RUBRIC_VERSION_V8 = "v8c-20261113"


print(f"[BUILD] {V8_BUILD} BEDTIME_V8={os.getenv('BEDTIME_V8', '0')}", flush=True)


def v8_enabled() -> bool:
    """BEDTIME_V8=1 (and v5 not switched off)."""
    return (
        os.getenv("BEDTIME_V8", "0").strip().lower() in {"1", "true", "yes", "on"}
        and Q.prompts_v5_enabled()
    )


def band(age: Any) -> str:
    try:
        a = int(age or 0)
    except Exception:
        a = 0
    if a <= 4:
        return "young"
    if a <= 7:
        return "middle"
    return "older"


BAND_LABEL = {"young": "AGES 3-4", "middle": "AGES 5-7", "older": "AGES 8+"}


# ---------------------------------------------------------------------------
# Story design, peril and voice by age (one block each)
# ---------------------------------------------------------------------------

# v8c (13 Nov): owner found "too much going on" in all three v8 stories (Sky-Herder had xenon, telemetry, a
# gantry, a crane, a reflector dish, a satellite array and four failed tries). A budget of things per age, and
# new events made from those same things, not from new characters, places or gadgets.
_BUDGET = {
    "young": "one place (or a short trip with 2 stops), the child plus at most 2 others, ONE special object",
    "middle": "at most 2 places, the child plus at most 3 others, at most 2 special objects",
    "older": "at most 3 places, the child plus at most 4 others, at most 3 special objects",
}

_DESIGN = {
    "young": (
        "- One want, in one place or a short trip with at most 2 stops.\n"
        "- At most 2 other characters (a group, such as \"the ducks\", counts as one).\n"
        "- A pattern the child can join in: a refrain, a sound, or three tries that build.\n"
        "- Every set-up pays off; nothing is mentioned that does not matter later.\n"
        "- The answer is simple and clearly the child's own: they notice, try or use the thing seen earlier."
    ),
    "middle": (
        "- One want, at most one comic side character, and one real setback.\n"
        "- The answer is a small, clever idea of the child's that uses something seen earlier.\n"
        "- Every set-up pays off."
    ),
    "older": (
        "- One want and a small cast (at most 3 other named characters).\n"
        "- At most two tries fail before the answer; the child may have a small flaw that causes one of them.\n"
        "- A mystery is welcome, with its true explanation revealed on the page by Page 6.\n"
        "- The answer takes insight or courage: facing a fear, owning up to a mistake, or a clever risk.\n"
        "- Every set-up pays off."
    ),
}

_PERIL = {
    "young": "a little excitement: a chase, a wobble, a splash, a surprise or a funny near miss, and everyone is safe again within the page",
    "middle": "real adventure: a race against time, a near miss, a short scary moment that is quickly safe",
    "older": "real danger, chases, tension and near misses; nothing graphic, cruel or hopeless",
}

_RHYTHM = {
    "young": "Short, bouncy sentences (mostly 4-9 words) with sounds to say aloud; one longer sentence now and then.",
    "middle": "Mostly short, lively sentences (6-12 words); now and then a longer build-up before a very short punchline.",
    "older": "Clear sentences, mostly under 15 words, with variety; wit, inference and a real dilemma are welcome. Easy to read aloud; no adult literary or fantasy-novel prose.",
}

_HUMOUR = {
    "young": ("- Give the story a REFRAIN: one short line or sound that returns at least three times, changing a little each\n"
              "  time, so the child can join in. Funny means obvious silliness: wrong hats, funny sounds, animals acting oddly."),
    "middle": ("- A warm, playful narrator who may speak to the listener once or twice (\"Now, you might think that was the end\n"
               "  of it. It was not.\"). One side character has a comic trait or catchphrase that returns."),
    "older": ("- A witty narrator with dry humour; dialogue where characters tease, argue and joke, each sounding different;\n"
              "  one running gag that pays off at the end."),
}


_CHILD_LOGIC_BY_BAND = {
    "young": "sing the grumpy cloud a lullaby, tickle the stuck door, give the sad clock a hug",
    "middle": "make the shy volcano laugh so it stops rumbling, read the hatching books a bedtime story",
    "older": "obey the hat law by wearing a teapot, flatter the sulking map until it shows the way",
}

_CRAZY_BY_BAND = {
    "young": "silly-crazy: a cloud that is afraid of heights, a shoe that wants to be a boat, a moon with hiccups",
    "middle": "funny-crazy: a library where the books hatch, a dragon who collects lost socks, a rain that falls upwards on Tuesdays",
    "older": "clever-crazy: a law that only applies to people wearing hats, a map that sulks, a town where shadows go on strike",
}


_KID_HUMOUR_BY_BAND = {
    "young": "things done backwards or wrong (a sock worn as a hat, a cow that says quack), slapstick, funny sounds and made-up names",
    "middle": "slapstick, silly rules taken literally, upside-down situations, made-up names and sound words",
    "older": "wordplay, puns and double meanings, sayings taken literally (it really rains cats and dogs), absurd logic followed with a straight face",
}


def imagination_block(age: Any) -> str:
    """Owner, 12 Nov: 'add something a little crazy... think like a kid' (slides from the top of the palace to the
    garden, a magic carpet to travel faster, a train that serves only hot chocolate and cornflakes). Grounded in
    research: children's humour is incongruity (the world turned upside down; slapstick 2-7; puns from about 7, McGhee),
    young children think animistically and magically (Piaget), and children love wish fulfilment with children in
    charge (Dahl). One block, sent once to every writer and the planner."""
    b = band(age)
    return (
        "A LITTLE BIT CRAZY (this is what children remember and ask for again):\n"
        f"- Build the story on ONE wonderfully odd big idea and take it completely seriously ({_CRAZY_BY_BAND[b]}).\n"
        "- The big idea IS the fun: do not add other odd things beside it. Explore that one idea from new angles instead.\n"
        "- A world a child would design: ONE wish-come-true detail that belongs to the big idea and matters to the plot, like\n"
        "  slides instead of stairs from the top of the palace to the garden, a magic carpet for when you are late, or a train\n"
        "  that serves only hot chocolate and cornflakes. Invent a fresh one for this story; never copy these examples.\n"
        "- Think like a child, not an adult: objects, weather and animals have feelings and opinions; wishes and rules work\n"
        "  literally; small things matter hugely; nothing is simply black and white. The child is in charge of the adventure.\n"
        f"- Funny the way children of this age find funny: {_KID_HUMOUR_BY_BAND[b]}.\n"
        f"- The solution makes sense by child logic ({_CHILD_LOGIC_BY_BAND[b]}), not adult practicality,\n"
        "  and it uses the thing set up earlier, so it still feels fair.\n"
        "- Surprise the listener at least once with a turn no adult would predict, then make it make sense by the story's own rules.\n"
        "- Odd, not random: the world keeps its own silly rules, and every crazy detail connects to the want, the problem or the solution."
    )


def budget_block(age: Any) -> str:
    return (
        "FEW THINGS, USED WELL (the most important rule; children lose the thread when too much is going on):\n"
        f"- The whole story uses at most {_BUDGET[band(age)]}. Count them; a busy story is a worse story.\n"
        "- After Page 2 add no new character, place or important object. Every new event comes from what is already\n"
        "  there doing something new: the same characters, the same object, the same big idea, taken further.\n"
        "- No gadgets, machines or technical words (no telemetry, xenon, gantries, sluices, calibration). Plain, concrete,\n"
        "  touchable things a child already knows: a spoon, a kite, a bucket, a ribbon.\n"
        "- One thing happens at a time: one action per sentence, one problem at a time, so the listener can picture it."
    )


def design_block(age: Any) -> str:
    b = band(age)
    return (
        f"STORY DESIGN FOR {BAND_LABEL[b]}:\n" + _DESIGN[b] + "\n"
        f"- Excitement level: {_PERIL[b]}.\n\n" + budget_block(age) + "\n\n" + imagination_block(age)
    )


def peril_block(age: Any) -> str:
    return (
        "ADVENTURE AND SAFETY:\n"
        "- Excitement and adventure are wanted: they are why children ask for PillowTales again. Use the excitement level\n"
        "  above; never graphic, cruel or hopeless, and nobody is left frightened at the end.\n"
        "- The child keeps a safe distance from dangerous wild animals (snarling, cornered or hurt ones): they help by\n"
        "  calling, luring, using a tool or fetching help, never by touching. Friendly, magical or tame creatures can be touched."
    )


def clean_vocabulary(text: str) -> str:
    """The age vocabulary standard without the parts v8 says elsewhere or that clash with v8:
    its own sentence-rhythm line (v8 has one rhythm line per age), the generic Oxford rule (said by the
    age block itself), and the storybook words that the voice rules ban (twinkle, sparkle, whisper)."""
    out, skip = [], False
    for line in str(text or "").splitlines():
        if line.strip().startswith("OXFORD-INSPIRED VOCABULARY RULE"):
            skip = True
            continue
        if skip:
            if line.strip().startswith("-"):
                continue
            skip = False
        if line.strip().startswith(("- Sentence rhythm:", "- Word budget:")):
            continue  # v8 has one rhythm line per age; a percentage quota of rich words made the AI force them in
        out.append(line)
    text = "\n".join(out).strip()
    text = re.sub(r"\b(?:twinkle|sparkle|whisper),\s*", "", text)
    # Word lists are a guide to the level, not words to use (they were being forced in: "remarkable", "contradiction").
    text = re.sub(r"^- Use ([^:\n]*vocabulary[^:\n]*): ", r"- Level guide (\1; words this age understands, never words to put in): ", text, flags=re.M)
    return text


def language_block(age: Any, vocabulary_block: str = "") -> str:
    """Reading level for this age only: the vocabulary standard plus one rhythm line."""
    try:
        a = int(age or 0)
    except Exception:
        a = 0
    vocab = clean_vocabulary(vocabulary_block)
    return (
        f"LANGUAGE LEVEL (reading age {a}; never state the age in the story):\n"
        f"- {_RHYTHM[band(age)]}\n"
        "- Clear, vivid, everyday words beat impressive ones. Never add a word just to show vocabulary; a harder word is\n"
        "  welcome only when it is the most fun or exact word and the sentence makes its meaning clear.\n"
        + (vocab + "\n" if vocab else "")
    )


def voice_block(age: Any) -> str:
    return (
        "VOICE (how it sounds read aloud):\n" + _HUMOUR[band(age)] + "\n"
        "- Write like an experienced children's author: specific, original, never generated-sounding, never imitating\n"
        "  a named author. One surprising, story-only detail beats three adjectives; at most two adjectives on a noun.\n"
        "- A real laugh from character on most pages: a habit, a misunderstanding, a comic contrast.\n"
        "- Dialogue sounds spoken, shows who someone is and changes something; never use it to explain the plot or the rules.\n"
        "- Show feelings through what characters do. No stock phrases (tummy flip-flops, sparkling or widened eyes, gasps,\n"
        "  sinking hearts, \"she realised that\", \"With a determined breath\").\n"
        "- Avoid overused words: shimmering, shimmer, glimmer, glowing, sparkling, sparkly, twinkling, whispering (as\n"
        "  description), iridescent, velvety, moonlit. Never describe the child as little, small, tiny or young.\n"
        "- Write for the ear: vary sentence length and shape, and silently read each page aloud before returning it."
    )


TROPE_BANS_V8 = """WORN-OUT TROPES (never use; they are in too many recent stories):
- Stakes shown as a light, glow or colour fading or dimming, or something going cold or grey. Use a fresher visible
  sign instead: a queue getting longer, the tide creeping up the steps, a clock about to strike, guests starting to leave.
- A creature that sneezes, hiccups or burps puffs, bubbles, sparks or steam as the running joke.
- A note, scroll, inscription or book page that tells the child the answer or the lesson.
- The child giving up or handing over a treasured, favourite or prized possession as the climax, for any moral.
  Find a cleverer act: an idea, a skill, a trick, teamwork, a turn, half of something ordinary.
- The usual shape: something fades, the first try fails, the child gives up a treasure, it glows again."""


def story_rules_block(request: Any, moral_requested: bool, moral: Any) -> str:
    theme = getattr(request, "customTheme", None) or getattr(request, "theme", "")
    moral_line = (
        f"- MORAL: {moral}. It is private. Never use the moral word or explain the lesson; show it only through one real\n"
        "  choice and its consequence by Page 6. With all commentary removed, the actions alone must make it clear."
        if moral_requested else
        "- No moral was requested: do not invent a lesson."
    )
    return f"""STORY RULES (each stated once):
- THEME: "{theme}" is a real promise. Its central subject is literally in the story, affects the plot and meets the child early.
- THE CHILD SOLVES IT: the child's own idea or action causes the solution on Page 6. Helpers may help or get it funnily
  wrong; no adult, helper, magic, gift or luck solves it for the child.
- PLANT AND PAY OFF: the thing the child uses to solve it is seen by Page 2. Nothing the child needs at the climax
  appears for the first time there. Every set-up that is made to look important pays off.
- MOMENTUM: every page has at least one NEW event (an action, discovery, surprise, choice or consequence), made from the
  characters and things already in the story, never by adding more. At most two
  sentences of pure scenery per page. Never restate what the listener already knows. The first sentence of each page
  already moves the story; Page 2 opens with something happening.
- NAMES: introduce the child by name, then use natural pronouns; bring the name back when it helps clarity. Nobody knows
  a new character's name until it is given on the page.
- FAMILY: only family the parent added may appear. Never invent the child's mum, dad, grandparents, brothers, sisters,
  aunts, uncles or cousins, and never say one is away, missing or has died. For warmth, use a story character, the
  child's own memory or habit, or a favourite toy.
{moral_line}
- Plain prose only: no Markdown, asterisks, headings or bullets in the story text."""


# ---------------------------------------------------------------------------
# Plot shapes and the page map
# ---------------------------------------------------------------------------

_YOUNG_SHAPES = ["escalating_tries", "cumulative_tale", "journey_home", "too_big_too_small"]
_YOUNG_SHAPE_TEXT = {
    "journey_home": "JOURNEY HOME: the child helps someone small get home through two different places, each with its own little problem.",
}


def shape_text(key: str, age: Any) -> str:
    if band(age) == "young" and key in _YOUNG_SHAPE_TEXT:
        return _YOUNG_SHAPE_TEXT[key]
    return Q._PLOT_SHAPE_TEXT.get(key, "")


def choose_shapes(age: Any, recent: Iterable[str] = (), count: int = 3) -> List[str]:
    """Shapes for this age the child has not heard recently. Ages 3-4 get only patterned shapes."""
    shapes = Q.choose_plot_shapes(int(age or 0), recent, count=12)
    if band(age) == "young":
        shapes = [s for s in shapes if s in _YOUNG_SHAPES]
    out = shapes[:count]
    while out and len(out) < count:
        out.append(out[len(out) % len(shapes)])
    return out


def shape_line(key: str, age: Any) -> str:
    return f"PLOT SHAPE FOR THIS STORY: {key} - {shape_text(key, age)}"


DEFAULT_PAGE_MAP = {
    2: "Something happens straight away that starts the adventure.",
    3: "The first try, which goes wrong in a new or funny way.",
    4: "A new event: a bigger try, a surprise or a new problem.",
    5: "The big moment: the most exciting page, where it comes closest to going wrong.",
    6: "The child solves it with their own idea, using what was seen earlier; every question is answered on this page.",
    7: "The happy result of what the child did, a callback, and a satisfying final moment.",
}


def page_map_block(plot_shape: Optional[str], age: Any, pages: Iterable[int]) -> str:
    pages = [p for p in pages if 2 <= p <= 7]
    if not pages:
        return ""
    desc = shape_text(plot_shape or "", age)
    lines = ["PAGE MAP:"]
    if desc:
        lines.append(f"- The plot shape sets Pages 2-5: {desc}")
        lines.append("  Spread it across Pages 2-5 so each page is a new step, with the big moment (the most exciting point) on Page 4 or 5.")
        for p in pages:
            if p in (6, 7):
                lines.append(f"- Page {p}: {DEFAULT_PAGE_MAP[p]}")
    else:
        for p in pages:
            lines.append(f"- Page {p}: {DEFAULT_PAGE_MAP[p]}")
    return "\n".join(lines)


def ending_block(final_page: int = 7) -> str:
    return f"""ENDING:
- Page {final_page - 1}: the child's own act solves it, and every question raised is answered on the page, including any
  hidden explanation.
- Page {final_page}: the happy result of what the child did, a callback to something from earlier, and a satisfying final
  moment: a last laugh, a warm image, or the refrain one last time. It happens at the story's own time of day; it does
  not need to be bedtime or sleepy.
- Page {final_page} adds nothing new: no new character, task, place, clue or problem, and no explanation that belongs on Page {final_page - 1}.
- The story is complete: no cliffhanger and no "another adventure was waiting". It should be an ending a child loves
  enough to ask for PillowTales again tomorrow.
- Never end on a summary of feelings ("felt very proud", "was ready for anything") and never explain the lesson.
- The final sentence is a concrete image, action, sound or line of dialogue from this story. Then write exactly: The End."""


NOT_FINAL_BLOCK = """NOT THE ENDING YET:
- Do not resolve the whole story on these pages.
- End each page with a reason to keep listening: a surprise, a choice, a funny complication or a clear next step."""


# ---------------------------------------------------------------------------
# Page 1 (with outline) and the judge
# ---------------------------------------------------------------------------

OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "big_idea": {"type": "string"},
        "want": {"type": "string"},
        "solution_object": {"type": "string"},
        "solution_seen_on_page": {"type": "integer"},
        "big_moment": {"type": "string"},
        "ending": {"type": "string"},
        "time_of_day": {"type": "string"},
        "plot_shape": {"type": "string"},
    },
    "required": ["big_idea", "want", "solution_object", "solution_seen_on_page", "big_moment", "ending", "time_of_day", "plot_shape"],
}


def page1_schema(base_schema: dict) -> dict:
    schema = {**base_schema, "properties": {**base_schema.get("properties", {}), "outline": OUTLINE_SCHEMA}}
    schema["required"] = [*base_schema.get("required", []), "outline"]
    return schema


def clean_outline(outline: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(outline, dict):
        return None
    out = {k: re.sub(r"\s+", " ", str(outline.get(k) or "")).strip()[:300]
           for k in ("big_idea", "want", "solution_object", "big_moment", "ending", "time_of_day", "plot_shape")}
    try:
        seen = int(outline.get("solution_seen_on_page") or 0)
    except Exception:
        seen = 0
    out["solution_seen_on_page"] = seen if seen in (1, 2) else 2
    if not (out["want"] and out["solution_object"] and out["ending"]):
        return None
    return out


OUTLINE_INSTRUCTIONS = """OUTLINE (private; return it with Page 1 so the whole story is designed before it is written):
- big_idea: the one wonderfully odd idea at the heart of the story, the thing a child would tell a friend about,
  in one sentence. It must already show on Page 1.
- want: what the child wants or must do, in one sentence.
- solution_object: the one object, habit, skill or clue the child will use to solve it. It is seen on Page 1, or
  arrives on Page 2; set solution_seen_on_page to 1 or 2. If it is 1, show it naturally on Page 1 without pointing at it.
- big_moment: the most exciting moment of the story, where it comes closest to going wrong.
- ending: how the child solves it and the happy result, in one sentence. It must fit the plot shape and the age design.
- time_of_day: when the story happens (morning, after school, a stormy afternoon, night...). The ending happens then too.
- plot_shape: the key of the plot shape given above."""


def hook_block(hook: str) -> str:
    return (
        "HOOK FOR THIS STORY (use this kind of hook, made specific to this child, theme and world):\n"
        f"- {Q.HOOK_TYPES.get(hook, hook)}.\n"
        "- It gives the child something to WANT or DO, so that later the child, not a helper, can solve it.\n"
    )


def inject_hook(prompt: str, hook: str) -> str:
    marker = "PAGE 1 JOB:"
    block = hook_block(hook)
    return prompt.replace(marker, block + marker, 1) if marker in prompt else prompt + "\n" + block


def build_page1_judge_prompt(request: Any, candidates: List[Dict[str, Any]], age: int, age_standard: str = "") -> str:
    theme = getattr(request, "customTheme", None) or getattr(request, "theme", "")

    def _outline(c: Dict[str, Any]) -> str:
        o = c.get("_outline") or {}
        if not o:
            return "Outline: (none given)"
        return ("Outline: big idea: " + (o.get("big_idea") or "(none)") + " | wants " + o.get("want", "") + " | solves it with " + o.get("solution_object", "")
                + f" (seen on Page {o.get('solution_seen_on_page')})" + " | big moment: " + o.get("big_moment", "")
                + " | ending: " + o.get("ending", "") + " | time: " + o.get("time_of_day", "")
                + " | shape: " + str(c.get("_shape") or o.get("plot_shape") or ""))

    listing = "\n\n".join(
        f"CANDIDATE {i + 1}\nTitle: {c.get('title')}"
        + (f"  (TITLE PROBLEM: {'; '.join(c.get('_title_flags') or [])})" if c.get("_title_flags") else "")
        + f"\nPage 1: {(c.get('pages') or [''])[0]}\n" + _outline(c)
        for i, c in enumerate(candidates)
    )
    return f"""You are choosing the opening page for a seven-page PillowTales story (listener age {age}, theme "{theme}").
Each candidate is a whole story: a Page 1 and an outline of the rest. Pick the one a child is most likely to love and
ask for again tomorrow night. Return only JSON matching the schema:
- best: the winning candidate number; reason: why, in one sentence.
- story_promise: what the winning Page 1 promises the listener, in one sentence.
- childs_want: what the child wants or must do, in one sentence.
- stakes_seed: what could go wrong if the child fails, in one sentence.
- spare_ideas: 1-3 short, specific ideas from the OTHER candidates worth reusing (a funny detail, image or habit), compatible with the winner.
- fresh_title: ONLY if the winner's title has a TITLE PROBLEM, a new title for it (up to 7 words, includes the child's name,
  specific to this story, none of the overused words, not the moral). Otherwise "".

JUDGE THE WHOLE STORY:
- The big idea (most important): one wonderfully odd idea a child would tell a friend about, working by child logic,
  in a world a child would design (wish-come-true details such as slides instead of stairs).
  Prefer a little bit crazy over sensible; mark down ordinary set-ups (a runaway pet, a contest, a lost item, a rule
  that needs explaining).
- Few things, used well: prefer the candidate whose outline needs the fewest characters, places and objects. A busy
  outline (many gadgets, helpers or places) loses to a simpler one, however clever.
- Fun and excitement: a hook a child wants to follow, a big moment with real adventure for this age, real laughs.
- The child has something to want or do and solves it with their own idea, using a thing seen on Page 1 or 2.
- It hangs together: the ending fits Page 1, its time of day and its plot shape; every set-up pays off.
- Fresh: an original premise, not a generic "found something, return it" set-up; the theme is literally present.
- It fits the age: {design_block(age).splitlines()[0].replace('STORY DESIGN FOR ', '').rstrip(':').lower()} design and language.
- Avoid candidates with overused names (Barnaby, Pip, Sparkle...) or overused title words (Whispering, Secret, Lost...).
{("AGE STANDARD:" + chr(10) + age_standard.strip()) if age_standard else ""}

{listing}
"""


def opening_notes_block(notes: Optional[Dict[str, Any]]) -> str:
    """Judge notes for the planner (v8 leaves out the old hook climax shapes, which asked for a sacrifice)."""
    if not notes:
        return ""
    lines = ["OPENING NOTES FROM THE EDITOR (use them):"]
    for key, label in (("story_promise", "What this opening promises"), ("childs_want", "What the child wants"),
                       ("stakes_seed", "What could go wrong")):
        value = str((notes or {}).get(key) or "").strip()
        if value:
            lines.append(f"- {label}: {value}")
    spare = [str(i).strip() for i in ((notes or {}).get("spare_ideas") or []) if str(i).strip()][:3]
    if spare:
        lines.append("- Strong ideas from the openings not chosen; use one or two only where they genuinely fit (never contradict Page 1): " + "; ".join(spare))
    return "\n".join(lines) + "\n" if len(lines) > 1 else ""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

_ACT_BY_BAND = {
    "young": "Keep it simple and clearly the child's own.",
    "middle": "It is a small, clever idea of the child's.",
    "older": "It takes insight or courage (facing a fear, owning up to a mistake, a clever risk), never giving up a treasured thing.",
}
_DELIGHT_BY_BAND = {
    "young": "the refrain or sound the child can join in",
    "middle": "a comic trait or catchphrase",
    "older": "a running gag",
}


def outline_block(outline: Optional[Dict[str, Any]], age: Any) -> str:
    if not outline:
        return ""
    shape = outline.get("plot_shape") or ""
    return (
        "OUTLINE CHOSEN WITH PAGE 1 (expand it; change it only to fix a clear clash with Page 1):\n"
        + (f"- The big idea (keep it at the centre of every page): {outline.get('big_idea')}\n" if outline.get("big_idea") else "")
        + f"- The child wants: {outline.get('want')}\n"
        f"- Solved with: {outline.get('solution_object')} (seen on Page {outline.get('solution_seen_on_page')})\n"
        f"- Big moment: {outline.get('big_moment')}\n"
        f"- Ending: {outline.get('ending')}\n"
        f"- Time of day: {outline.get('time_of_day')}\n"
        + (f"- Plot shape: {shape} - {shape_text(shape, age)}\n" if shape_text(shape, age) else "")
    )


def build_plan_prompt(request: Any, title: str, page_one: str, moral_requested: bool, avoid_block: str, age: int) -> str:
    theme = getattr(request, "customTheme", None) or getattr(request, "theme", "")
    moral = getattr(request, "moral", "")
    state = Q.get_state(request)
    outline = state.get("v8_outline")
    b = band(age)
    options = state.get("plot_shape_options") or []
    if outline and outline.get("plot_shape") in Q._PLOT_SHAPE_TEXT:
        shape_part = f"- plot_shape: \"{outline['plot_shape']}\" (from the outline).\n"
    elif options:
        shape_part = ("- plot_shape: choose ONE of these that fits Page 1 best and build the beats on it:\n"
                      + "\n".join(f"  - {k}: {shape_text(k, age)}" for k in options) + "\n")
    else:
        shape_part = ""
    moral_line = (
        f"- Private moral: {moral}. The child's decisive act shows it through action and consequence; nobody names or explains it."
        if moral_requested else "- No moral was requested."
    )
    none_ok = ("hidden_explanation, child_flaw, comic_trait and signature_delight may be \"none\"; leave them out "
               "rather than add things to fill them, and never add a character, place or object just for a field")
    return f"""You are the story editor for a seven-page PillowTales story. Page 1 is written and fixed. Plan Pages 2-7 so the
finished story is one a child will ask for again tomorrow. Return only JSON matching the schema, written in English.

STORY FACTS:
- Title: {title}
- Child (protagonist): {getattr(request, 'childName', '')}
- Theme: {theme}
- Reading age: {age}
{moral_line}

PAGE 1 (FIXED):
{page_one}

{outline_block(outline, age)}
{design_block(age)}

{peril_block(age)}

PLAN FIELDS (each idea once; {none_ok}):
- opening_question: the want, question or problem Page 1 raises, in one sentence.
- answer_to_opening_question: how it is answered. It is answered on Page 6.
- hidden_explanation: the truth behind any mystery, revealed aloud in a Page 2-6 beat.
- stakes: what goes wrong if the child fails. stake_for_child: who the child would let down or what they would miss,
  in words a child feels (a friend left waiting, a promise broken, a party that will not happen). stake_sign: a visible
  sign that the loss is coming closer, seen by Page 3 (not fading, dimming or going cold).
- beats: exactly six strings, for Pages 2-7, 1-2 sentences each. Each beat starts with a NEW event that changes the
  situation, made from the characters and things already in the story; no beat only describes, gets ready or repeats
  the problem, and no beat after Page 2 adds a new character, place or important object. Follow the page map below.
- setback: the big moment, where it comes closest to going wrong.
- solution_key: the one object, habit, skill or clue the child uses on Page 6 (the outline's solution);
  solution_planted_page: 1 or 2, where the listener first sees it.
- childs_decisive_act: what the child does on Page 6 with the solution_key; the child's own idea. {_ACT_BY_BAND[b]}
- callback_detail: a Page 1 detail that returns on Page 7.
- final_image: the last image: happy and satisfying, at the story's own time of day.
- threads_to_close: every open question; each is answered in a specific Page 2-6 beat.
- signature_delight: {_DELIGHT_BY_BAND[b]}, which returns on Page 7.
- child_flaw: a small, funny, believable flaw that makes the first try fail.
- comic_trait: one side character's strong comic trait.
{shape_part}
{page_map_block((outline or {}).get("plot_shape") if outline else None, age, range(2, 8))}

PLAN RULES:
- Keep Page 1 facts exactly. No new important character after Page 4; keep within the cast in the design above.
- THEME: "{theme}" is literally present and matters (a princess theme needs a real princess; dragons need a real dragon).
- TITLE: honour every word of the title (if it says "grumpy", someone is truly grumpy and we learn why).
- Everything the child uses on Page 6 is already seen in Page 1 or an earlier beat.
- NAMES: no new character may use a name from the avoid list below; a plan that uses one is rejected.
- FAMILY: only family the parent added; never invent the child's mum, dad, grandparents, siblings, aunts, uncles or
  cousins, in any field. Use a story character, the child's own memory or habit, or a favourite toy instead.

{TROPE_BANS_V8}

{Q.pronoun_block(request)}{(Q.comfort_theme_block(request) + chr(10) + chr(10)) if Q.comfort_theme_block(request) else ""}SELF-CHECK BEFORE ANSWERING (fix the plan if any answer is "no"):
1. Does the child's own act on Page 6 solve it?
2. Is the solution_key seen on Page 1 or in the Page 2 beat?
2b. Does the story stay within the FEW THINGS budget, with nothing new added after Page 2?
3. Is every question answered in a Page 2-6 beat?
4. Is everything consistent with Page 1?
5. Is Page 7 only the happy result, with nothing new?{(chr(10) + "6. Does the decisive act show the moral without anyone stating it?") if moral_requested else ""}
{avoid_block}
"""


def _real(value: Any) -> bool:
    text = str(value or "").strip().lower().strip(".")
    return bool(text) and text not in {"none", "n/a", "na", "no", "nothing", "-", "not needed"}


def plan_spine_block(plan: Optional[dict]) -> str:
    if not Q.plan_is_valid(plan):
        return ""
    beats = "\n".join(f"  - Page {i + 2}: {str(beat).strip()}" for i, beat in enumerate(plan["beats"][:6]))
    threads = "; ".join(str(t).strip() for t in (plan.get("threads_to_close") or []) if _real(t)) or "none listed"
    lines = [
        "LOCKED STORY PLAN (private notes; follow them and never quote them):",
        f"- The want: {plan.get('opening_question')}",
        f"- Answered on Page 6: {plan.get('answer_to_opening_question')}",
    ]
    if _real(plan.get("hidden_explanation")):
        lines.append(f"- Hidden explanation (reveal it on the page by Page 6): {plan.get('hidden_explanation')}")
    stake = str(plan.get("stakes") or "").strip()
    if _real(plan.get("stake_sign")):
        stake += f" Visible sign, seen by Page 3: {plan.get('stake_sign')}"
    lines.append(f"- At stake: {stake}")
    lines.append(f"- Big moment: {plan.get('setback')}")
    if _real(plan.get("solution_key")):
        lines.append(f"- The solution: {plan.get('solution_key')} (first seen on Page {plan.get('solution_planted_page') or 2})")
    lines.append(f"- The child's own act on Page 6: {plan.get('childs_decisive_act')}")
    for key, label in (("signature_delight", "Funny thing that returns"), ("comic_trait", "Comic trait"),
                       ("child_flaw", "The child's flaw")):
        if _real(plan.get(key)):
            lines.append(f"- {label}: {plan.get(key)}")
    lines.append(f"- Callback on Page 7: {plan.get('callback_detail')}; final image: {plan.get('final_image')}")
    lines.append(f"- Questions to answer by Page 6, in words the listener hears: {threads}")
    lines.append("- Page beats:")
    return "\n" + "\n".join(lines) + "\n" + beats + "\n"


PLAN_PRECEDENCE_V8 = (
    "- If a beat clashes with the story rules (Page 6: the child's own act solves it and every question is answered; "
    "Page 7: only the happy result), follow the rules and adjust the beat."
)


def structural_plan_issues(plan: Any, request_for_family: Any = None, page_one: str = "") -> List[tuple]:
    """v8 deterministic checks: the solution is seen on Page 1 or in the Page 2-3 beat and used on Page 6; a stake
    sign is seen by Page 3; no invented family; no worn-out tropes; the plot shape is one that was offered."""
    if not isinstance(plan, dict):
        return []
    beats = list(plan.get("beats") or [])
    if len(beats) < 6:
        return []
    issues: List[tuple] = []
    key = Q._key_words(plan.get("solution_key"))
    if not key:
        issues.append(("solution_planted", "solution_key is missing: name the exact object, habit, clue or skill the child uses on Page 6."))
    else:
        if not (Q._mentions(page_one, key) or Q._mentions(beats[0], key) or Q._mentions(beats[1], key)):
            issues.append(("solution_planted", f"The solution ({plan.get('solution_key')}) is not seen on Page 1 or in the Page 2 or 3 beat. "
                                               "Put it into the Page 2 beat in a natural, memorable way."))
        if not Q._mentions(beats[4], key):
            issues.append(("solution_planted", f"The Page 6 beat does not use the solution ({plan.get('solution_key')})."))
    sign = Q._key_words(plan.get("stake_sign"))
    if sign and not (Q._mentions(page_one, sign) or Q._mentions(beats[0], sign) or Q._mentions(beats[1], sign)):
        issues.append(("stake_felt", f"The visible stake sign ({plan.get('stake_sign')}) is not seen by Page 3; put it in the Page 2 or 3 beat."))
    if request_for_family is not None:
        fam = Q.invented_family_mentions(Q.plan_text(plan), request_for_family, already_told=page_one)
        if fam:
            issues.append(("invented_family", "The plan gives the child family the parent did not add (" + ", ".join(fam[:4])
                           + "). Replace them with a story character (a kindly keeper, a neighbour, a creature friend) or the child's own memory."))
    issues.extend(Q.trope_issues(plan))
    offered = (Q.get_state(request_for_family).get("plot_shape_options") or []) if request_for_family is not None else []
    if offered and str(plan.get("plot_shape") or "").strip() not in offered:
        issues.append(("plot_shape", f"plot_shape must be one of {', '.join(offered)}, and the beats must follow that shape."))
    return issues


# ---------------------------------------------------------------------------
# Plan check: 5 real failures only
# ---------------------------------------------------------------------------

PLAN_CHECK_KEYS_V8 = [
    ("child_causes_solution", True, "The Page 6 beat shows the child's own act causing the outcome; no adult, helper, creature, magic, luck or coincidence fixes, rescues or finishes it."),
    ("solution_planted_early", True, "The solution_key is seen on Page 1 or in the Page 2 (at the latest Page 3) beat, and the Page 6 beat uses it."),
    ("answered_by_page6", True, "The hidden explanation (if any) and every item in threads_to_close is answered inside a specific Page 2-6 beat (check the beat text itself)."),
    ("no_contradictions", True, "Nothing contradicts Page 1 or another part of the plan, and no name or fact is known before it is learned."),
    ("few_things", True, "The plan stays within the FEW THINGS budget for this age (count characters, places and important objects across Page 1 and all beats) and adds no new character, place or important object after the Page 2 beat."),
    ("page7_result_only", True, "The Page 7 beat shows only the happy result, the callback and the final moment: nothing new arrives, is rescued, revealed, explained or decided."),
]

PLAN_CHECK_SCHEMA_V8 = {
    "type": "object",
    "properties": {
        **{key: {"type": "boolean"} for key, _, _ in PLAN_CHECK_KEYS_V8},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": [*[key for key, _, _ in PLAN_CHECK_KEYS_V8], "issues"],
}


def build_plan_check_prompt(request: Any, page_one: str, plan: dict, age: int) -> str:
    theme = getattr(request, "customTheme", None) or getattr(request, "theme", "")
    checks = "\n".join(f"- {key}: {desc}" for key, _, desc in PLAN_CHECK_KEYS_V8)
    return f"""You are a strict children's book editor checking a story plan before any page is written.
Return only JSON matching the schema. Answer false only when the plan clearly breaks a check; do not fail a plan for
taste or style. For every false answer, add one short, specific issue that says exactly what to change.

STORY FACTS: theme "{theme}"; listener age {age}.
FEW THINGS BUDGET FOR THIS AGE: at most {_BUDGET[band(age)]}.

PAGE 1 (FIXED):
{page_one}

PLAN:
{Q.plan_text(plan)}

CHECKS:
{checks}
"""


# ---------------------------------------------------------------------------
# Editor and scorer: check what the writer was told
# ---------------------------------------------------------------------------

EDITOR_CHECKLIST_V8 = """- Momentum (most important): flag ANY page where nothing new happens, that mostly describes scenery or getting ready,
  or that repeats the previous page in new words. Its fix must name the new event that should happen there.
- The child's own idea or action solves it on Page 6, using something seen by Page 2; nothing needed at the climax
  appears for the first time there. No helper, adult, magic or luck solves it.
- Every question and thread is answered by Page 6. Page 7 is the happy result, a callback and a satisfying final
  moment, with nothing new and no cliffhanger.
- Excitement: the story has real adventure for this age and a big moment where it nearly goes wrong. Flag a flat,
  safe-all-the-way story as much as a frightening, graphic or cruel one.
- The child never touches a dangerous wild animal (snarling, cornered or hurt); friendly, magical or tame creatures are fine.
- Family: no invented family for the child (mum, dad, grandparents, siblings, aunts, uncles, cousins) unless the parent added them.
- Continuity: no contradictions; nobody knows a name or fact before learning it; objects and names stay consistent.
- Age fit: words and sentences match the age standard.
- Too much going on (check this first): flag any page that adds a new character, place or object after Page 2,
  uses gadgets or technical words, or crams more than one thing happening into a sentence. Its fix must remove things
  (cut the extra helper, gadget or place and give its job to something already in the story), never add.
- A little bit crazy: flag a story that has become sensible or ordinary, a big idea that fades after Page 1, or a
  solution that works by adult practicality instead of the story's own child logic.
- Voice and laughs: flag pages with no funny or delightful moment, forced "impressive" words, dialogue that only explains the plot, stacked
  adjectives, stock phrases, or a worn-out trope (fading/dimming/going-cold stakes, sneeze/hiccup puffs, a note giving
  the answer, a treasured item given up as the climax).
- The moral (if any) is shown through action, never stated. The theme and title promise are delivered."""


SCORE_CRITERIA_V8 = [
    ("opening_promise_resolved", "Every question/wish/mystery raised on Page 1 is clearly answered by the end."),
    ("child_drives_climax", "The child's own idea or action causes the resolution; not luck, an accident or a helper."),
    ("stakes_and_tension", "Real excitement and adventure for this age: something concrete is at stake, a big moment where it nearly goes wrong, and a reason to worry. A flat story where nothing is really at risk scores 6 at most. Lower it only for content that is graphic, cruel or leaves the child frightened at the end."),
    ("coherence", "Easy to follow: few characters, places and objects, each used well, and no confusing contradictions or loose threads. A busy story (many helpers, gadgets, places or technical words) scores 7 at most."),
    ("pacing", "Something NEW happens on every page and the listener always wants the next page. A page that mostly describes scenery, shows getting ready, or repeats the problem from the page before is a serious fault: any such page caps this score at 6, two or more at 5. The ending is not crammed."),
    ("moral_shown_not_told", "The moral (if any) is shown through choices and consequences, never stated or lectured."),
    ("age_fit_language", "Vocabulary and sentence craft match the listener's age, neither babyish nor too adult. Hard words put in to sound impressive lower this score."),
    ("imagery_and_humour", "A distinct read-aloud voice (refrain, narrator personality or wit, by age), specific surprising images rather than stacked adjectives, and real laughs from character on several pages. Generic description with one mild funny moment scores 7 at most; worn-out tropes (fading/dimming stakes, sneeze/hiccup puffs, shimmering) lower it."),
    ("satisfying_ending", "The ending: the happy result of the child's act, an earned callback and a satisfying final moment (a last laugh, a warm image). Complete, with no cliffhanger. It does not need to be sleepy."),
    ("originality", "One wonderfully odd big idea, taken seriously and working by child logic, that a child would tell a friend about. A sensible, ordinary premise (a runaway pet, a contest, a lost item) scores 7 at most."),
    ("want_again", "Would a child ask to hear this story again tomorrow night? 10 = they would beg for it; 7 = they would enjoy it once; 5 or less = they would lose interest."),
]

SCORE_RESPONSE_SCHEMA_V8 = {
    "type": "object",
    "properties": {
        **{key: {"type": "number"} for key, _ in SCORE_CRITERIA_V8},
        "weakest_page": {"type": "integer"},
        "top_issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": [*[key for key, _ in SCORE_CRITERIA_V8], "weakest_page", "top_issues"],
}


# ---------------------------------------------------------------------------
# Page 7 review (safety net): align two lines with the v8 ending
# ---------------------------------------------------------------------------

def adjust_review_prompt(prompt: str) -> str:
    return (
        prompt
        .replace("includes a brief settled afterglow, not simply a stopped scene.",
                 "ends on a happy, satisfying final moment, not simply a stopped scene. It does not need to be sleepy or set at bedtime.")
        .replace("interchangeable bedtime paragraph", "interchangeable ending paragraph")
        .replace("- Page 7 must answer or fulfil the wish, question, promise, relationship need, or problem introduced on Page 1.",
                 "- By the end (answered on Page 6, shown on Page 7), the wish, question, promise, relationship need, or problem introduced on Page 1 is answered or fulfilled.")
    )
