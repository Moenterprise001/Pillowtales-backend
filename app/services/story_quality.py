"""Standard Bedtime consistency layer (2026-11-09).

Five fixes, all scoped to Standard Bedtime Stories (Story Worlds untouched):

1. Hidden story plan  - generated in the background right after Page 1 so
   Pages 2-7 follow one decided arc (opening question + its answer, setback,
   the child's decisive act, callback, final image). No parent-visible delay.
2. Quality score      - the finished story is scored against a fixed rubric
   after Page 7 is saved. Score only: pages are already live, so no rewrites.
3. No repeats         - avoid opening seeds, titles and side-character names
   used in the child's recent stories plus globally overused names.
4. Input fixes        - custom moral reaches the model, custom themes get a
   matching opening seed, unsafe custom theme/moral text is rejected.
5. Generation trace   - one JSON record per story describing exactly which
   rules, seed, plan, retries, review results and score produced it.

Every helper here fails open: if anything goes wrong the story is generated
exactly as before and the failure is recorded in the trace.
"""
from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

QUALITY_BUILD = (
    "bedtime_story_plan=20261109 bedtime_story_score=20261109 "
    "bedtime_no_repeat=20261109 custom_moral_passthrough=20261109 "
    "custom_theme_seed_match=20261109 custom_text_safety=20261109 "
    "generation_trace=20261109"
)

# Plan/score are not used for ages 0-2: those stories deliberately have no
# multi-beat plot (see bedtime_age_0_1_2_calibration).
PLAN_MIN_AGE = 3
SCORE_MIN_AGE = 3
PLAN_TIMEOUT_SECONDS = 15
SCORE_TIMEOUT_SECONDS = 30
RECENT_HISTORY_LIMIT = 10
MAX_TRACE_EVENTS = 80


# ---------------------------------------------------------------------------
# Request-scoped state (attached to the GenerateStoryRequest instance, which
# travels from the route through Page 1 and into the background task).
# ---------------------------------------------------------------------------

def get_state(request: Any) -> Dict[str, Any]:
    state = getattr(request, "_quality_state", None)
    if not isinstance(state, dict):
        state = {}
        try:
            object.__setattr__(request, "_quality_state", state)
        except Exception:
            pass
    return state


def get_trace(request: Any) -> Dict[str, Any]:
    state = get_state(request)
    trace = state.get("trace")
    if not isinstance(trace, dict):
        trace = {
            "quality_build": QUALITY_BUILD,
            "created_at_epoch": round(time.time(), 3),
            "events": [],
        }
        state["trace"] = trace
    return trace


def trace_set(request: Any, key: str, value: Any) -> None:
    try:
        get_trace(request)[key] = value
    except Exception:
        pass


def trace_event(request: Any, name: str, **data: Any) -> None:
    try:
        events = get_trace(request).setdefault("events", [])
        if len(events) >= MAX_TRACE_EVENTS:
            return
        clean = {k: (str(v)[:300] if isinstance(v, str) else v) for k, v in data.items()}
        events.append({"t": round(time.time(), 2), "event": name, **clean})
    except Exception:
        pass


def fingerprint(text: Any) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Fix 4a: custom text safety (deterministic, zero latency).
# This is a first-line guard only. Gemini's own safety settings remain.
# ---------------------------------------------------------------------------

_UNSAFE_WORDS = {
    # sexual
    "sex", "sexy", "sexual", "erotic", "erotica", "porn", "porno", "nude", "nudes",
    "naked", "prostitute", "prostitution", "stripper", "brothel", "fetish", "horny",
    "sexo", "erotico", "erotica", "desnudo", "desnuda", "prostitucion", "puta",
    "sexe", "sexuel", "nackt", "sesso",
    # hate / extremism
    "nazi", "nazis", "skinhead", "skinheads", "kkk", "hitler", "swastika",
    "genocide", "terrorist", "terrorism", "jihad",
    # violence / harm
    "violence", "violent", "murder", "murderer", "kill", "killing", "killer",
    "gore", "torture", "rape", "suicide", "selfharm", "stab", "stabbing",
    "shooting", "massacre", "violencia", "matar", "asesinato", "suicidio",
    # drugs / alcohol
    "drugs", "cocaine", "heroin", "meth", "weed", "marijuana", "cannabis",
    "vodka", "whisky", "whiskey", "drunk", "droga", "drogas", "cocaina",
}
_UNSAFE_PHRASES = ("self harm", "white power", "white pride", "blow job", "one night stand")


def _normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text.lower()).strip()


def unsafe_custom_text(*values: Any) -> Optional[str]:
    """Return the offending term if any custom theme/moral text is unsuitable."""
    for value in values:
        text = _normalise_text(value)
        if not text:
            continue
        for phrase in _UNSAFE_PHRASES:
            if phrase in text:
                return phrase
        for word in re.findall(r"[a-z]+", text):
            if word in _UNSAFE_WORDS:
                return word
    return None


# ---------------------------------------------------------------------------
# Fix 4b: theme -> opening-seed families for custom / unmapped themes.
# ---------------------------------------------------------------------------

THEME_SEED_ALLOWLIST: Dict[str, set] = {
    "space": {"ancient_observatory", "star_painter_cottage", "river_of_stars", "moon_bakery", "pillow_harbour", "floating_cloud_city", "northern_lights_village"},
    "dragons": {"dragon_market", "dragon_post_office", "sleepy_castle_hall"},
    "princess": {"glass_slipper_cafe", "sleepy_castle_hall", "hidden_garden_gate"},
    "animals": {"forest_school", "honeybee_palace", "little_lighthouse_cafe", "sleepy_forest_path", "amazon_treehouse", "jungle_waterfall", "whale_island"},
    "forest": {"sleepy_forest_path", "forest_school", "hidden_garden_gate", "rainbow_garden", "lantern_treehouse"},
    "dinosaurs": {"dinosaur_kindergarten", "hidden_dinosaur_valley"},
    "underwater": {"underwater_palace", "mermaid_library", "whale_island", "seaside_cave"},
    "adventure": {"pillow_harbour", "glowing_attic", "hidden_garden_gate", "toymaker_workshop", "little_lighthouse_cafe", "pirate_harbour", "desert_caravan", "enchanted_hot_air_balloon", "sky_train_station"},
    "magic": {"moonlit_library", "toymaker_workshop", "rainbow_garden", "hidden_garden_gate", "clockwork_greenhouse", "sleepy_puppet_theatre"},
    "winter": {"snowy_village", "northern_lights_village"},
    "trains": {"dream_train_carriage", "sky_train_station"},
    "circus": {"cloud_circus", "sleepy_puppet_theatre"},
}
THEME_ALIASES = {
    "dragon": "dragons", "animal": "animals", "dinosaur": "dinosaurs", "princesses": "princess",
    "superheroes": "adventure", "superhero": "adventure", "emotions": "forest",
}
# Seeds that adapt to almost any subject; used when a custom theme matches nothing.
NEUTRAL_SEED_FAMILIES = {"glowing_attic", "hidden_garden_gate", "toymaker_workshop", "moonlit_library", "pillow_harbour", "lantern_treehouse"}

CUSTOM_THEME_KEYWORDS: Dict[str, Iterable[str]] = {
    "underwater": ("mermaid", "merman", "ocean", "sea", "fish", "whale", "shark", "dolphin", "underwater", "beach", "octopus", "turtle", "coral", "sirena", "mar", "oceano", "pez"),
    "dinosaurs": ("dinosaur", "dino", "trex", "rex", "raptor", "dinosaurio"),
    "dragons": ("dragon", "dragons", "dragones"),
    "space": ("space", "star", "stars", "moon", "planet", "rocket", "astronaut", "alien", "galaxy", "comet", "espacio", "luna", "estrella", "cohete"),
    "winter": ("snow", "winter", "christmas", "santa", "reindeer", "ice", "frozen", "penguin", "nieve", "navidad", "invierno"),
    "trains": ("train", "trains", "railway", "tren"),
    "circus": ("circus", "clown", "juggler", "puppet", "circo", "payaso"),
    "princess": ("princess", "prince", "castle", "queen", "king", "knight", "palace", "princesa", "principe", "castillo", "reina", "rey"),
    "magic": ("magic", "wizard", "witch", "fairy", "fairies", "unicorn", "spell", "potion", "elf", "magia", "mago", "bruja", "hada", "unicornio"),
    "animals": ("animal", "animals", "rabbit", "bunny", "cat", "kitten", "dog", "puppy", "farm", "horse", "pony", "bear", "fox", "owl", "pet", "zoo", "lion", "tiger", "monkey", "elephant", "panda", "hamster", "conejo", "perro", "gato", "granja", "caballo", "oso"),
    "forest": ("forest", "woods", "tree", "trees", "garden", "flower", "flowers", "friendship", "friend", "friends", "feelings", "bosque", "jardin", "amistad"),
    "adventure": ("pirate", "pirates", "treasure", "adventure", "explorer", "explore", "balloon", "hero", "heroes", "super", "superhero", "soccer", "football", "sport", "sports", "race", "robot", "robots", "car", "cars", "desert", "aventura", "pirata", "tesoro", "futbol", "heroe"),
}


def resolve_theme_seed_families(theme: Any, custom_theme: Any) -> tuple[Optional[set], str]:
    """Return (allowed seed families, pool label) for a theme request."""
    custom = _normalise_text(custom_theme)
    key = _normalise_text(theme).replace("-", "_").replace(" ", "_")
    key = THEME_ALIASES.get(key, key)

    if not custom and key in THEME_SEED_ALLOWLIST:
        return THEME_SEED_ALLOWLIST[key], "theme_filtered"

    text = custom or _normalise_text(theme)
    words = set(re.findall(r"[a-z]+", text))
    # Light stemming so plurals and compounds match ("mermaids", "starfish").
    stems = words | {w[:-1] for w in words if w.endswith("s") and len(w) > 3} | {w[:-2] for w in words if w.endswith("es") and len(w) > 4}
    for theme_key, keywords in CUSTOM_THEME_KEYWORDS.items():
        if stems.intersection(keywords) or any(
            len(keyword) >= 4 and word.startswith(keyword) for word in words for keyword in keywords
        ):
            return THEME_SEED_ALLOWLIST[theme_key], f"custom_keyword:{theme_key}"
    if key in THEME_SEED_ALLOWLIST:
        return THEME_SEED_ALLOWLIST[key], "theme_filtered"
    if text:
        return NEUTRAL_SEED_FAMILIES, "custom_neutral"
    return None, "full_age_pool"


# ---------------------------------------------------------------------------
# Fix 3: no repeats.
# ---------------------------------------------------------------------------

# Data-driven (Supabase, 9 Nov 2026): Barnaby appeared in 17 bedtime stories.
OVERUSED_DEFAULT_NAMES = [
    "Barnaby", "Pip", "Sparkle", "Squeaky", "Whiskers", "Willow", "Barry",
    "Twinkle", "Pippin", "Bartholomew", "Mr. Fluffles",
]
COMPANION_SHORT_NAMES = {"Luna", "Milo", "Spark", "Stella", "Bramble"}


def load_recent_history(story_repo: Any, user_id: Any, child_name: Any, limit: int = RECENT_HISTORY_LIMIT) -> Dict[str, Any]:
    history: Dict[str, Any] = {"families": set(), "titles": [], "names": [], "loaded": False}
    if not story_repo or not user_id:
        return history
    client = getattr(story_repo, "client", None)
    if client is None:
        return history

    def _query(columns: str):
        query = (
            client.table("stories").select(columns)
            .eq("user_id", str(user_id))
            .is_("story_world_slug", "null")
            .order("created_at", desc=True)
            .limit(limit)
        )
        if child_name:
            query = query.eq("child_name", str(child_name))
        return query.execute().data or []

    try:
        try:
            rows = _query("title,characters,generation_trace")
        except Exception:
            rows = _query("title,characters")  # column not migrated yet
    except Exception as exc:
        history["error"] = str(exc)[:200]
        return history

    child_key = _normalise_text(child_name)
    for row in rows:
        title = str(row.get("title") or "").strip()
        if title:
            history["titles"].append(title)
        for character in row.get("characters") or []:
            name = str((character or {}).get("name") if isinstance(character, dict) else character or "").strip()
            if name and _normalise_text(name) != child_key and name not in history["names"]:
                history["names"].append(name)
        trace = row.get("generation_trace") or {}
        family = ((trace.get("seed") or {}).get("family")) if isinstance(trace, dict) else None
        if family:
            history["families"].add(family)
    history["loaded"] = True
    return history


def avoid_names_for(request: Any, history: Dict[str, Any], companion: Optional[dict]) -> List[str]:
    keep = {_normalise_text(getattr(request, "childName", ""))}
    if companion:
        keep.add(_normalise_text(companion.get("short_name")))
        keep.add(_normalise_text(companion.get("name")))
    for character in getattr(request, "characters", None) or []:
        keep.add(_normalise_text(getattr(character, "name", "")))
    names: List[str] = []
    for name in [*history.get("names", []), *OVERUSED_DEFAULT_NAMES, *sorted(COMPANION_SHORT_NAMES)]:
        if _normalise_text(name) in keep or name in names:
            continue
        # Skip generic family labels such as Nan / Daddy / Grandma.
        if _normalise_text(name) in {"nan", "nana", "daddy", "mummy", "mommy", "mum", "mom", "dad", "grandma", "grandad", "grandpa", "granny"}:
            continue
        names.append(name)
    return names[:25]


def avoid_repeats_block(request: Any, history: Dict[str, Any], companion: Optional[dict]) -> str:
    names = avoid_names_for(request, history, companion)
    titles = history.get("titles", [])[:RECENT_HISTORY_LIMIT]
    if not names and not titles:
        return ""
    lines = ["FRESHNESS — AVOID REPEATS:"]
    if titles:
        lines.append("- This child has already heard stories with these titles. Do not reuse or closely echo them: " + "; ".join(titles) + ".")
    if names:
        lines.append("- Do not use these names for any new side character (they are overused or already heard): " + ", ".join(names) + ". Invent a fresh, easy-to-say name that suits the world.")
    lines.append("- Do not repeat a premise this child has already heard (for example the same lost object, same letter mystery, same school).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fix 1: hidden story plan.
# ---------------------------------------------------------------------------

PLAN_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "opening_question": {"type": "string"},
        "answer_to_opening_question": {"type": "string"},
        "hidden_explanation": {"type": "string"},
        "stakes": {"type": "string"},
        "beats": {"type": "array", "items": {"type": "string"}, "minItems": 6, "maxItems": 6},
        "setback": {"type": "string"},
        "childs_decisive_act": {"type": "string"},
        "callback_detail": {"type": "string"},
        "final_image": {"type": "string"},
        "threads_to_close": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "opening_question", "answer_to_opening_question", "hidden_explanation", "stakes", "beats",
        "setback", "childs_decisive_act", "callback_detail", "final_image", "threads_to_close",
    ],
}


def _age_stakes_line(age: int) -> str:
    if age <= 4:
        return "Stakes: one small, concrete, gentle worry (a lost toy, a sad friend). No danger."
    if age <= 6:
        return "Stakes: a clear, gentle problem with a visible consequence if unsolved. No real danger."
    if age <= 8:
        return "Stakes: a real problem with a clear cost if it fails and one moment where it looks like it might fail."
    return (
        "Stakes for an older child: a real worry with a concrete cost or time limit, one genuine setback where the "
        "child's first idea fails, and a solution that requires the child's own insight. Non-graphic and bedtime-safe."
    )


def build_plan_prompt(request: Any, title: str, page_one: str, moral_requested: bool, avoid_block: str, age: int) -> str:
    theme = getattr(request, "customTheme", None) or getattr(request, "theme", "")
    moral = getattr(request, "moral", "")
    moral_line = (
        f"- Private moral target: {moral}. The child's decisive act must DEMONSTRATE it through action and consequence. "
        "The plan must never have any character name, explain, or state the moral as a lesson."
        if moral_requested else
        "- No moral was requested. Do not plan one."
    )
    return f"""You are the story editor for a seven-page children's bedtime story. Page 1 is already written and has
already been read aloud, so it is fixed. Plan Pages 2-7 so the finished story is excellent.
Return only JSON matching the schema. Write the plan in English even if the story language differs.

STORY FACTS:
- Title: {title}
- Child (protagonist): {getattr(request, 'childName', '')}
- Theme: {theme}
- Internal reading age: {age}
{moral_line}

PAGE 1 (FIXED):
{page_one}

PLANNING RULES:
- opening_question: the single question, wish or problem Page 1 raises, in one sentence.
- answer_to_opening_question: exactly how Page 7 answers it. Every question Page 1 raises must be answered
  (for example: who sent the letter, what hatches from the egg, why the song is lost).
- hidden_explanation: the true explanation behind any mystery on Page 1, decided now, consistent with Page 1.
  It must be REVEALED ALOUD in one of the beats (say it in that beat), never left only in these notes.
- {_age_stakes_line(age)}
- stakes: what is at risk. The listener must HEAR the stakes on the page by Page 3 (put them in the Page 2 or
  Page 3 beat), and the setback beat must make it look as if the bad outcome could really happen.
- beats: exactly six strings, one per page for Pages 2,3,4,5,6,7. Each 1-2 sentences describing what HAPPENS.
  Every page must move the story forward; no page may only repeat a search or re-explain the problem.
  Page 2 develops the hook. Pages 3-4 complicate it. Page 4 or 5 contains the setback.
  Page 6 is the climax and contains the child's decisive act. Page 7 is consequence, answered question,
  callback and a calm, settled bedtime landing. Page 7 must not need to fit a climax as well.
- setback: the moment the child's first approach fails or things get worse.
- childs_decisive_act: what the child THEMSELF does on Page 6 that causes the resolution. It must be the
  child's own idea using something established earlier. Never luck, an accident, an adult or helper solving it.
  {_decisive_act_cost_line(age)}
- callback_detail: one specific Page 1 detail that returns meaningfully on Page 7.
- final_image: the concrete, calm last image of the story.
- threads_to_close: every open thread that must be resolved by Page 7. Each must be answered in words the
  listener hears on the page, so put each answer into a specific beat.
- Introduce no new important character after Page 4. Keep the cast small.
- Keep Page 1 facts exactly; do not contradict them.
- NAMES: any new character or creature you name must NOT use a name from the avoid list below. Banned names
  are banned in the plan too; a plan that uses one is rejected.
{avoid_block}
"""


def _decisive_act_cost_line(age: int) -> str:
    if age >= 7:
        return ("For this age the decisive act must cost the child something real (giving up something they love, "
                "facing a fear, admitting a mistake) or take visible courage, so the moral lands without being stated.")
    return "Keep the decisive act simple, gentle and clearly the child's own."


def banned_names_in_text(text: Any, names: Iterable[str]) -> List[str]:
    """Avoided names used as names (whole word, capitalised as given) in text.

    Case-sensitive on purpose: "her voice was squeaky" must not match the
    banned name "Squeaky", and "a willow tree" must not match "Willow".
    """
    haystack = str(text or "")
    found = []
    for name in names:
        name = str(name or "").strip()
        if len(name) < 2:
            continue
        if re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", haystack):
            found.append(name)
    return found


def plan_text(plan: Any) -> str:
    try:
        import json as _json
        return _json.dumps(plan, ensure_ascii=False)
    except Exception:
        return str(plan)


def plan_is_valid(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    beats = plan.get("beats")
    if not isinstance(beats, list) or len(beats) < 6 or not all(str(b).strip() for b in beats[:6]):
        return False
    for key in ("opening_question", "answer_to_opening_question", "childs_decisive_act", "callback_detail", "final_image"):
        if not str(plan.get(key) or "").strip():
            return False
    return True


def plan_spine_block(plan: Optional[dict]) -> str:
    """Appended to the story spine: seen by every continuation page and by the Page 7 reviewer."""
    if not plan_is_valid(plan):
        return ""
    beats = "\n".join(f"  - Page {index + 2}: {str(beat).strip()}" for index, beat in enumerate(plan["beats"][:6]))
    threads = "; ".join(str(t).strip() for t in (plan.get("threads_to_close") or []) if str(t).strip()) or "none listed"
    return f"""
LOCKED STORY PLAN — PRIVATE EDITORIAL NOTES (follow it; never quote these notes in the story):
- Opening question: {plan.get('opening_question')}
- It is answered on Page 7 like this: {plan.get('answer_to_opening_question')}
- Hidden explanation (stay consistent with it): {plan.get('hidden_explanation')}
- Stakes: {plan.get('stakes')}
- Setback: {plan.get('setback')}
- The child's decisive act (Page 6, the child's own idea): {plan.get('childs_decisive_act')}
- Callback detail for Page 7: {plan.get('callback_detail')}
- Final image: {plan.get('final_image')}
- Threads that must be closed by Page 7, each answered in words the listener hears: {threads}
- The stakes must be heard on the page by Page 3 and feel truly at risk at the setback.
- The hidden explanation must be revealed on the page, not left implied.
- Page beats:
{beats}
- The story is incomplete if the opening question is unanswered, the child's decisive act is replaced by luck, an accident or a helper, or any listed thread is left open.
"""


def plan_page_focus_block(plan: Optional[dict], next_page_number: int, batch_count: int) -> str:
    if not plan_is_valid(plan):
        return ""
    lines = []
    for page_number in range(next_page_number, next_page_number + max(batch_count, 1)):
        index = page_number - 2
        if 0 <= index < 6:
            lines.append(f"- Page {page_number} MUST deliver this beat: {str(plan['beats'][index]).strip()}")
    if not lines:
        return ""
    later = [
        f"Page {i + 2}" for i in range(6) if i + 2 >= next_page_number + max(batch_count, 1)
    ]
    tail = (
        f"- Do not jump ahead to the beats reserved for {', '.join(later)}."
        if later else
        "- This is the final page: deliver the planned answer, close every listed thread, use the callback and land on the final image."
    )
    return "THIS PAGE'S JOB FROM THE LOCKED PLAN:\n" + "\n".join(lines) + "\n" + tail + "\n"


def inject_before_output_format(prompt: str, block: str) -> str:
    if not block:
        return prompt
    marker = "OUTPUT FORMAT STRICT:"
    if marker in prompt:
        return prompt.replace(marker, block + "\n" + marker, 1)
    return prompt + "\n\n" + block


# ---------------------------------------------------------------------------
# Fix 2: quality score (score only).
# ---------------------------------------------------------------------------

SCORE_CRITERIA = [
    ("opening_promise_resolved", "Every question/wish/mystery raised on Page 1 is clearly answered by the end."),
    ("child_drives_climax", "The child's own idea or action causes the resolution; not luck, an accident or a helper."),
    ("stakes_and_tension", "There is a real, age-appropriate problem, a genuine setback, and a moment of doubt."),
    ("coherence", "No confusing contradictions, muddled objects or loose threads; every introduced element matters."),
    ("pacing", "Each page moves the story forward; the middle does not stall and the ending is not crammed."),
    ("moral_shown_not_told", "The moral (if any) is shown through choices and consequences, never stated or lectured."),
    ("age_fit_language", "Vocabulary and sentence craft match the listener's age, neither babyish nor too adult."),
    ("imagery_and_humour", "Specific, memorable images and at least one genuinely funny or delightful moment."),
    ("bedtime_landing", "The ending gives consequence, an earned callback and a calm, settled close."),
    ("originality", "Fresh premise and characters; nothing generic or interchangeable with another story."),
]

SCORE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        **{key: {"type": "number"} for key, _ in SCORE_CRITERIA},
        "weakest_page": {"type": "integer"},
        "top_issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": [*[key for key, _ in SCORE_CRITERIA], "weakest_page", "top_issues"],
}


def build_score_prompt(request: Any, title: str, pages: List[str], moral_requested: bool, age: int) -> str:
    theme = getattr(request, "customTheme", None) or getattr(request, "theme", "")
    moral = getattr(request, "moral", "") if moral_requested else "none requested (score moral_shown_not_told as 10)"
    criteria = "\n".join(f"- {key}: {desc}" for key, desc in SCORE_CRITERIA)
    story = "\n\n".join(f"Page {i + 1}: {page}" for i, page in enumerate(pages))
    return f"""You are a demanding children's book editor. Score this bedtime story for a {age}-year-old listener.
Return only JSON matching the schema. Score each criterion 0-10 (one decimal allowed).

CALIBRATION:
- 10 = publishable picture-book quality with no weaknesses on that criterion.
- 9 = excellent, only a trivial nitpick.
- 7-8 = good but with a noticeable weakness a parent might feel.
- 5-6 = a clear flaw (e.g. an unanswered opening question, the problem solved by accident, a crammed ending).
- 0-4 = broken on that criterion.
Be strict and consistent. Do not reward length or pretty words. Judge only what is on the page.

STORY FACTS: title "{title}"; theme {theme}; moral {moral}.

CRITERIA:
{criteria}
- weakest_page: the page number (1-{len(pages)}) that most needs improvement.
- top_issues: 1-4 short, specific issues, most important first.

STORY:
{story}
"""


def summarise_score(result: Dict[str, Any]) -> Dict[str, Any]:
    scores: Dict[str, float] = {}
    for key, _ in SCORE_CRITERIA:
        try:
            scores[key] = max(0.0, min(10.0, float(result.get(key))))
        except Exception:
            continue
    overall = round(sum(scores.values()) / len(scores), 2) if scores else None
    return {
        "overall": overall,
        "criteria": scores,
        "weakest_page": result.get("weakest_page"),
        "top_issues": [str(i)[:200] for i in (result.get("top_issues") or [])][:4],
        "rubric_version": "20261109",
    }
