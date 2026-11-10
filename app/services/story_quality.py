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
    "role_names_fix=20261110 plan_rename=20261110 title_fresh=20261110 page2_first=20261110c draft_tolerant=20261110b escaped_newline_fix=20261110 page1_best_of=20261110 draft_revise=20261110 oxford_age_standard=20261110 model_routing=20261110 bedtime_prompts_v5c=20261110 plan_check=20261110 opening_seeds=20261110 bedtime_story_plan=20261109b plan_thinking=20261109 page1_name_freshness=20261109 bedtime_story_score=20261109 "
    "bedtime_no_repeat=20261109 custom_moral_passthrough=20261109 "
    "custom_theme_seed_match=20261109 custom_text_safety=20261109 "
    "generation_trace=20261109"
)

# Plan/score are not used for ages 0-2: those stories deliberately have no
# multi-beat plot (see bedtime_age_0_1_2_calibration).
PLAN_MIN_AGE = 3
SCORE_MIN_AGE = 3
PLAN_TIMEOUT_SECONDS = 20          # thinking plan (first attempt)
PLAN_FAST_TIMEOUT_SECONDS = 12     # fast fallback / banned-name retry
PLAN_MAX_OUTPUT_TOKENS = 6000      # thinking tokens share this budget
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
    "dragons": {"dragon_market", "dragon_post_office", "dragon_nursery", "dragon_lighthouse", "dragon_bakery", "dragon_flying_school", "dragon_cloud_stables", "dragon_strawberry_garden"},
    "princess": {"glass_slipper_cafe", "sleepy_castle_hall", "princess_tea_garden", "princess_lantern_festival", "princess_sky_ponies", "princess_palace_nursery"},
    "animals": {"rock_pool_friends", "forest_school", "honeybee_palace", "little_lighthouse_cafe", "sleepy_forest_path", "amazon_treehouse", "jungle_waterfall", "whale_island"},
    "forest": {"sleepy_forest_path", "forest_school", "hidden_garden_gate", "rainbow_garden", "lantern_treehouse"},
    "dinosaurs": {"dinosaur_kindergarten", "hidden_dinosaur_valley", "dinosaur_museum_night", "dinosaur_island_farm", "dinosaur_puddle_park", "fossil_dig"},
    "underwater": {"underwater_palace", "mermaid_library", "whale_island", "seaside_cave", "rock_pool_friends", "bubble_reef_school"},
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


def seed_text_signatures(seed_families: Iterable[dict]) -> Dict[str, str]:
    """A distinctive lowercase phrase from each opening, used to spot it in old Page 1 text."""
    signatures: Dict[str, str] = {}
    for seed in seed_families or []:
        words = re.findall(r"[a-z']+", _normalise_text(str(seed.get("en") or "").replace("{childName}", "")))
        if len(words) >= 8:
            signatures[seed.get("family")] = " ".join(words[2:8])
    return signatures


def load_recent_history(story_repo: Any, user_id: Any, child_name: Any, limit: int = RECENT_HISTORY_LIMIT,
                        seed_signatures: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
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
            rows = _query("title,characters,generation_trace,first_page:pages->>0")
        except Exception:
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
        if isinstance(trace, dict) and trace.get("page1_hook"):
            history.setdefault("hooks", []).append(trace.get("page1_hook"))
        family = ((trace.get("seed") or {}).get("family")) if isinstance(trace, dict) else None
        if family:
            history["families"].add(family)
        elif row.get("first_page") and seed_signatures:
            # Stories made before the trace existed: recognise the opening from Page 1.
            page_one = " ".join(re.findall(r"[a-z']+", _normalise_text(row.get("first_page"))))
            for fam, signature in seed_signatures.items():
                if signature and signature in page_one:
                    history["families"].add(fam)
                    history.setdefault("families_from_text", []).append(fam)
                    break
    history["loaded"] = True
    return history


# Titles and role words saved as "character names" (e.g. "The Princess", "Princess Lily",
# "Captain Thistle") must not ban the word itself: a princess story could not say "Princess".
ROLE_WORDS = {
    "the", "a", "an", "princess", "prince", "king", "queen", "captain", "elder", "old", "young", "little", "baby", "big",
    "mr", "mrs", "ms", "miss", "mister", "lady", "lord", "sir", "dame", "dr", "doctor", "professor", "uncle", "aunt",
    "auntie", "granny", "grandma", "grandpa", "grandad", "grandmother", "grandfather", "nan", "nana", "mum", "mummy",
    "mom", "mommy", "dad", "daddy", "papa", "mama", "farmer", "wizard", "witch", "fairy", "mermaid", "dragon", "giant",
    "royal", "announcer", "advisor", "guard", "knight", "teacher", "chef", "baker", "keeper", "star", "painter",
}


def clean_character_name(name: Any) -> str:
    """'Princess Lily' -> 'Lily'; 'The Princess' -> ''; 'Captain Thistle' -> 'Thistle'."""
    words = str(name or "").split()
    kept = [w for w in words if re.sub(r"[^a-z]", "", w.lower()) not in ROLE_WORDS]
    return " ".join(kept).strip()


def avoid_names_for(request: Any, history: Dict[str, Any], companion: Optional[dict]) -> List[str]:
    keep = {_normalise_text(getattr(request, "childName", ""))}
    if companion:
        keep.add(_normalise_text(companion.get("short_name")))
        keep.add(_normalise_text(companion.get("name")))
    for character in getattr(request, "characters", None) or []:
        keep.add(_normalise_text(getattr(character, "name", "")))
    names: List[str] = []
    # Overused names first: they must never be pushed out by a long history.
    history_names = [clean_character_name(n) for n in history.get("names", [])]
    for name in [*OVERUSED_DEFAULT_NAMES, *sorted(COMPANION_SHORT_NAMES), *history_names]:
        if not name or _normalise_text(name) in keep or name in names:
            continue
        # Skip generic family labels such as Nan / Daddy / Grandma.
        if _normalise_text(name) in {"nan", "nana", "daddy", "mummy", "mommy", "mum", "mom", "dad", "grandma", "grandad", "grandpa", "granny"}:
            continue
        names.append(name)
    return names[:40]


# Data-driven (Supabase, 10 Nov 2026): "whispering" was in 22 bedtime titles,
# "secret" 11, "lost" 8, "sleepy" 7, "brave" 5.
OVERUSED_TITLE_WORDS = ["Whispering", "Secret", "Lost", "Sleepy", "Brave", "Silver", "Little", "Starry", "Magical", "Mysterious"]


def overused_title_words_in(title: Any) -> List[str]:
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", str(title or ""))}
    return [w for w in OVERUSED_TITLE_WORDS if w.lower() in words]


def avoid_repeats_block(request: Any, history: Dict[str, Any], companion: Optional[dict]) -> str:
    names = avoid_names_for(request, history, companion)
    titles = history.get("titles", [])[:RECENT_HISTORY_LIMIT]
    if not names and not titles and not prompts_v5_enabled():
        return ""
    lines = ["FRESHNESS — AVOID REPEATS:"]
    if titles:
        lines.append("- This child has already heard stories with these titles. Do not reuse or closely echo them: " + "; ".join(titles) + ".")
    if names:
        lines.append("- Do not use these names for any new side character (they are overused or already heard): " + ", ".join(names) + ". Invent a fresh, easy-to-say name that suits the world.")
    lines.append("- Do not repeat a premise this child has already heard (for example the same lost object, same letter mystery, same school).")
    if prompts_v5_enabled():
        lines.append("- TITLE: do not use these overused title words: " + ", ".join(OVERUSED_TITLE_WORDS) + ". Never put the moral or a form of it in the title. Name the specific thing this story is about.")
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
- PLANT AND PAYOFF: the decisive act must use something specific planted on Page 2, 3 or 4 (an object, a habit,
  a clue, a skill). It must be clever: something only this child would think of. A helper or the creature could
  not simply have done it themselves.
- TITLE PROMISE: honour every word of the title (if the title says "grumpy", someone is truly grumpy and why is answered).
- THEME PROMISE: the theme "{theme}" must be literally present, not only a metaphor. A princess theme needs a real
  princess (or the child becoming one); dragons need a real dragon; space needs space. If Page 1 only hints at the
  theme, bring it in for real by Page 3 in a way that fits Page 1.
- BEDTIME LANDING: the final image is calm and settled, ideally the child at home, evening or night, with a quiet
  feeling of having made something right.

{(V5_PLAN_RULES + chr(10) + chr(10)) if prompts_v5_enabled() else ""}SELF-CHECK BEFORE ANSWERING (fix the plan if any answer is "no"):
1. Is the stakes line heard on the page by Page 3, and does it feel at real risk at the setback?
2. Could any character other than the child have solved it just as easily? (Must be "no".)
3. Does every beat move the story forward, with the answer to every thread placed in a specific beat?
4. Is there any contradiction with Page 1 or inside the plan (e.g. a "new" thing that already exists, a name known before it is learned)?
5. Does the decisive act show the moral through action, without anyone stating it?
{avoid_block}
"""


def plan_thinking_budget() -> int:
    """Thinking tokens for the background plan call (0 disables). Env: BEDTIME_PLAN_THINKING_BUDGET."""
    import os
    try:
        return max(0, min(8192, int(os.getenv("BEDTIME_PLAN_THINKING_BUDGET", "2048"))))
    except Exception:
        return 2048


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


def clean_beats(beats: Iterable[Any]) -> List[str]:
    return [re.sub(r"^\s*page\s*\d+\s*[:.\-–]\s*", "", str(beat).strip(), flags=re.IGNORECASE) for beat in beats]


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
    precedence = ("\n" + V5_PLAN_PRECEDENCE) if prompts_v5_enabled() else ""
    return "THIS PAGE'S JOB FROM THE LOCKED PLAN:\n" + "\n".join(lines) + "\n" + tail + precedence + "\n"


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
    ("stakes_and_tension", "There is a concrete, bedtime-safe stake shown on the page (something will be lost, missed, broken or spoiled, and someone will mind), a genuine setback where that loss nearly happens, and a moment of doubt. Do not reward danger or fear: this is a bedtime story, and frightening content should lower this score."),
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


def build_score_prompt(request: Any, title: str, pages: List[str], moral_requested: bool, age: int,
                       age_standard: str = "") -> str:
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
{("AGE STANDARD (score age_fit_language against this; lower it for words or sentences above this age band):" + chr(10) + age_standard.strip() + chr(10)) if age_standard else ""}
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
        "rubric_version": "20261110b",
    }


# ---------------------------------------------------------------------------
# Page 1 name freshness. Page 1 is written for speed with no plan, and Gemini
# defaults to "Barnaby" even when told not to. Page 1 is not yet shown when
# this runs, so a whole-word swap to a fresh name is safe and instant.
# ---------------------------------------------------------------------------

FRESH_NAME_POOL = [
    "Quill", "Fennel", "Nutmeg", "Tansy", "Sorrel", "Juniper", "Tumble", "Pebblewick", "Marlo",
    "Clementine", "Hollis", "Wren", "Bellamy", "Ottoline", "Figgy", "Rosco", "Tibbet", "Mabel",
    "Ziggy", "Pernel", "Lark", "Dimble", "Saffron", "Brindle", "Orla", "Thistle", "Cosmo", "Nell",
]


def freshen_page_one_names(title: str, page_one: str, banned: Iterable[str], extra_avoid: Iterable[str] = ()) -> tuple:
    """Replace banned names in title/Page 1. Returns (title, page_one, {old: new})."""
    used = banned_names_in_text(f"{title}\n{page_one}", banned)
    if not used:
        return title, page_one, {}
    avoid = {_normalise_text(n) for n in [*banned, *extra_avoid]}
    present = set(re.findall(r"[A-Za-z]+", f"{title} {page_one}"))
    import random as _random
    pool = [n for n in FRESH_NAME_POOL if _normalise_text(n) not in avoid and n not in present]
    _random.shuffle(pool)
    mapping = {}
    for old in used:
        if not pool:
            break
        new = pool.pop()
        mapping[old] = new
        pattern = r"(?<![\w])" + re.escape(old) + r"(?![\w])"
        title = re.sub(pattern, new, title)
        page_one = re.sub(pattern, new, page_one)
    return title, page_one, mapping


def rename_names_in_plan(plan: Any, names: Iterable[str], extra_avoid: Iterable[str] = (), keep_text: str = "") -> tuple:
    """Swap banned names in a plan for fresh ones (whole word, every field). Returns (plan, {old: new})."""
    import json as _json
    import random as _random
    names = [n for n in names if n]
    if not names:
        return plan, {}
    text = _json.dumps(plan, ensure_ascii=False)
    avoid = {_normalise_text(n) for n in [*names, *extra_avoid]}
    present = set(re.findall(r"[A-Za-z]+", text + " " + keep_text))
    pool = [n for n in FRESH_NAME_POOL if _normalise_text(n) not in avoid and n not in present]
    _random.shuffle(pool)
    mapping: Dict[str, str] = {}
    for old in names:
        if not pool:
            break
        mapping[old] = pool.pop()
        text = re.sub(r"(?<![\w])" + re.escape(old) + r"(?![\w])", mapping[old], text)
    try:
        return _json.loads(text), mapping
    except Exception:
        return plan, {}


def freshen_continuation_names(pages: List[str], banned: Iterable[str], state: Dict[str, Any],
                               keep_text: str = "", extra_avoid: Iterable[str] = ()) -> tuple:
    """Rename banned names newly introduced on Pages 2+, consistently across pages.

    keep_text (Page 1 and the plan) holds names that are already part of the
    story; those are never renamed. Returns (pages, new_mapping_entries).
    """
    mapping: Dict[str, str] = state.setdefault("continuation_name_map", {})
    banned_list = [n for n in banned if not banned_names_in_text(keep_text, [n])]
    found = banned_names_in_text("\n".join(pages), banned_list)
    added = {}
    if found:
        avoid = {_normalise_text(n) for n in [*banned, *extra_avoid, *mapping.values()]}
        present = set(re.findall(r"[A-Za-z]+", keep_text + " " + " ".join(pages)))
        import random as _random
        pool = [n for n in FRESH_NAME_POOL if _normalise_text(n) not in avoid and n not in present]
        _random.shuffle(pool)
        for old in found:
            if old not in mapping and pool:
                mapping[old] = pool.pop()
                added[old] = mapping[old]
    if not mapping:
        return pages, added
    out = []
    for page in pages:
        for old, new in mapping.items():
            page = re.sub(r"(?<![\w])" + re.escape(old) + r"(?![\w])", new, page)
        out.append(page)
    return out, added


# ===========================================================================
# v5 (2026-11-10): de-duplicated, contradiction-free continuation prompt and
# a plan checker. Everything here is behind switches:
#   BEDTIME_PROMPTS_V5=0       -> continuation prompts revert to v4 exactly
#   BEDTIME_PLAN_CHECK_ENABLED=0 -> no plan check / revision
# ===========================================================================

V5_BUILD = "bedtime_prompts_v5=20261110 plan_check=20261110"


def _env_on(name: str, default: str = "1") -> bool:
    import os
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def prompts_v5_enabled() -> bool:
    return _env_on("BEDTIME_PROMPTS_V5")


def plan_check_enabled() -> bool:
    return _env_on("BEDTIME_PLAN_CHECK_ENABLED")


def v5_age_block(age: int) -> str:
    if age <= 5:
        level = (
            "- Use clear early-childhood bedtime language. Most sentences should be 5-10 words.\n"
            "- One place at a time, no more than one helper, and obvious cause and effect.\n"
            "- Concrete everyday words; no layered clues, abstract feelings or poetic description."
        )
    elif age == 6:
        level = (
            "- Use simple, lively sentences of about 6-12 words with the occasional fun new word.\n"
            "- Keep one clear thread and make each cause and effect easy to follow."
        )
    elif age <= 8:
        level = (
            "- Use confident child-friendly adventure language. Most sentences should be 8-14 words.\n"
            "- Keep to the existing cast and one main thread; do not pile up new names, objects, places or rules.\n"
            "- No adult, poetic, cinematic or fantasy-novel prose, and no advanced words unless truly age-suitable. If a sentence sounds impressive, simplify it."
        )
    else:
        level = (
            "- Use richer but bedtime-safe language with varied sentences. Inference, a real dilemma and gentle wit are welcome.\n"
            "- Keep one main thread. Avoid dense world-building or adult literary prose."
        )
    return (
        f"AGE CALIBRATION FOR PAGES 2-7 (internal reading age {age}; never state the number in the story):\n" + level
    )


def v5_craft_block(moral_requested: bool, moral: Any) -> str:
    moral_rule = (
        f"- Requested moral: {moral}. It is private design metadata. Never use the moral word in the title or prose, and never name, "
        "define, paraphrase, interpret or explain it in dialogue, narration, thought or the ending. Build it into a concrete temptation, "
        "one meaningful choice and its natural consequence, shown by Pages 4-6. MORAL REMOVAL TEST: with all commentary removed, "
        "the actions and consequences alone must make the idea inferable. Once they do, move on."
        if moral_requested else
        "- No moral was requested. Do not invent or force a lesson."
    )
    return f"""STORY CRAFT — EACH RULE STATED ONCE:
Voice
- Write like an experienced, warm, witty children's author reading aloud at bedtime: original, never imitating a named author, never sounding generated.
- Make the prose specific to this story. One unusual story-only detail beats several generic glowing, sparkling or magical descriptions. Use at most one memorable comparison per page, and sensory detail sparingly and concretely.
- Include at least one line, comic beat, behaviour or image across the story that only this story could have and a child might repeat tomorrow.
- Avoid stock storybook reactions and phrases: tummy flutters or flip-flops, widened or sparkling eyes, warmth in the chest, gasps, sinking hearts, slumped shoulders, "her eyes twinkled", "This was definitely...", "With a determined breath...", "she realised that...". Show feelings through what characters do, say, choose or avoid.
- Avoid overusing: tiny, little, small, soft, gentle, golden, silver, shimmering, glowing, sparkling, whispering, moonlit, sleepy, softly, slowly. Never describe the child as little, small, tiny or young (no "little hands", "little girl", "young explorer").

Characters and dialogue
- Give characters personality through what they say, notice, misunderstand, attempt and choose. Let the protagonist hesitate, guess wrongly or make a harmless mistake when it strengthens the eventual solution.
- Prefer short dialogue, action and funny behaviour over narrator explanation. Dialogue must sound spoken and must change something: a decision, clue, disagreement, plan or relationship. Never use it only to explain the plot.
- A quoted line and its attribution are one beat: write "Where is it?" she wondered. Never start a new paragraph with a short speech tag or reaction that belongs to the line before.
- Introduce the child clearly by name, then prefer natural pronouns. Bring the name back after a change of focus or when pronouns could be unclear. Avoid several sentences in a row that start with the same name or pronoun.
- Nobody knows a new character's name until it is given on the page (they say it, someone introduces them, or it is written somewhere). Until then, call them by what they look like.

Story
- The theme is a real promise: its central subject must affect the plot and interact with the child, early enough to matter, not sit in the background or arrive only at the end.
- By the end of Page 2 the listener knows what the child wants or must solve. The central conflict is visible by Page 3, and Pages 4 and 5 deepen that same conflict rather than replacing it.
- The world must not hand the child every clue. The child works out at least one important connection, makes the meaningful choices and causes the solution. Helpers may offer information or opportunities and may make funny mistakes, but they never solve it, and no adult, magic object, gift or coincidence fixes it for the child.
- Every conspicuous setup (object, smell, phrase, rule, promise, joke, mystery) must pay off later or not be emphasised. Bring an earlier detail back when it becomes useful or meaningful.
- Use magic where it matters across the story. This page does not need a new magical element: a quiet choice, conversation, failed idea or funny moment can carry a scene. Harmless silliness is welcome when it grows from character.
- Keep the seven-page structure invisible: no narrator commentary or transition phrases that announce what a page is for.

Stakes — bedtime-safe is not stake-free
- Bedtime-safe means no fear, injury, cruelty or danger and no unresolved cliffhanger. It does NOT mean nothing is at risk.
- Make the stakes concrete and visible on the page: show what will be lost, missed, broken, spoiled or left behind, and who will be sad, if the child fails. Let one moment make that loss look likely before the child's decisive act.

Moral
{moral_rule}

Read-aloud rhythm
- Write for the ear. Vary sentence length and shape within the age level: mix short action or dialogue lines with occasional longer flowing ones, avoid runs of sentences with the same pattern or opening, and use punctuation as breathing guidance.
- Paragraphs follow real story beats and need not be equal. Keep refrains or comic repetition only when the return does new work.
- Before returning a page, read it silently as spoken prose and fix anything mechanical, rushed or awkward without adding filler.

Plain prose
- Story text is plain prose only: no Markdown, asterisks, underscores, headings, bullets or backticks. Show emphasis through wording and rhythm.
- Never state the child's numerical age.
"""


def v5_continuation_job(remaining_page_count: int, next_page_number: int, final_page_number: int) -> str:
    pages_word = "page" if remaining_page_count == 1 else "pages"
    return f"""CONTINUATION JOB:
- Write exactly {remaining_page_count} new {pages_word}: Page {next_page_number} through Page {final_page_number}.
- Continue from the latest existing page. Do not recap or contradict it.
- Keep one main story idea visible. Something must change on every page (see FORWARD PROGRESSION); pages must not only search, wait, look around or explain.
- Compare the last two existing pages before writing. If they used the same kind of scene, this page must use a different kind.
- Every page includes something a child can picture or remember.
- Pages before the resolution end with a small reason to keep listening: a clue, choice, funny complication or clear next step, never a frightening cliffhanger.
- Where it fits, include a warm funny or surprising moment that changes what happens next.
"""


V5_PAGE5_OLD = "- Success should briefly feel uncertain, but never frightening."
V5_PAGE5_NEW = (
    "- Success must look genuinely in doubt: the concrete loss from the stakes nearly happens. Keep it bedtime-safe "
    "(no fear, injury or danger), but do not soften the loss itself."
)


def v5_page_role(page_role: str) -> str:
    return page_role.replace(V5_PAGE5_OLD, V5_PAGE5_NEW)


V5_PLAN_PRECEDENCE = (
    "- If this beat conflicts with the page role or ending rules (Page 6: the child's own act resolves it; "
    "Page 7: aftermath and settling only, nothing new happens), follow the page role and adjust the beat."
)

V5_PLAN_RULES = """- PAGE 6 COMPLETES THE SOLUTION: the child's decisive act on Page 6 must itself cause the outcome. Helpers may assist,
  but no adult, royal, parent, wizard, creature or magic may heal, fix, rescue, decide or finish it for the child.
- PAGE 7 IS AFTERMATH ONLY: the visible result, the callback and calm settling. No character arrives, and nothing is
  rescued, healed, repaired, revealed, travelled to or decided on Page 7.
- VISIBLE STAKES: name one visible sign of the possible loss that the listener sees on the page by Page 3 and that gets
  worse at the setback (for example: the market stays dark and the cocoa goes cold; the egg grows cooler each hour).
- ANSWER EVERYTHING BY PAGE 6: put the hidden explanation and the answer to every thread into the text of a specific
  Page 2-6 beat. Page 7 may show results but must not explain anything new.
- FRESH IMAGES: avoid overused motifs such as whispering woods, sparkly feathers, glowing moss and "a shimmer of magic"."""


PLAN_CHECK_KEYS = [
    ("stakes_visible_by_page3", True, "A concrete, visible sign of what will be lost appears in the Page 2 or Page 3 beat and gets worse at the setback."),
    ("child_causes_solution", True, "The Page 6 beat shows the child's own act causing the outcome; no adult, royal, helper, creature, magic, luck or coincidence heals, fixes, rescues or finishes it."),
    ("page7_aftermath_only", True, "The Page 7 beat contains only the result, callback and settling: no arrival, rescue, healing, repair, reveal, journey or decision."),
    ("threads_answered_by_page6", True, "The hidden explanation and every item in threads_to_close is actually answered inside a specific Page 2-6 beat (check the beat text itself, not the thread list); nothing is left to be explained on Page 7."),
    ("payoff_planted_early", True, "The decisive act uses an object, habit, clue or skill planted in the Page 2, 3 or 4 beat."),
    ("no_contradictions", True, "Nothing contradicts Page 1 or another part of the plan, and no name or fact is known before it is learned."),
    ("theme_literal", True, "The chosen theme's central subject is literally present and matters to the plot."),
    ("bedtime_landing", False, "The final image is calm and settled, ideally at home in the evening or at night."),
]

PLAN_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        **{key: {"type": "boolean"} for key, _, _ in PLAN_CHECK_KEYS},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": [*[key for key, _, _ in PLAN_CHECK_KEYS], "issues"],
}


def build_plan_check_prompt(request: Any, page_one: str, plan: dict, age: int) -> str:
    theme = getattr(request, "customTheme", None) or getattr(request, "theme", "")
    checks = "\n".join(f"- {key}: {desc}" for key, _, desc in PLAN_CHECK_KEYS)
    return f"""You are a strict children's book editor checking a story plan before any page is written.
Return only JSON matching the schema. Answer false whenever you are in doubt. For every false answer, add one short,
specific issue that says exactly what to change.

STORY FACTS: theme "{theme}"; listener age {age}.

PAGE 1 (FIXED):
{page_one}

PLAN:
{plan_text(plan)}

CHECKS:
{checks}
"""


def evaluate_plan_check(result: Any) -> Dict[str, Any]:
    result = result if isinstance(result, dict) else {}
    failed_critical = [key for key, critical, _ in PLAN_CHECK_KEYS if critical and result.get(key) is not True]
    failed_other = [key for key, critical, _ in PLAN_CHECK_KEYS if not critical and result.get(key) is not True]
    return {
        "passed": not failed_critical,
        "failed_critical": failed_critical,
        "failed_other": failed_other,
        "issues": [str(i)[:240] for i in (result.get("issues") or [])][:6],
    }


def build_plan_revision_prompt(original_prompt: str, plan: dict, check: Dict[str, Any],
                               locked_first_beat: Optional[str] = None) -> str:
    descriptions = {key: desc for key, _, desc in PLAN_CHECK_KEYS}
    problems = [f"- {key}: {descriptions[key]}" for key in check["failed_critical"] + check["failed_other"]]
    problems += [f"- {issue}" for issue in check["issues"]]
    return (
        original_prompt
        + "\nREVISE THIS PLAN. It failed the editor's checks. Keep everything that works and Page 1 exactly; fix ONLY these problems:\n"
        + "\n".join(problems)
        + "\n\nPLAN TO REVISE:\n" + plan_text(plan) + "\n"
        + (("PAGE 2 IS ALREADY BEING WRITTEN from the first beat, so keep that beat exactly as it is and make every fix in the later beats:\n- "
            + str(locked_first_beat) + "\n") if locked_first_beat else "")
    )


# ===========================================================================
# Model routing (2026-11-10). Every Gemini call is labelled with a role, so
# each role can use its own model, set in Render's environment:
#   GEMINI_PAGES_MODEL    Pages 2+ of every story type (Bedtime, Story Worlds, ...)
#   GEMINI_PAGE1_MODEL    Page 1 (leave unset to keep the fast default)
#   GEMINI_PLANNER_MODEL  plan, plan check, Page 7 review, quality score
#   GEMINI_PAGES_THINKING_BUDGET  thinking tokens for Pages 2+ (default 0;
#                                 -1 = the model's own default)
# Unset -> GEMINI_MODEL, exactly as before. A role model that errors falls
# back to GEMINI_MODEL automatically.
# ===========================================================================
import contextvars as _contextvars

GEN_CTX: "_contextvars.ContextVar" = _contextvars.ContextVar("pillowtales_gen_ctx", default=None)
MODEL_ROUTING_BUILD = "model_routing=20261110"

_ROLE_ENV = {
    "pages": "GEMINI_PAGES_MODEL",
    "page1": "GEMINI_PAGE1_MODEL",
    "planner": "GEMINI_PLANNER_MODEL",
}


def role_model(role: Optional[str]) -> str:
    import os
    env = _ROLE_ENV.get(role or "")
    return (os.getenv(env, "") if env else "").strip()


def role_thinking_budget(role: Optional[str], requested: Optional[int]) -> Optional[int]:
    """Only the pages role can be overridden; everything else keeps what the caller asked for."""
    import os
    if role != "pages":
        return requested
    raw = os.getenv("GEMINI_PAGES_THINKING_BUDGET", "").strip()
    if not raw:
        return requested
    try:
        value = int(raw)
    except Exception:
        return requested
    return -1 if value < 0 else min(value, 8192)


def record_usage(role: Optional[str], model_name: str, response: Any, fallback_from: Optional[str] = None) -> None:
    """Add this call's model and token counts to the story's trace (best effort)."""
    try:
        ctx = GEN_CTX.get()
        request = ctx[1] if ctx else None
        if request is None:
            return
        meta = getattr(response, "usage_metadata", None)
        tokens_in = int(getattr(meta, "prompt_token_count", 0) or 0)
        tokens_out = int(getattr(meta, "candidates_token_count", 0) or 0)
        tokens_think = int(getattr(meta, "thoughts_token_count", 0) or 0)
        usage = get_trace(request).setdefault("usage", {})
        bucket = usage.setdefault(role or "other", {"model": model_name, "calls": 0, "in": 0, "out": 0, "thinking": 0})
        bucket["model"] = model_name
        bucket["calls"] += 1
        bucket["in"] += tokens_in
        bucket["out"] += tokens_out
        bucket["thinking"] += tokens_think
        if fallback_from:
            bucket["fallback_from"] = fallback_from
    except Exception:
        pass


# ===========================================================================
# Best-of-N Page 1 + draft-and-revise (2026-11-10)
#   BEDTIME_PAGE1_CANDIDATES  number of Page 1 candidates (default 3; 1 = off)
#   BEDTIME_DRAFT_REVISE      draft Pages 2-7 in one call, edit, revise (default on)
# Standard Bedtime, ages 3+, only while BEDTIME_PROMPTS_V5 is on.
# ===========================================================================
BEST_OF_BUILD = "page1_best_of=20261110 draft_revise=20261110"
PAGE1_EXTRA_WAIT_SECONDS = 3.0     # after the first candidate, wait this long for the others
PAGE1_JUDGE_TIMEOUT_SECONDS = 6
DRAFT_TIMEOUT_SECONDS = 45
EDITOR_TIMEOUT_SECONDS = 20
REWRITE_TIMEOUT_SECONDS = 35
MAX_REWRITE_PAGES = 3
EDITOR_THINKING_BUDGET = 1024

HOOK_TYPES: Dict[str, str] = {
    "funny_wish": "a funny or ambitious WISH the child badly wants to come true",
    "mistake": "a small, funny MISTAKE (by the child or a friend) that now has to be put right",
    "job": "a JOB or responsibility the child has been trusted with for the first time",
    "misunderstanding": "a MISUNDERSTANDING between characters that only the child notices",
    "rule_puzzle": "a strange RULE or PUZZLE in this world that the child must work out",
    "challenge": "a CONTEST, race or challenge the child wants to take part in",
    "promise": "a PROMISE the child made that has suddenly become hard to keep",
    "visitor": "an unexpected VISITOR who needs something that only the child can give or do",
    "found_object": "something unusual the child FINDS (only if it leads to a choice, not a simple return)",
}


def page1_candidate_count() -> int:
    import os
    try:
        return max(1, min(4, int(os.getenv("BEDTIME_PAGE1_CANDIDATES", "3"))))
    except Exception:
        return 3


def draft_revise_enabled() -> bool:
    return _env_on("BEDTIME_DRAFT_REVISE")


def choose_hooks(count: int, recent_hooks: Iterable[str] = ()) -> List[str]:
    import random as _random
    recent = list(recent_hooks or [])
    pool = [h for h in HOOK_TYPES if h != "found_object" and h not in recent[:4]]
    if len(pool) < count:
        pool = [h for h in HOOK_TYPES if h != "found_object"]
    _random.shuffle(pool)
    hooks = pool[:count]
    # The found-object hook is allowed only occasionally, and never twice in a row.
    if count >= 3 and "found_object" not in recent[:3] and _random.random() < 0.15:
        hooks[-1] = "found_object"
    return hooks


def hook_block(hook: str) -> str:
    return (
        "HOOK FOR THIS STORY (use this kind of hook, made specific to this child, theme and world):\n"
        f"- {HOOK_TYPES.get(hook, hook)}.\n"
        "- The hook must give the child something to WANT or DO, so that later the child, not a helper, can solve it.\n"
    )


def inject_hook(prompt: str, hook: str) -> str:
    marker = "PAGE 1 JOB:"
    block = hook_block(hook)
    return prompt.replace(marker, block + marker, 1) if marker in prompt else prompt + "\n" + block


PAGE1_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "best": {"type": "integer"},
        "reason": {"type": "string"},
        "story_promise": {"type": "string"},
        "childs_want": {"type": "string"},
        "stakes_seed": {"type": "string"},
        "spare_ideas": {"type": "array", "items": {"type": "string"}},
        "fresh_title": {"type": "string"},
    },
    "required": ["best", "reason", "story_promise", "childs_want", "stakes_seed", "spare_ideas"],
}


def _title_key(title: Any, child_name: Any = "") -> str:
    words = re.findall(r"[a-z]+", str(title or "").lower())
    drop = {"and", "the", "a", "an", "of", "s"} | set(re.findall(r"[a-z]+", str(child_name or "").lower()))
    return " ".join(w for w in words if w not in drop)


def title_flags(title: Any, recent_titles: Iterable[str], child_name: Any = "", moral: Any = "") -> List[str]:
    """Why a Page 1 title should be replaced: repeats a recent title, overused words, the moral word."""
    flags = []
    key = _title_key(title, child_name)
    if key and any(key == _title_key(t, child_name) for t in recent_titles or []):
        flags.append("repeats a recent title")
    flags += [f"overused word '{w}'" for w in overused_title_words_in(title)]
    moral_word = str(moral or "").strip().lower()
    if moral_word and moral_word in re.findall(r"[a-z]+", str(title or "").lower()):
        flags.append("contains the moral word")
    return flags

# Each hook implies its own natural climax; the plan is told which shape to aim for.
HOOK_CLIMAX_SHAPES: Dict[str, str] = {
    "funny_wish": "the wish comes true in an unexpected way because of something the child does, not by magic alone",
    "mistake": "the child puts the mistake right with their own idea, and owns up to it",
    "job": "the child completes the job in their own clever way when the usual way fails",
    "misunderstanding": "the child works out the truth and brings the characters together",
    "rule_puzzle": "the child works out the rule and uses it at the decisive moment",
    "challenge": "the child wins, or wins something better, through a clever or kind choice rather than strength",
    "promise": "the child keeps the promise even though it costs them something",
    "visitor": "the child gives the visitor what only they could, at some cost to themselves",
    "found_object": "the child makes a real choice about the object instead of simply returning it",
}


def opening_notes_block(notes: Optional[Dict[str, Any]], hook: Optional[str]) -> str:
    """Editor notes from the Page 1 judge, passed into the plan."""
    if not notes and not hook:
        return ""
    lines = ["OPENING NOTES FROM THE EDITOR (use them):"]
    if hook and hook in HOOK_CLIMAX_SHAPES:
        lines.append(f"- Hook type: {hook}. Aim for this climax shape on Page 6: {HOOK_CLIMAX_SHAPES[hook]}.")
    for key, label in (("story_promise", "What this opening promises"), ("childs_want", "What the child wants"), ("stakes_seed", "Seed of the stakes")):
        value = str((notes or {}).get(key) or "").strip()
        if value:
            lines.append(f"- {label}: {value}")
    spare = [str(i).strip() for i in ((notes or {}).get("spare_ideas") or []) if str(i).strip()][:3]
    if spare:
        lines.append("- Strong ideas from the openings not chosen; use one or two where they genuinely fit (never contradict Page 1): " + "; ".join(spare))
    return "\n".join(lines) + "\n"


def build_page1_judge_prompt(request: Any, candidates: List[Dict[str, Any]], age: int, age_standard: str = "") -> str:
    theme = getattr(request, "customTheme", None) or getattr(request, "theme", "")
    listing = "\n\n".join(
        f"CANDIDATE {i + 1}\nTitle: {c.get('title')}"
        + (f"  (TITLE PROBLEM: {'; '.join(c.get('_title_flags') or [])})" if c.get("_title_flags") else "")
        + f"\nPage 1: {(c.get('pages') or [''])[0]}"
        for i, c in enumerate(candidates)
    )
    return f"""You are choosing the opening page for a seven-page children's bedtime story (listener age {age}, theme "{theme}").
Pick the candidate most likely to become an outstanding story. Return only JSON matching the schema:
- best: the winning candidate number; reason: why, in one sentence.
- story_promise: what the winning Page 1 promises the listener, in one sentence.
- childs_want: what the child wants or must do, in one sentence.
- stakes_seed: the bedtime-safe loss that could happen if the child fails, in one sentence.
- spare_ideas: 1-3 short, specific ideas from the OTHER candidates worth reusing later (a funny detail, image or character habit), compatible with the winner.
- fresh_title: ONLY if the winner's title has a TITLE PROBLEM, a new title for it (up to 7 words, includes the child's name,
  specific to this Page 1, none of the overused words, not the moral). Otherwise "".

JUDGE BY:
- A clear, specific hook that makes a child want to hear Page 2.
- The child has something to want or do, so the child (not a helper) can solve it later.
- Room for real but bedtime-safe stakes (something could be lost, missed or spoiled).
- Freshness: an original premise, not a generic "found something, return it" set-up.
- The theme is literally present and matters.
- Language fits the age standard below and reads aloud beautifully.
- Avoid candidates with overused names (Barnaby, Pip, Sparkle...) or overused title words (Whispering, Secret, Lost...).
{("AGE STANDARD:" + chr(10) + age_standard.strip()) if age_standard else ""}

{listing}
"""


EDITOR_SCHEMA = {
    "type": "object",
    "properties": {
        "pages_to_rewrite": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"page": {"type": "integer"}, "problem": {"type": "string"}, "fix": {"type": "string"}},
                "required": ["page", "problem", "fix"],
            },
        },
        "overall_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["pages_to_rewrite", "overall_notes"],
}

EDITOR_CHECKLIST = """- Stakes: by Page 3 the listener SEES what will be lost, missed or spoiled, and it nearly happens at the setback.
- The child's own idea or action causes the solution on Page 6 (planted earlier); no helper, adult, magic or luck solves it.
- Page 7 is aftermath only: result, callback, calm bedtime landing. No new events, arrivals, rescues or explanations.
- Every question and thread is answered by Page 6; the hidden explanation is revealed on the page.
- Continuity: no contradictions; nobody knows a name or fact before learning it; objects and names stay consistent.
- Age fit: words and sentences match the age standard; replace words from older bands.
- The moral is shown through action, never stated, explained or lectured.
- No stock phrases (tummy flip-flops, sparkling eyes...), no repetition of the same image or idea, no crammed page.
- The theme and title promise are delivered."""


def build_editor_prompt(request: Any, title: str, pages: List[str], plan: Optional[dict], age: int,
                        moral_requested: bool, age_standard: str = "", final_page_number: Optional[int] = None) -> str:
    theme = getattr(request, "customTheme", None) or getattr(request, "theme", "")
    moral = getattr(request, "moral", "") if moral_requested else "none requested"
    story = "\n\n".join(f"Page {i + 1}: {p}" for i, p in enumerate(pages))
    partial = ""
    if final_page_number and len(pages) < final_page_number:
        partial = (f"NOTE: this draft stops at Page {len(pages)}; the story continues to Page {final_page_number} afterwards. "
                   "Do not judge the ending or ask for it to be wrapped up; judge only the pages shown.\n")
    return f"""You are the senior editor of a children's bedtime story app. Read this complete draft as a whole, the way a parent will hear it,
and decide which pages need rewriting before anyone hears them. Pages 1 and 2 are already published and FIXED; only Pages 3-{len(pages)} may be rewritten.
Return only JSON matching the schema. List at most {MAX_REWRITE_PAGES} pages, the ones that matter most. For each, name the specific problem and an
exact, actionable fix that keeps the rest of the story consistent. If the draft is already excellent, return an empty list. Do not rewrite for taste alone.
{partial}
STORY FACTS: title "{title}"; theme {theme}; private moral {moral}; listener age {age}.
PLAN THE STORY WAS WRITTEN FROM:
{plan_text(plan) if plan else "(none)"}

CHECKLIST:
{EDITOR_CHECKLIST}
{("AGE STANDARD:" + chr(10) + age_standard.strip()) if age_standard else ""}

DRAFT:
{story}
"""


REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "pages": {
            "type": "array",
            "items": {"type": "object", "properties": {"page": {"type": "integer"}, "text": {"type": "string"}}, "required": ["page", "text"]},
        }
    },
    "required": ["pages"],
}


def editor_rewrite_targets(result: Any, page_count: int) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for item in (result or {}).get("pages_to_rewrite") or []:
        try:
            page = int(item.get("page"))
        except Exception:
            continue
        if 3 <= page <= page_count and page not in {t["page"] for t in targets}:
            targets.append({"page": page, "problem": str(item.get("problem") or "")[:400], "fix": str(item.get("fix") or "")[:600]})
    return sorted(targets, key=lambda t: t["page"])[:MAX_REWRITE_PAGES]
