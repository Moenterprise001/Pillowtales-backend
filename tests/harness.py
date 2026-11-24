"""Offline harness: stubs fastapi + google.genai, fakes Gemini and Supabase."""
import asyncio, json, os, re, sys, types as pytypes

ROOT = sys.argv[1]
sys.path.insert(0, ROOT)

# ---- stubs ---------------------------------------------------------------
fastapi = pytypes.ModuleType("fastapi")
class HTTPException(Exception):
    def __init__(self, status_code=500, detail=""):
        super().__init__(detail); self.status_code = status_code; self.detail = detail
fastapi.HTTPException = HTTPException
class _Router:
    def __init__(self, *a, **k): pass
    def _d(self, *a, **k): return lambda f: f
    post = get = put = delete = _d
fastapi.APIRouter = _Router
fastapi.Depends = lambda *a, **k: None
sys.modules["fastapi"] = fastapi

google = pytypes.ModuleType("google"); genai = pytypes.ModuleType("google.genai")
gtypes = pytypes.ModuleType("google.genai.types")
class _Any:
    def __init__(self, *a, **k): self.kw = k
gtypes.GenerateContentConfig = _Any; gtypes.ThinkingConfig = _Any
genai.types = gtypes; genai.Client = _Any; google.genai = genai
sys.modules.update({"google": google, "google.genai": genai, "google.genai.types": gtypes})

supa = pytypes.ModuleType("supabase"); supa.Client = object; sys.modules["supabase"] = supa
os.environ["GEMINI_API_KEY"] = "test"

from app.models.story import GenerateStoryRequest
from app.services import story_quality as Q
Q._ALLOW_OLD_VERSIONS = True  # tests only: lets the parity tests reach the old paths (servers are locked)
from app.services.story_service import StoryService

AIFRIC = json.load(open(os.path.join(os.path.dirname(__file__), "aifric.json")))

class Resp:
    def __init__(self, text): self.text = text; self.candidates = []

CALLS = []
THINKING = []
def fake_generate(self, prompt, response_schema=None, max_output_tokens=None, thinking_budget=None):
    if prompt.startswith("You are the story editor"): THINKING.append(thinking_budget)
    kind = "other"
    if prompt.startswith("You are choosing the opening page"):
        import time as _t
        if os.environ.get("FAKE_JUDGE_SLOW"): _t.sleep(float(os.environ["FAKE_JUDGE_SLOW"]))
        out = {"best": 2, "reason": "strongest hook", "story_promise": "What hatches?", "childs_want": "to hatch the egg",
               "stakes_seed": "the egg may go cold", "spare_ideas": ["a dragon who hiccups bubbles"]}
        if os.environ.get("FAKE_FRESH_TITLE"): out["fresh_title"] = os.environ["FAKE_FRESH_TITLE"]
        CALLS.append(("page1_judge", prompt)); return Resp(json.dumps(out))
    if prompt.startswith("You are the senior editor"):
        targets = [{"page": 4, "problem": "slow middle", "fix": "make her choose"}, {"page": 7, "problem": "crammed", "fix": "settle"}]
        out = {"pages_to_rewrite": [] if os.environ.get("FAKE_EDITOR_CLEAN") else targets, "overall_notes": ["good draft"]}
        CALLS.append(("editor", prompt)); return Resp(json.dumps(out))
    if prompt.startswith("Revise specific pages"):
        import re as _re
        nums = [int(n) for n in _re.findall(r"^PAGE (\d+):", prompt, flags=_re.M)]
        out = {"pages": [{"page": n, "text": AIFRIC[n - 1].replace("Aifric", "Aifric", 1) + " She grinned at the result."} for n in nums]}
        if os.environ.get("FAKE_REWRITE_LEAK"): out = {"pages": [{"page": n, "text": AIFRIC[n - 1] + " Such patience."} for n in nums]}
        CALLS.append(("rewrite", prompt)); return Resp(json.dumps(out))
    if prompt.startswith("You are a strict children's book editor"):
        if os.environ.get("FAKE_CHECK_SLOW"): import time as _t2; _t2.sleep(float(os.environ["FAKE_CHECK_SLOW"]))
        fail = bool(os.environ.get("FAKE_PLAN_CHECK_FAIL"))
        out = {k: True for k, _, _ in (Q.plan_check_keys() if hasattr(Q, "plan_check_keys") else Q.PLAN_CHECK_KEYS)}
        out["issues"] = []
        if fail:
            out["child_causes_solution"] = False; out["issues"] = ["The princess heals the wing on Page 7."]
        CALLS.append(("plan_check", prompt)); return Resp(json.dumps(out))
    if prompt.startswith("You are the story editor") and "REVISE THIS PLAN" in prompt:
        out = {"opening_question": "What will hatch from the purple egg?",
               "answer_to_opening_question": "A tiny Dream Dragon hatches for Aifric.",
               "hidden_explanation": "Dream Dragon eggs hatch only when they feel safe.",
               "stakes": "The egg grows cooler each hour.",
               "beats": [f"revised beat {i}" for i in range(2, 8)],
               "setback": "Her words stop working.", "childs_decisive_act": "Aifric gives up her own blanket.",
               "callback_detail": "the cocoa-warming dragons", "final_image": "The dragon asleep on her pillow.",
               "threads_to_close": ["who the seller is"]}
        CALLS.append(("plan_revision", prompt)); return Resp(json.dumps(out))
    if prompt.startswith("You are the story editor"):
        kind = "plan"
        out = {"opening_question": "What will hatch from the purple egg?",
               "answer_to_opening_question": "A tiny Dream Dragon hatches for Aifric.",
               "hidden_explanation": "Dream Dragon eggs hatch only when they feel safe and cared for.",
               "stakes": "If the egg is not cared for it will go cold and never hatch.",
               "beats": [f"beat {i}" for i in range(2, 8)] if "REJECTED" in prompt else ["Barnaby the badger appears"] + [f"beat {i}" for i in range(3, 8)],
               **({} if (os.environ.get("FAKE_PLAN_NO_V5") or not Q.prompts_v5_enabled()) else {"solution_key": "blanket", "solution_planted_page": 3, "stake_for_child": "the dragon never hatches",
                   "stake_sign": "the queue of waiting guests grows longer", "signature_delight": "Wobbleton's catchphrase",
                   "plot_shape": (re.search(r"^  - (\w+): ", prompt, flags=re.M) or [None, "escalating_tries"])[1] if "PLOT SHAPE" in prompt else "escalating_tries",
                   "child_flaw": "rushes", "comic_trait": "a snail who narrates his own speed"}),
               "setback": "Her usual words stop working.",
               "childs_decisive_act": "Aifric builds a nest from her own blanket.",
               "callback_detail": "the cocoa-warming dragons",
               "final_image": "The dragon asleep on her pillow.",
               "threads_to_close": ["who the seller is"]}
        if out.get("solution_key"):
            b = out["beats"]
            b[0] += "; the queue of waiting guests grows longer and Wobbleton's catchphrase rings out"; b[1] += "; Aifric's blanket is by the fire"
            b[2] += "; the queue of guests grows longer"; b[4] += "; Aifric wraps it in her blanket"; b[5] += "; Wobbleton's catchphrase again"
    elif prompt.startswith("You are a demanding children's book editor"):
        kind = "score"
        out = {k: 8.5 for k, _ in (Q.score_criteria() if hasattr(Q, "score_criteria") else Q.SCORE_CRITERIA)}; out["weakest_page"] = 6; out["top_issues"] = ["moral stated on page 6"]
    elif prompt.startswith("Review the ending"):
        kind = "review"
        if os.environ.get("FAKE_P7_FAIL_ONCE") and not getattr(fake_generate, "_p7_failed", False):
            fake_generate._p7_failed = True
            props = (response_schema or {}).get("properties", {})
            out = {k: (False if k == "resolves_opening_promise" else True) for k in props if props[k].get("type") == "boolean"}
            out.update({"reason": "not resolved", "required_changes": ["resolve it"]})
            CALLS.append((kind, prompt)); return Resp(json.dumps(out))
        props = (response_schema or {}).get("properties", {})
        out = {k: True for k in props if props[k].get("type") == "boolean"}
        out.update({"reason": "ok", "required_changes": []})
    elif prompt.startswith("Analyze this bedtime story"):
        kind = "metadata"; out = {"summary": "s", "characters": [], "setting": "x"}
    elif "Write Page 1 only" in prompt:
        kind = "page1"
        import re as _re
        hk = _re.search(r"HOOK FOR THIS STORY.*?\n- (?:a |an |something )?(\w+)", prompt, flags=_re.S)
        out = {"title": "Aifric and the Dream Dragon" + (f" ({hk.group(1)})" if hk else ""), "pages": [AIFRIC[0]]}
        if os.environ.get("FAKE_P1_TITLE"): out["title"] = os.environ["FAKE_P1_TITLE"]
        if "OUTLINE (private" in prompt and not os.environ.get("FAKE_NO_OUTLINE"):  # v8: outline in the same call
            sh = _re.search(r"^PLOT SHAPE FOR THIS STORY: (\w+)", prompt, flags=_re.M)
            out["outline"] = {"big_idea": "a dragon egg that hums whenever someone tells a joke", "want": "to hatch the humming egg before the bakery opens", "solution_object": "the warm bread oven",
                              "solution_seen_on_page": 1, "big_moment": "the egg rolls toward the cliff edge",
                              "ending": "Aifric tucks the egg by the oven and it hatches", "time_of_day": "early morning",
                              "plot_shape": sh.group(1) if sh else "escalating_tries"}
    else:
        kind = "continuation"
        if os.environ.get("FAKE_PAGE_SLOW"): import time as _t3; _t3.sleep(float(os.environ["FAKE_PAGE_SLOW"]))
        m = re.search(r"exactly (\d+) strings", prompt)
        n = int(m.group(1)) if m else 1
        start = len(re.findall(r"^Page \d+:", prompt, flags=re.M))
        nxt = re.search(r"Page (\d+) MUST deliver", prompt)
        page_no = int(nxt.group(1)) if nxt else None
        if page_no is None:
            pm = re.search(r"For Page (\d+), aim for", prompt); page_no = int(pm.group(1)) if pm else 2
        out = {"pages": AIFRIC[page_no - 1: page_no - 1 + n]}
        m_len = re.search(r"aim for (\d+)-(\d+) words", prompt)
        if m_len and Q.prompts_v5_enabled():  # like the real model, write to the requested length
            cap = int(m_len.group(2)); trimmed = []
            for pg in out["pages"]:
                words = pg.split()
                if len(words) > cap:
                    cut = " ".join(words[:cap]); dot = cut.rfind(".")
                    pg = cut[:dot + 1] if dot > len(cut) // 2 else cut.rstrip(",;:") + "."
                trimmed.append(pg)
            out = {"pages": trimmed}
        if n > 1 and os.environ.get("FAKE_DRAFT_FAIL"): out = {"pages": AIFRIC[page_no - 1: page_no + 1]}
        if n > 1 and os.environ.get("FAKE_DRAFT_EMPTY"): out = {"pages": []}
        if n > 1 and os.environ.get("FAKE_DRAFT_SHORT"): out = {"pages": AIFRIC[page_no - 1: page_no - 1 + n - 1]}
        if n > 1 and os.environ.get("FAKE_DRAFT_FAMILY"):
            pg = list(out["pages"]); i = int(os.environ["FAKE_DRAFT_FAMILY"]) - page_no
            if 0 <= i < len(pg): pg[i] = pg[i] + " She remembered her grandmother's lullaby."
            out = {"pages": pg}
        if n > 1 and os.environ.get("FAKE_DRAFT_LEAK"):
            pg = list(out["pages"]); i = int(os.environ["FAKE_DRAFT_LEAK"]) - page_no
            if 0 <= i < len(pg): pg[i] = pg[i] + " It was a lesson in patience."
            out = {"pages": pg}
    CALLS.append((kind, prompt))
    return Resp(json.dumps(out))

StoryService._generate_content_sync = fake_generate

class FakeQuery:
    def __init__(self, rows, cols): self.rows, self.cols = rows, cols
    def __getattr__(self, name): return lambda *a, **k: self
    def execute(self):
        if "generation_trace" in self.cols and os.environ.get("NO_TRACE_COL"):
            raise Exception("column stories.generation_trace does not exist")
        return pytypes.SimpleNamespace(data=self.rows)
class FakeClient:
    def __init__(self, rows, memory=None): self.rows = rows; self.memory = memory if memory is not None else []
    def table(self, name):
        rows = self.memory if name == "story_memory" else self.rows
        client = self
        class T:
            def select(_s, cols):
                if name == "story_memory" and os.environ.get("NO_MEMORY_TABLE"):
                    raise Exception('relation "public.story_memory" does not exist')
                return FakeQuery(sorted(rows, key=lambda r: str(r.get("created_at") or ""), reverse=True), cols)
            def insert(_s, row):
                if name == "story_memory" and os.environ.get("NO_MEMORY_TABLE"):
                    raise Exception('relation "public.story_memory" does not exist')
                client.memory.append(dict(row, created_at=row.get("created_at") or "2026-11-10T23:59:00"))
                return pytypes.SimpleNamespace(execute=lambda: pytypes.SimpleNamespace(data=[row]))
        return T()
class FakeRepo:
    def __init__(self, rows=None, memory=None): self.client = FakeClient(rows or [], memory); self.updates = []
    def update(self, story_id, user_id, values):
        if "generation_trace" in values and os.environ.get("NO_TRACE_COL"):
            raise Exception("column generation_trace does not exist")
        self.updates.append(values); return values

def make_request(**kw):
    base = dict(userId="u1", childName="Aifric", age=9, theme="dragons", moral="kindness", calmLevel="calm")
    base.update(kw); return GenerateStoryRequest(**base)
