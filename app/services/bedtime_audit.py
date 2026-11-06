"""Opt-in, non-blocking Bedtime Story generation audit (no child text stored)."""
from __future__ import annotations
import contextvars
import functools
import hashlib
import json
import logging
import os
import time
import uuid

logger = logging.getLogger("pillowtales.bedtime_audit")
_active = contextvars.ContextVar("bedtime_audit", default=None)
RULES = (
    "_children_author_voice_rules", "_first_page_spine_setup_rules",
    "_natural_name_pronoun_rules", "_standard_bedtime_first_page_quality_rules",
    "_natural_read_aloud_cadence_rules", "_first_page_language_style_block",
    "_bedtime_native_editorial_block", "_standard_bedtime_page_budget",
    "_story_spine_block", "_standard_bedtime_elite_quality_rules",
    "_first_page_age_prompt_rules", "_bedtime_story_flow_essentials",
    "_bedtime_narrative_progression_rules", "_bedtime_prose_polish_essentials",
    "_very_young_bedtime_page_role", "_very_young_bedtime_override",
    "_ending_engine_rules", "_storycraft_rules", "_story_flow_rules",
    "_narrative_progression_repetition_rules", "_literary_polish_rules",
    "_age_readability_block", "_age_vocabulary_block", "_age_quality_control_block",
)
def enabled():
    return os.getenv("PILLOWTALES_BEDTIME_AUDIT", "0") == "1"
def begin(request):
    if not enabled():
        return None
    try:
        return {"run_id": str(uuid.uuid4()), "started_at": time.time(), "story_id": None,
                "age": int(request.age), "theme": str(request.customTheme or request.theme or "")[:80],
                "moral": str(request.moral or "")[:80], "duration_min": request.durationMin,
                "language_code": str(request.storyLanguageCode or "")[:10],
                "events": [], "_rule_outputs": [], "status": "started"}
    except Exception:
        logger.exception("Failed to initialise bedtime audit")
        return None
def activate(trace):
    return _active.set(trace)
def deactivate(token):
    _active.reset(token)
def current():
    return _active.get()
def digest(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
def event(kind, **fields):
    trace = current()
    if trace is None:
        return
    try:
        if len(trace["events"]) < 1500:
            trace["events"].append({"event_type": kind, "metadata": fields,
                                     "elapsed_ms": round((time.time() - trace["started_at"]) * 1000)})
    except Exception:
        logger.exception("Bedtime audit event failed")
def _is_bedtime(self, request):
    return not self._is_canon_request(request) and not self._is_folk_adventure_request(request)
def install(service_cls):
    """Wrap methods without altering returned prompts or generation results."""
    for name in RULES:
        original = getattr(service_cls, name, None)
        if not callable(original):
            continue
        def wrap_rule(method, rule_id):
            @functools.wraps(method)
            def wrapped(self, *args, **kwargs):
                result = method(self, *args, **kwargs)
                if current() is not None:
                    current()["_rule_outputs"].append((rule_id, str(result)))
                    event("rule_evaluated", rule_id=rule_id, output_sha256=digest(result),
                          output_chars=len(str(result)), nonempty=bool(result))
                return result
            return wrapped
        setattr(service_cls, name, wrap_rule(original, name))
    seed_original = service_cls._select_opening_seed
    @functools.wraps(seed_original)
    def seed(self, request, *args, **kwargs):
        result = seed_original(self, request, *args, **kwargs)
        if current() is not None and _is_bedtime(self, request):
            event("opening_selected", family=result.get("family"),
                  seed_sha256=digest(result.get("sentence", "")))
        return result
    service_cls._select_opening_seed = seed
    for name in ("_build_first_page_prompt", "_build_remaining_pages_prompt"):
        original = getattr(service_cls, name)
        def wrap_prompt(method, builder):
            @functools.wraps(method)
            def wrapped(self, request, *args, **kwargs):
                trace = current()
                if trace is None or not _is_bedtime(self, request):
                    return method(self, request, *args, **kwargs)
                before = len(trace["events"])
                before_outputs = len(trace["_rule_outputs"])
                result = method(self, request, *args, **kwargs)
                rules = [e for e in trace["events"][before:] if e["event_type"] == "rule_evaluated"]
                # Check exact helper output inclusion without persisting the prompt or child names.
                # SHA alone cannot prove inclusion; use a temporary digest->output registry below.
                included = [{"rule_id": rule_id, "included_exact": bool(value) and value in result}
                            for rule_id, value in trace["_rule_outputs"][before_outputs:]]
                del trace["_rule_outputs"][before_outputs:]
                event("prompt_built", builder=builder, page_number=kwargs.get("next_page_number", 1),
                      rule_inclusion=included,
                      prompt_sha256=digest(result), prompt_chars=len(result),
                      invoked_rule_ids=[e["metadata"]["rule_id"] for e in rules],
                      not_invoked_rule_ids=[r for r in RULES if r not in {e["metadata"]["rule_id"] for e in rules}])
                return result
            return wrapped
        setattr(service_cls, name, wrap_prompt(original, name))
    validate_original = service_cls._validate_final_page
    @functools.wraps(validate_original)
    def validate(self, *args, **kwargs):
        result = validate_original(self, *args, **kwargs)
        if current() is not None:
            event("final_page_deterministic_validation", passed=bool(result[0]), reason=str(result[1])[:200])
        return result
    service_cls._validate_final_page = validate
    semantic_original = service_cls._review_final_page_semantics
    @functools.wraps(semantic_original)
    async def semantic(self, *args, **kwargs):
        result = await semantic_original(self, *args, **kwargs)
        if current() is not None:
            event("final_page_semantic_review", passed=bool(result[0]), reason=str(result[1])[:200])
        return result
    service_cls._review_final_page_semantics = semantic

def persist(trace, client):
    """Call from asyncio.to_thread, after story ID exists. Never fail story generation."""
    if trace is None or not trace.get("story_id"):
        return
    try:
        client.table("bedtime_generation_audits").upsert({
            "run_id": trace["run_id"], "story_id": trace["story_id"],
            "age": trace["age"], "theme": trace["theme"], "moral": trace["moral"],
            "duration_min": trace["duration_min"], "language_code": trace["language_code"],
            "generation_path": "page_1_first_background", "final_status": trace["status"],
            "first_page_ready_ms": trace.get("first_page_ready_ms"),
            "backend_commit": os.getenv("RENDER_GIT_COMMIT", "")[:80],
        }).execute()
        events = [{"run_id": trace["run_id"], "event_type": e["event_type"],
                   "page_number": e["metadata"].get("page_number"),
                   "rule_id": e["metadata"].get("rule_id"),
                   "metadata": {**e["metadata"], "elapsed_ms": e["elapsed_ms"]}}
                  for e in trace["events"]]
        for offset in range(0, len(events), 100):
            client.table("bedtime_generation_events").insert(events[offset:offset+100]).execute()
        logger.info("BEDTIME_AUDIT_SAVED run_id=%s story_id=%s events=%d",
                    trace["run_id"], trace["story_id"], len(events))
    except Exception:
        logger.exception("Bedtime audit persistence failed; story generation unaffected")
