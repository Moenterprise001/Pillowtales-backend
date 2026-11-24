import sys, os, re, asyncio
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, '/tmp/claude-0/-home-claude/682f5477-d3a8-54db-b33f-3f249283615f')
from harness import *
import app.services.story_service as ss
sys.path.insert(0, '/tmp/claude-0/-home-claude/682f5477-d3a8-54db-b33f-3f249283615f/scratchpad')
from new_seeds import NEW
fams = {s["family"]: s for s in ss.OPENING_SEED_FAMILIES}
bad = []
for fam, bands, langs in NEW:
    seed = fams[fam]
    for lang in ["en","es","fr","it","de"]:
        txt = seed.get(lang, "")
        if txt.count("{childName}") != 1 or re.search(r"\{(?!childName\})", txt): bad.append((fam, lang, "placeholder"))
        if Q.banned_names_in_text(txt, Q.OVERUSED_DEFAULT_NAMES): bad.append((fam, lang, "banned name"))
    for band in bands:
        if fam not in ss.AGE_SEED_FAMILY_ALLOWLIST[band]: bad.append((fam, band, "missing band"))
print("seed problems:", bad or "none")
print("total openings:", len(ss.OPENING_SEED_FAMILIES), "| unique families:", len(fams))
# render a non-English local fallback with a new seed (fallback uses the seed text verbatim)
svc = StoryService(FakeRepo())
for lang in ["es","de"]:
    req = make_request(age=8, theme="dragons", storyLanguageCode=lang, childName="Lucía")
    seed = svc._select_opening_seed(req)
    print(lang, seed["family"], "->", seed["sentence"][:90])
