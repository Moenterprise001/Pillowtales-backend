import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
svc = StoryService(FakeRepo())
raw = 'Emily knelt down.\\n\\n"It is all right," she said.\\nGlimmer peeked out.'
out = svc._sanitize_generated_page_text(raw)
print(("PASS" if "\\n" not in out and "\n\n" in out else "FAIL"), "escaped newlines become real breaks:", repr(out))
real = "One.\n\nTwo."
print(("PASS" if svc._sanitize_generated_page_text(real) == real else "FAIL"), "real paragraph breaks untouched")
r = make_request(age=8); p1 = svc._build_remaining_pages_prompt_v5 if hasattr(svc, "_build_remaining_pages_prompt_v5") else None
print(("PASS" if "Nobody knows a new character's name" in Q.v5_craft_block(True, "kindness") else "FAIL"), "name-before-introduction rule in craft block")
for raw, want in [("parade day was tomorrow.", "Parade day was tomorrow."), ("“wait,” she said.", "“Wait,” she said."), ("Emily ran.", "Emily ran."), ("¿dónde está?", "¿Dónde está?")]:
    got = svc._sanitize_generated_page_text(raw)
    print(("PASS" if got == want else "FAIL"), "page start capitalised:", repr(got))
import asyncio, json
class Repo(FakeRepo):
    def update(self, sid, uid, v): self.updates.append(v); return v
async def go():
    repo = Repo(); s2 = StoryService(repo); req = make_request(childName="Emily", age=8, theme="dragons", moral="patience")
    d = await s2.generate_story_first_page(req, None)
    await s2.complete_story_background(request=req, user_id="u1", story_id="s-time", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return Q.get_trace(req)
tr = asyncio.run(go())
print(("PASS" if set(tr.get("page_shown_s", {})) >= {"2", "7"} else "FAIL"), "page-shown timings saved:", tr.get("page_shown_s"))
