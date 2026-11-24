import asyncio, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
from app.models.story import StoryCharacter
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
E = make_request(childName="Emily", age=8)
J = make_request(childName="Jay", age=12)
real = {
 "Jay remembered his promise to his grandmother to plant every seed he found.": (J, True),
 "She dashed to her sewing basket for her mother’s brass kettle-warmer.": (E, True),
 "Emily pulled out her grandfather's brass counting-beads.": (E, True),
 "Remembering her grandmother's comforting hum, Emily began to hum.": (E, True),
 "Nana had given her a toy silver tea set on the dresser.": (E, True),
 "Tonight, her Uncle Jed, who studied the sky, had set up a telescope.": (E, True),
 "Emily's mum hummed a sunny market melody.": (E, True),
 "Grandpa said the jar was lucky.": (E, True),
 "The baby dragon's mother circled overhead.": (E, False),
 "The mother duck led her ducklings to the pond.": (E, False),
 "Emily lived near the cloud stables, where gentle dragons rested.": (E, False),
 "The grandfather clock ticked in the hall.": (E, False),
 "Old Nell, the largest copper drake, coughed.": (E, False),
 "The princess and her royal advisor whispered.": (E, False),
}
for text, (req, want) in real.items():
    got = Q.invented_family_mentions(text, req)
    ok(bool(got) == want, f"{'flag' if want else 'leave'}: {text[:60]!r} -> {got}")
with_gran = make_request(childName="Emily", age=8, characters=[StoryCharacter(name="Nana Rose", relationship="Grandmother")])
ok(Q.invented_family_mentions("Emily remembered her grandmother's hum.", with_gran) == [], "a grandmother the parent added is allowed")
ok(Q.invented_family_mentions("Emily's dad waved.", with_gran) != [], "...but other family is still flagged")
es = make_request(childName="Emily", age=8, storyLanguageCode="es")
ok(Q.invented_family_mentions("Emily's grandpa", es) == [], "non-English: no word check (the written rule still applies)")
# rule text reaches every stage
p1 = Q.hook_block("job"); ok("unless the parent added them" in p1, "Page 1 options told not to invent family")
ok("never invent the child's mum, dad, grandparents" in Q.v5_craft_block(True, "kindness"), "page-writing rules forbid invented family")
ok("Family:" in Q.EDITOR_CHECKLIST, "editor checks for invented family")
# plan check catches it
plan = {"beats": ["a", "b", "c", "d", "e", "f"], "stake_for_child": "Breaking her promise to Grandpa", "solution_key": "a", "stake_sign": "b", "signature_delight": ""}
iss = Q.structural_plan_issues(plan, E)
ok(any(k == "invented_family" for k, _ in iss), f"plan with 'promise to Grandpa' sent back for revision")
ok(not any(k == "invented_family" for k, _ in Q.structural_plan_issues(plan, make_request(childName="Emily", age=8, characters=[StoryCharacter(name="Bill", relationship="Grandad")]))), "...unless the parent added a grandparent")
# draft: an invented-family page is rewritten
os.environ["FAKE_DRAFT_FAMILY"] = "4"
s = open(os.path.join(os.path.dirname(__file__), "harness.py")).read()
class Repo(FakeRepo):
    def update(self, sid, uid, v): self.updates.append(json.loads(json.dumps(v, default=str))); return v
async def run():
    repo = Repo(); svc = StoryService(repo); req = make_request(childName="Aifric", age=8, theme="dragons", moral="patience", characters=[StoryCharacter(name="Maeve", relationship="Aunt")])
    d = await svc.generate_story_first_page(req, None)
    await svc.complete_story_background(request=req, user_id="u1", story_id="s1", title=d["title"], current_pages=d["pages"], companion=None, expected_pages=7)
    return req, repo
req, repo = asyncio.run(run()); dr = Q.get_trace(req)["draft_revise"]
fin = [u for u in repo.updates if u.get("generation_status") == "complete"][-1]["pages"]
ok(dr.get("invented_family_pages") == [4] and 4 in dr["rewritten_pages"] and "grandmother" not in fin[3], f"draft page with an invented grandmother is rewritten ({dr.get('invented_family_pages')}, {dr['rewritten_pages']})")
del os.environ["FAKE_DRAFT_FAMILY"]
ok(Q.invented_family_mentions('"My grandfather never left an ordinary trail," said Captain Coram.', E) == [], "a character's own family ('My grandfather', said Coram) is not flagged")
ok(Q.invented_family_mentions("The twin lanterns glowed.", E) == [] and Q.invented_family_mentions("their twin stars", E) == [], "'twin' things are not siblings")
p1 = "Emily promised her Grandpa she would watch the jar."
ok(Q.invented_family_mentions("Grandpa smiled at the jar.", E, already_told=p1) == [], "family already on Page 1 stays consistent on later pages")
ok(Q.invented_family_mentions("Her mum waved.", E, already_told=p1) == ["Her mum"], "...but new family is still flagged")
dad = make_request(childName="Emily", age=7, theme="custom", customTheme="Daddy stopped the pain in my stomach")
ok(Q.invented_family_mentions("Daddy came over. Her dad smiled and Daddy said hello.", dad) == [], "family named in the parent's own theme is allowed ('Daddy stopped the pain...')")
ok(Q.invented_family_mentions("Her grandmother waved.", dad) != [], "...other family is still flagged")
gran_moral = make_request(childName="Jay", age=9, customMoral="Help Granny")
ok(Q.invented_family_mentions("Jay remembered his granny's advice.", gran_moral) == [], "family named in a custom moral is allowed")
ok(Q.invented_family_mentions("Her mum smiled.", make_request(childName="Emily", age=7, theme="custom", customTheme="Mermaid momentum")) != [], "'momentum' in a theme does not allow mum")
