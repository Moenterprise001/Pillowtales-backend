import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
import app.services.story_service as ss
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
sig = Q.seed_text_signatures(ss.OPENING_SEED_FAMILIES)
vals = list(sig.values())
ok(len(sig) == len(ss.OPENING_SEED_FAMILIES) and len(set(vals)) == len(vals), f"every opening has a unique signature ({len(sig)}/{len(ss.OPENING_SEED_FAMILIES)})")
real = [
 "Emily went to a tiny woodland school where squirrels rang the bell with acorns, and sleepy owls taught lessons beneath the quiet trees.",
 "Emily went to a tiny woodland school where squirrels rang the bell with acorns and owls taught lessons beneath the quiet trees.",
 "Emily lived right next to a mountain flying school, where young dragons learned to fly.",
 "Emily spent her days in the coziest mountain market, where gentle dragons warmed cups of cocoa with their breath.",
]
rows = [{"title": f"t{i}", "characters": [], "generation_trace": None, "first_page": p} for i, p in enumerate(real)]
h = Q.load_recent_history(FakeRepo(rows), "u1", "Emily", seed_signatures=sig)
ok("forest_school" in h["families"], f"old woodland-school stories detected from Page 1 text -> {sorted(h['families'])}")
ok("dragon_flying_school" in h["families"], "verbatim flying-school opening detected")
print("   (reworded openings like 'the coziest mountain market' are only caught when traced:", "dragon_market" in h["families"], ")")
# name list: overused names survive a long history
hist = {"names": [f"Name{i}" for i in range(60)]}
names = Q.avoid_names_for(make_request(childName="Emily"), hist, None)
ok(all(n in names for n in ["Barnaby", "Pip", "Sparkle", "Squeaky"]), "overused names kept even with 60 recent names")
ok(len(names) == 40, f"list capped at 40 ({len(names)})")
# selection now skips forest_school for Emily
svc = StoryService(FakeRepo(rows))
fams = set()
for _ in range(40):
    r = make_request(childName="Emily", theme="animals"); fams.add(svc._select_opening_seed(r)["family"])
ok("forest_school" not in fams, f"animals story skips forest_school -> {sorted(fams)}")
