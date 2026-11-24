import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m)
cases = {"The tiny dragon": "", "The Young Unicorn": "", "The Baker": "", "Sun-Drake chick": "", "The Princess": "", "Princess": "",
         "Princess Lily": "Lily", "Captain Thistle": "Thistle", "Old Nell": "Nell", "Elder Roka": "Roka", "Pipit": "Pipit", "Name12": "Name12", "Glimmer": "Glimmer"}
for raw, want in cases.items():
    ok(Q.clean_character_name(raw) == want, f"{raw!r} -> {Q.clean_character_name(raw)!r}")
r = make_request(childName="Emily", theme="space")
banned = Q.avoid_names_for(r, {"names": list(cases)}, None)
ok(not any(n[:1].islower() for n in banned) and "Unicorn" not in banned and "tiny" not in banned, "no ordinary words on the avoid list")
ok(Q.banned_names_in_text("Emily lived in a tiny village", ["tiny", "Barnaby"]) == [], "lowercase word never treated as a name")
t, p, m = Q.freshen_page_one_names("Emily and the Star Chart", "Emily lived in a tiny village.", ["tiny"])
ok(p == "Emily lived in a tiny village." and not m, "Page 1 'tiny village' untouched")
ok(Q.banned_names_in_text("Barnaby waved", ["Barnaby"]) == ["Barnaby"], "real banned names still caught")
