import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
import app.services.story_service as ss
svc = StoryService(FakeRepo())
themes = ["dragons","princess","animals","space","adventure","dinosaurs","underwater","magic","forest"]
ages = [2,4,6,8,10]
print("theme      " + " ".join(f"age{a:>2}" for a in ages))
for t in themes:
    allowed,_ = Q.resolve_theme_seed_families(t, None)
    row=[]
    for a in ages:
        pool=[s["family"] for s in svc._seed_pool_for_age(a) if s["family"] in allowed]
        row.append(f"{len(pool):>5}")
    print(f"{t:<10} " + " ".join(row))
print("dragons age 8:", [s["family"] for s in svc._seed_pool_for_age(8) if s["family"] in Q.THEME_SEED_ALLOWLIST["dragons"]])
print("dragons age 10:", [s["family"] for s in svc._seed_pool_for_age(10) if s["family"] in Q.THEME_SEED_ALLOWLIST["dragons"]])
