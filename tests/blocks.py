import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
svc = StoryService(FakeRepo())
print("=== READABILITY 8"); print(svc._age_readability_block(8))
print("=== COGNITIVE LOAD 8"); print(svc._age_cognitive_load_block(8))
print("=== PAGE BUDGET age 3 / 8 / 11:", [svc._standard_bedtime_page_budget(a, 3) for a in (3, 8, 11)])
