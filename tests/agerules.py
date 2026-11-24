import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
svc = StoryService(FakeRepo())
for a in (3,5,7,9,11): print(f"=== age {a}"); print(svc._first_page_age_prompt_rules(a))
