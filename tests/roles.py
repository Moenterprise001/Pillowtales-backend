import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
svc = StoryService(FakeRepo())
for n in (2,4,5,6,7): print(svc._page_narrative_role(n)); print()
