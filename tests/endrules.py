import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
svc = StoryService(FakeRepo()); req = make_request(childName="Emily", age=8, theme="princess", moral="sharing")
e = svc._ending_engine_rules()
print("ENDING ENGINE words:", len(e.split())); print(e)
import inspect
src = inspect.getsource(svc._page_narrative_role)
print("PAGE ROLE fn has page 6/7 text:", "PAGE 7" in src)
for n in (6,7):
    try: print(svc._page_narrative_role(n, 7))
    except TypeError: print(svc._page_narrative_role(request=req, page_number=n, total_pages=7) if 'request' in inspect.signature(svc._page_narrative_role).parameters else inspect.signature(svc._page_narrative_role))
