import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import *
svc = StoryService(FakeRepo()); req = make_request(childName="Emily")
svc._recent_story_history(req)
data = {"title": "Emily and the Whispering Map", "pages": ["Emily found a map."]}
svc.freshen_page_one_names(req, data)
print("PASS" if Q.get_trace(req).get("title_overused_words") == ["Whispering"] else "FAIL", "title words logged via route path:", Q.get_trace(req).get("title_overused_words"))
