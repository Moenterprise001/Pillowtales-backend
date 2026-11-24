import sys, types, importlib.abc, importlib.machinery, os
ROOT = sys.argv[1]; sys.path.insert(0, ROOT)
STUB_PREFIXES = ("fastapi", "google", "supabase", "jose", "jwt", "dotenv", "stripe", "elevenlabs", "openai", "gotrue", "postgrest", "storage3", "uvicorn", "httpx", "requests", "boto3", "azure", "pydub", "mutagen", "multipart", "slowapi", "sentry_sdk", "resend", "firebase_admin", "PIL")
class Dummy:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k: return a[0]
        return Dummy()
    def __getattr__(self, n): return Dummy()
    def __iter__(self): return iter(())
    def __mro_entries__(self, bases): return (object,)
class StubMod(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        if n[:1].isupper():
            return type(n, (Exception if "Error" in n or "Exception" in n else object,), {"__init__": lambda s, *a, **k: None, "__getattr__": lambda s, x: Dummy(), "__call__": lambda s, *a, **k: (a[0] if len(a)==1 and callable(a[0]) and not k else Dummy())})
        return Dummy()
class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in STUB_PREFIXES:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec): m = StubMod(spec.name); m.__path__ = []; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, Finder())
os.environ.setdefault("GEMINI_API_KEY", "x"); os.environ.setdefault("SUPABASE_URL", "http://x"); os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "x")
import app.main
print("IMPORT OK: app.main")
