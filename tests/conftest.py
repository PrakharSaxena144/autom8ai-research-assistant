"""Test configuration: no network, no API keys, no model downloads.

Environment variables are set before any `app.*` import so `app.config.settings` picks them up
(python-dotenv never overrides variables that already exist, even empty ones).
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp = tempfile.mkdtemp(prefix="ra-tests-")
os.environ.update({
    "GROQ_API_KEY": "",
    "TAVILY_API_KEY": "",
    "QDRANT_URL": "",
    "QDRANT_PATH": str(Path(_tmp) / "qdrant"),
    "MODEL_CACHE_DIR": str(Path(_tmp) / "models"),
    "SEED_ON_STARTUP": "false",
    "WEB_SEARCH_PROVIDER": "none",
    "MAX_SUB_QUESTIONS": "3",
    "MAX_KB_ATTEMPTS": "2",
    "MAX_REVISIONS": "1",
    "MAX_FOLLOWUP_ROUNDS": "1",
})

CORPUS = ROOT / "data" / "corpus"
