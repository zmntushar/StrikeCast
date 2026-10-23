"""Test setup: isolated SQLite database, no scheduler, no network."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TMP = tempfile.mkdtemp(prefix="strikecast_test_")
os.environ["STRIKECAST_DATABASE_URL"] = f"sqlite:///{Path(_TMP, 'test.db').as_posix()}"
os.environ["STRIKECAST_SCHEDULER_ENABLED"] = "false"
os.environ["STRIKECAST_HTTP_MAX_RETRIES"] = "1"
