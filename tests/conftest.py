import os
import time
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = REPO_ROOT / "AGENT_EXECUTION_LOG.md"

def pytest_sessionstart(session):
    """Verify that AGENT_EXECUTION_LOG.md exists, is non-empty, and recently updated."""
    if not LOG_FILE.exists():
        pytest.fail(f"MANDATORY AUDIT RULE VIOLATION: {LOG_FILE} does not exist.")
    
    content = LOG_FILE.read_text(encoding="utf-8").strip()
    if len(content) < 50:
        pytest.fail(f"MANDATORY AUDIT RULE VIOLATION: {LOG_FILE} is empty or insufficient.")
    
    # Check that file was modified within the last 24 hours (current working session)
    mtime = LOG_FILE.stat().st_mtime
    age_seconds = time.time() - mtime
    if age_seconds > 86400:
        pytest.fail(f"MANDATORY AUDIT RULE VIOLATION: {LOG_FILE} has not been updated in this session.")
