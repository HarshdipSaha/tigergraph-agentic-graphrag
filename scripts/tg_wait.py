"""Poll TigerGraph until the workspace responds, or give up after a timeout.

Savanna free-tier workspaces auto-suspend after ~60 minutes with no database traffic (see
docs/status.md "Real infra incidents"). A suspended workspace returns 500/502 on the token endpoint
and needs a manual resume from the Savanna dashboard before any pyTigerGraph call will succeed again
(usually 1-3 minutes to fully wake once resumed). This script polls tg_smoke-style so a build or eval
script can wait for that resume instead of failing immediately.

Usage: python scripts/tg_wait.py [max_minutes]   (default 15)
"""
import sys
import time

from agrag.graph.client import connect

max_minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
deadline = time.time() + max_minutes * 60

attempt = 0
while time.time() < deadline:
    attempt += 1
    try:
        conn = connect()
        print(f"attempt {attempt}: ready (version {conn.getVer()})")
        sys.exit(0)
    except Exception as e:
        print(f"attempt {attempt}: down ({type(e).__name__})")
        time.sleep(20)

print(f"gave up after {max_minutes} minutes; resume the workspace in the Savanna dashboard and retry")
sys.exit(1)
