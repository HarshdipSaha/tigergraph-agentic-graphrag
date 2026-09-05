"""Run schema.gsql (--schema) or queries.gsql (--queries) against the workspace.

  python scripts/tg_setup.py --schema    # needs TG_USERNAME/TG_PASSWORD; creates graph OlympicsRAG
  python scripts/tg_setup.py --queries   # needs TG_SECRET for OlympicsRAG; installs all queries
"""
import argparse
from pathlib import Path

from agrag.graph.client import connect

GRAPH_DIR = Path(__file__).resolve().parents[1] / "agrag" / "graph"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--queries", action="store_true")
    args = ap.parse_args()
    if args.schema:
        conn = connect(graphname="")  # DDL is global
        print(conn.gsql((GRAPH_DIR / "schema.gsql").read_text(encoding="utf-8")))
        print("Next: create a secret for graph OlympicsRAG in the Savanna UI, set TG_SECRET in .env, then --queries")
    if args.queries:
        conn = connect()
        print(conn.gsql((GRAPH_DIR / "queries.gsql").read_text(encoding="utf-8")))
