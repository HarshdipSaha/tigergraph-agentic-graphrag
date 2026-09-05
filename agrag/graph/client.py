from __future__ import annotations

from agrag.config import settings


def connect(graphname: str | None = None):
    """TigerGraph Savanna connection. Uses the graph secret; falls back to user/password for DDL."""
    from pyTigerGraph import TigerGraphConnection

    kwargs = dict(host=settings.tg_host, graphname=graphname or settings.tg_graph)
    if settings.tg_username:
        kwargs.update(username=settings.tg_username, password=settings.tg_password)
    if settings.tg_secret:
        kwargs.update(gsqlSecret=settings.tg_secret)
    conn = TigerGraphConnection(**kwargs)
    if settings.tg_secret:
        conn.getToken(settings.tg_secret)
    return conn
