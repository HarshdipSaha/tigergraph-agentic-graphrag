import os

import pytest
from dotenv import load_dotenv

load_dotenv()  # populate os.environ from .env before the skip check below reads it

pytestmark = pytest.mark.integration
if not os.getenv("TG_HOST"):
    pytest.skip("TG_HOST not set", allow_module_level=True)

from agrag.embed import FakeEmbedder
from agrag.graph.tg_backend import TigerGraphBackend


@pytest.fixture(scope="module")
def tg():
    return TigerGraphBackend.from_settings(embedder=FakeEmbedder(dim=384))


def test_structural_scan_matches_brief(tg):
    evs = tg.events_by_sport_games("biathlon", "2018 Winter")
    assert len(evs) == 11                       # docs/hackathon-brief.md §7 recomputation
    assert sum(1 for e in evs if (e.competitors or -1) > 73) == 5


def test_lookup_and_prev(tg):
    ev = tg.event_by_title("Athletics at the 2016 Summer Olympics – Men's 20 kilometres walk")
    assert ev is not None and ev.nations == 40
    assert tg.prev_event(ev.doc_id).gold == "Chen Ding"


def test_vector_search_returns_k(tg):
    hits = tg.vector_search(FakeEmbedder(dim=384).embed(["biathlon sprint"])[0], k=5)
    assert len(hits) == 5 and all(h.text for h in hits)
