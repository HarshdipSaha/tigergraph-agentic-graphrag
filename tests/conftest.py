from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def mini_corpus_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "mini_corpus.jsonl"


@pytest.fixture(scope="session")
def mini_public_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "mini_public.jsonl"
