import numpy as np

from agrag.embed import FakeEmbedder


def test_fake_embedder_is_deterministic_and_normalized():
    e = FakeEmbedder(dim=16)
    a = e.embed(["hello", "world"])
    b = e.embed(["hello", "world"])
    assert a.shape == (2, 16)
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert not np.allclose(a[0], a[1])
