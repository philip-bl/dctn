import torch

from dctn.contraction_path_cache import ContractionPathCache


def test_same_results():
    cache = ContractionPathCache()
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    ab0 = cache.contract("ij,jk->ijk", a, b)
    ab1 = cache.contract("ij,jk->ijk", a, b)

    ab2 = cache.contract(a, "ij", b, "jk", "ijk")
    ab3 = cache.contract(a, "ij", b, "jk", "ijk")

    ab4 = cache.contract(a, (0, 1), b, (1, 2), (0, 1, 2))
    ab5 = cache.contract(a, (0, 1), b, (1, 2), (0, 1, 2))

    for ab in (ab1, ab2, ab2, ab4, ab5):
        assert torch.all(ab == ab0)


def test_same_cache():
    cache1 = ContractionPathCache()
    cache2 = ContractionPathCache()
    assert cache1 is cache2
