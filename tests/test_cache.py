import torch

from vlm_driving.cache import AsyncTokenCache


def test_empty_cache_is_stale_and_has_no_tokens():
    cache = AsyncTokenCache(max_age_s=0.5)

    tokens, age_s, fresh = cache.read(now_s=10.0)

    assert tokens is None
    assert age_s == float("inf")
    assert fresh is False


def test_cache_reports_age_and_freshness_boundary():
    cache = AsyncTokenCache(max_age_s=0.5)
    source = torch.ones(2, 3, requires_grad=True)

    cache.update(source, timestamp_s=10.0)
    tokens, age_s, fresh = cache.read(now_s=10.5)

    assert tokens is not None
    assert tokens.requires_grad is False
    assert torch.equal(tokens, torch.ones(2, 3))
    assert age_s == 0.5
    assert fresh is True

    _, stale_age_s, stale = cache.read(now_s=10.51)
    assert stale_age_s > 0.5
    assert stale is False


def test_cache_clamps_negative_age_to_zero():
    cache = AsyncTokenCache(max_age_s=1.0)
    cache.update(torch.zeros(1), timestamp_s=5.0)

    _, age_s, fresh = cache.read(now_s=4.0)

    assert age_s == 0.0
    assert fresh is True
