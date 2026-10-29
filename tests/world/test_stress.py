import pytest

from altiseek.world.stress import stress_pair

from .test_sampler import CFG, LIBRARY


def test_stress_pair_preserves_targets_and_background():
    low, high = stress_pair(5000000, ("sharp", "group"), CFG, LIBRARY)
    assert low.targets == high.targets
    assert low.background == high.background
    assert low.prior_given == high.prior_given
    assert low.condition == high.condition
    assert low.world_id != high.world_id
    assert len(low.objects) - len(low.targets) <= 3
    assert 30 <= len(high.objects) - len(high.targets) <= 40
    assert stress_pair(5000000, ("sharp", "group"), CFG, LIBRARY) == (low, high)


def test_stress_rejects_t_library():
    from dataclasses import replace

    with pytest.raises(ValueError, match="B imagery"):
        stress_pair(5000000, ("sharp", "single"), CFG, replace(LIBRARY, split="T"))
