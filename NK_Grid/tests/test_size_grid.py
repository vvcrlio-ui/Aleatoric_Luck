from __future__ import annotations

import numpy as np
import pytest

from aleatoric_nk_grid.nk_grid import log2_size_grid


def test_log2_size_grid_production_panel_regression():
    result = log2_size_grid(3400, 20)

    assert result.tolist() == [
        1, 2, 3, 4, 6, 8, 13, 20, 31, 47, 72, 111, 170, 261, 400,
        614, 942, 1445, 2216, 3400,
    ]


def test_log2_size_grid_resolves_upper_bound_collisions():
    result = log2_size_grid(20, 12)

    assert len(result) == 12
    assert np.all(np.diff(result) > 0)
    assert result[0] == 1
    assert result[-1] == 20


def test_log2_size_grid_raises_when_capacity_is_insufficient():
    with pytest.raises(ValueError):
        log2_size_grid(5, 10)


def test_log2_size_grid_preserves_non_colliding_values():
    result = log2_size_grid(3400, 10)
    old_result = np.unique(
        np.clip(
            np.round(
                np.logspace(np.log2(1), np.log2(3400), num=10, base=2)
            ).astype(int),
            1,
            3400,
        )
    )

    assert np.array_equal(result, old_result)
