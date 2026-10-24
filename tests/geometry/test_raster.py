"""The world raster: north-up rows, the row flip, and windows on the pixel grid."""

import numpy as np
import pytest

from altiseek.geometry.raster import Raster

AREA = Raster.covering(300.0, 300.0, 0.05)  # 6000 x 6000 px


def test_covering_needs_whole_pixels():
    assert (AREA.width_px, AREA.height_px) == (6000, 6000)
    with pytest.raises(ValueError, match="whole number"):
        Raster.covering(300.0, 300.0, 0.07)


def test_rows_grow_southward():
    col, row = AREA.to_pixel([0.025, 299.975, 150.0], [299.975, 0.025, 150.0])
    assert col == pytest.approx([0.0, 5999.0, 2999.5])
    assert row == pytest.approx([0.0, 5999.0, 2999.5])  # north edge is row 0
    x, y = AREA.to_world(col, row)
    assert np.allclose(x, [0.025, 299.975, 150.0])
    assert np.allclose(y, [299.975, 0.025, 150.0])


def test_window_of_a_footprint():
    # The 50 m footprint of lattice cell (ix, iy) = (10, 20) with 2.5 m cells: x 25-75, y 50-100.
    row0, col0, size = AREA.window(25.0, 50.0, 50.0)
    assert (row0, col0, size) == (4000, 500, 1000)
    x, y = AREA.to_world(col0, row0)  # its north-west pixel
    assert (x, y) == pytest.approx((25.025, 99.975))
    with pytest.raises(ValueError, match="pixel grid"):
        AREA.window(25.01, 50.0, 50.0)
