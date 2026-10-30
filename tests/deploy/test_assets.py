"""Texture boundaries and camera geometry independent of a running simulator."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from altiseek.config import load_config
from altiseek.deploy.assets import camera_sensor, checker_texture, export_tiles, static_camera_world
from altiseek.imaging.config import CameraSpec


def test_tiles_preserve_pixels_and_north_edge(tmp_path):
    rgb = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    model = ET.fromstring(export_tiles(rgb, tmp_path, 0.5, tile_px=4))
    assert len(model.findall("link/visual")) == 4
    reconstructed = np.zeros_like(rgb)
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    for row, col in ((0, 0), (0, 4), (4, 0), (4, 4)):
        tile = np.array(Image.open(tmp_path / f"tile_{row}_{col}.png"))
        h, w = tile.shape[:2]
        reconstructed[row : row + h, col : col + w] = tile
        root = ET.parse(tmp_path / f"tile_{row}_{col}.dae")
        positions = root.find('.//c:float_array[@id="positions-array"]', ns)
        xyz = np.fromstring(positions.text, sep=" ").reshape(4, 3)
        assert xyz[:, 0].min() == col * 0.5
        assert xyz[:, 0].max() == (col + w) * 0.5
        assert xyz[:, 1].min() == (5 - row - h) * 0.5
        assert xyz[:, 1].max() == (5 - row) * 0.5
        assert np.all(xyz[:, 2] == 0)
    assert np.array_equal(reconstructed, rgb)


def test_camera_uses_frozen_resolution_and_fov():
    camera = load_config("configs/camera/default.yaml", CameraSpec)
    sensor = ET.fromstring(camera_sensor(camera, "/test"))
    assert sensor.findtext("camera/image/width") == "640"
    assert sensor.findtext("camera/image/height") == "640"
    assert float(sensor.findtext("camera/horizontal_fov")) == pytest.approx(np.deg2rad(80))
    world = ET.fromstring(static_camera_world("", camera, [30, 90]))
    poses = [np.fromstring(node.text, sep=" ") for node in world.findall("world/model/pose")]
    assert poses[0] == pytest.approx([150, 150, 30, 0, np.pi / 2, np.pi / 2])
    assert poses[1][2] == 90


def test_asymmetric_checker_markers():
    image = checker_texture()
    assert np.array_equal(image[560, 640], [255, 0, 0])  # NE at (160 m,160 m).
    assert np.array_equal(image[640, 560], [0, 0, 255])  # SW at (140 m,140 m).


def test_bad_texture_rejected(tmp_path):
    with pytest.raises(ValueError, match="uint8"):
        export_tiles(np.zeros((4, 4, 3)), tmp_path, 0.05)
    with pytest.raises(ValueError, match="Positive"):
        export_tiles(np.zeros((4, 4, 3), dtype=np.uint8), tmp_path, -0.05)
