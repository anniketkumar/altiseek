"""Image paths and share URLs (no network access)."""

from pathlib import Path

import numpy as np
from PIL import Image

from altiseek.data.config import ImageSource
from altiseek.data.images import (
    annotation_url,
    fetch_annotations,
    fetch_images,
    image_path,
    image_url,
    load_rgb,
)

from ._frames import FRAME


def source(root):
    return ImageSource(
        version="compressed",
        root=root,
        webdav_url="https://example.org/public.php/webdav/",
        share_token="TOKEN",
    )


def test_path_and_url_follow_the_share_layout(tmp_path):
    frame = FRAME
    assert image_path(source(tmp_path), frame) == tmp_path / "train" / "0.jpg"
    assert (
        image_url(source(tmp_path), frame)
        == "https://example.org/public.php/webdav/Compressed%20Version/images/train/0.jpg"
    )


def test_annotation_url_and_existing_files_are_not_fetched(tmp_path):
    assert annotation_url(source(tmp_path), "instances_val.json") == (
        "https://example.org/public.php/webdav/Compressed%20Version/annotations/instances_val.json"
    )
    present = tmp_path / "instances_val.json"
    present.write_text("{}", encoding="utf-8")
    assert fetch_annotations(source(tmp_path), [present]) == 0


def test_existing_images_are_not_fetched(tmp_path):
    path = image_path(source(tmp_path), FRAME)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"already here")
    assert fetch_images(source(tmp_path), [FRAME]) == 0
    assert path.read_bytes() == b"already here"


def test_load_rgb_returns_uint8_rgb(tmp_path):
    path = Path(tmp_path) / "grey.png"
    Image.new("L", (6, 4), color=128).save(path)
    rgb = load_rgb(path)
    assert rgb.shape == (4, 6, 3)
    assert rgb.dtype == np.uint8
    assert (rgb == 128).all()
