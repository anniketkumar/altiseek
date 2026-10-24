"""Local ODv2 images, fetched on demand from the dataset's public WebDAV share.

Images live at ``<root>/<official split>/<file_name>``, the layout of the share's ``images/``
folders (docs/data_notes.md). Files are written under a temporary name first, so an interrupted
fetch never leaves a truncated image behind and simply resumes on the next call.
"""

import base64
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import quote

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from altiseek.data.config import ImageSource
from altiseek.data.frames import Frame

#: Folder of each image version on the share.
VERSION_DIRS = {"compressed": "Compressed Version", "uncompressed": "Uncompressed Version"}


def image_path(source: ImageSource, frame: Frame) -> Path:
    """Where the image of ``frame`` lives locally."""
    return source.root / frame.official_split / frame.file_name


def image_url(source: ImageSource, frame: Frame) -> str:
    """WebDAV URL of the image of ``frame`` on the share."""
    parts = (VERSION_DIRS[source.version], "images", frame.official_split, frame.file_name)
    return source.webdav_url.rstrip("/") + "/" + "/".join(quote(p) for p in parts)


def download(
    source: ImageSource, frame: Frame, timeout_s: float = 120.0, attempts: int = 4
) -> bytes:
    """The encoded image of ``frame`` from the share, retrying with exponential backoff."""
    token = base64.b64encode(f"{source.share_token}:".encode()).decode()
    request = urllib.request.Request(
        image_url(source, frame), headers={"Authorization": f"Basic {token}"}
    )
    for attempt in range(attempts - 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2.0**attempt)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


def save_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temporary file, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(data)
    partial.replace(path)


def fetch_images(source: ImageSource, frames: Iterable[Frame], timeout_s: float = 120.0) -> int:
    """Download the images of ``frames`` that are missing locally; return how many were fetched."""
    fetched = 0
    for frame in frames:
        path = image_path(source, frame)
        if not path.exists():
            save_atomic(path, download(source, frame, timeout_s))
            fetched += 1
    return fetched


def load_rgb(path: str | Path) -> NDArray[np.uint8]:
    """An image file as an ``H x W x 3`` uint8 RGB array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))
