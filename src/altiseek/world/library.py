"""Library extraction (CLAUDE.md §11): empty-sea patches and object crops from one split's
frames, rectified into nadir-equivalent views at g0 (D26) by ``altiseek.geometry.rectify``. No
camera blur or JPEG is applied here; the simulated camera adds them at render time.

Backgrounds are square patches cut from rectified tiles of a frame's usable footprint. Every
pixel must be valid, no blackened (ignore-region) pixel may lie within ``black_margin_px``, and
the patch keeps ``object_margin_m`` from every annotated object's ground footprint.

Object crops paste the box footprint grown by an allowance, because some ODv2 boxes are offset
from their objects (docs/data_notes.md): the grown region holds the object, its splash and wake
even then. A pixel-based cut-out was tried and dropped (D27): dark wetsuits, splash and nearby
life rings defeat any colour threshold. Around the pasted region, a ring of the object's own
water fades the paste into the background and gives the water's colour for matching. The
object's position is the box centre. Crops with blackened pixels, or whose pasted region also
holds another annotated object, are rejected. The detector never selects or moves crops: that
would favour the swimmers it already finds.
"""

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from altiseek.data.frames import Box, Frame
from altiseek.data.horizon import BLACK_LUMA, LUMA
from altiseek.data.splits import SplitRow
from altiseek.data.tables import read_csv
from altiseek.detector.views import tile_windows
from altiseek.geometry.camera import TiltedCamera
from altiseek.geometry.rectify import ViewWindow, rectify
from altiseek.imaging.pipeline import CameraPipeline
from altiseek.world.appearance import Stats, stats
from altiseek.world.config import BackgroundSettings, ObjectSettings

#: Rectification only: blur and JPEG belong to the simulated camera, at render time.
NO_PIPELINE = CameraPipeline()


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle on the ground, metres."""

    x0_m: float
    y0_m: float
    x1_m: float
    y1_m: float

    def grown(self, margin_m: float) -> "Rect":
        return Rect(
            self.x0_m - margin_m, self.y0_m - margin_m, self.x1_m + margin_m, self.y1_m + margin_m
        )

    def intersects(self, other: "Rect") -> bool:
        return (
            self.x0_m < other.x1_m
            and other.x0_m < self.x1_m
            and self.y0_m < other.y1_m
            and other.y0_m < self.y1_m
        )

    def contains(self, x_m: NDArray[np.float64], y_m: NDArray[np.float64]) -> NDArray[np.bool_]:
        return (x_m >= self.x0_m) & (x_m < self.x1_m) & (y_m >= self.y0_m) & (y_m < self.y1_m)


@dataclass(frozen=True)
class BackgroundRow:
    """One background patch, a row of ``backgrounds.csv``."""

    patch_id: str
    image_id: int
    session_id: str
    video: str
    split: str
    sha256: str
    l_mean: float
    a_mean: float
    b_mean: float
    l_std: float
    a_std: float
    b_std: float
    texture: float
    #: Appearance condition, dark to bright; -1 until assigned.
    condition: int


@dataclass(frozen=True)
class ObjectRow:
    """One object crop, a row of ``objects.csv``."""

    crop_id: str
    category: str
    image_id: int
    annotation_id: int
    session_id: str
    video: str
    split: str
    sha256: str
    #: Side of the square crop, pixels at g0.
    size_px: int
    #: The object's position in the crop (the box centre; column, row, pixel centres at
    #: integers).
    centre_col: float
    centre_row: float
    #: The box footprint on the ground, metres.
    width_m: float
    height_m: float
    #: Lab statistics of the water ring.
    water_l: float
    water_a: float
    water_b: float
    water_l_std: float
    water_a_std: float
    water_b_std: float


def select_frames(
    frames: Iterable[Frame],
    split_rows: dict[int, SplitRow],
    split: str,
    camera_groups: Sequence[str],
    stride: int,
) -> list[Frame]:
    """The split's frames of ``camera_groups``: every ``stride``-th of each video, in frame
    order, so near-copies of consecutive frames are skipped."""
    videos: defaultdict[str, list[Frame]] = defaultdict(list)
    for f in frames:
        if split_rows[f.image_id].split == split and f.camera_group in camera_groups:
            videos[f.source_folder or ""].append(f)
    chosen = []
    for video in sorted(videos):
        ordered = sorted(videos[video], key=lambda f: (f.source_frame_no or 0, f.image_id))
        chosen.extend(ordered[::stride])
    return sorted(chosen, key=lambda f: f.image_id)


def box_footprints(camera: TiltedCamera, boxes: Sequence[Box]) -> list[Rect | None]:
    """Each box's ground footprint: the rectangle bounding its four corners, or None if a corner
    misses the water."""
    if not boxes:
        return []
    xywh = np.array([[b.x_px, b.y_px, b.w_px, b.h_px] for b in boxes], dtype=np.float64)
    x0, y0 = xywh[:, 0] - 0.5, xywh[:, 1] - 0.5  # pixel-corner to pixel-centre coordinates
    x1, y1 = x0 + xywh[:, 2], y0 + xywh[:, 3]
    gx, gy, water = camera.image_to_ground(
        np.stack([x0, x1, x0, x1], axis=1), np.stack([y0, y0, y1, y1], axis=1)
    )
    return [
        Rect(gx[i].min(), gy[i].min(), gx[i].max(), gy[i].max()) if water[i].all() else None
        for i in range(len(boxes))
    ]


def box_centres_m(camera: TiltedCamera, boxes: Sequence[Box]) -> NDArray[np.float64]:
    """Ground positions (``n x 2``) of box centres; NaN where the centre misses the water."""
    if not boxes:
        return np.zeros((0, 2))
    u = np.array([b.x_px + b.w_px / 2 - 0.5 for b in boxes])
    v = np.array([b.y_px + b.h_px / 2 - 0.5 for b in boxes])
    x, y, water = camera.image_to_ground(u, v)
    out = np.stack([x, y], axis=1)
    out[~water] = np.nan
    return out


def background_patches(
    rgb: NDArray[np.uint8],
    camera: TiltedCamera,
    boxes: Sequence[Box],
    *,
    gsd_mpp: float,
    max_off_nadir_deg: float,
    supersample: int,
    settings: BackgroundSettings,
) -> list[tuple[str, Rect, NDArray[np.uint8]]]:
    """Clean patches of one frame: ``(name, ground rectangle, image)``, named
    ``<tile>-<row>-<col>``."""
    p = settings.patch_px
    side = p * gsd_mpp
    keep_out = [r.grown(settings.object_margin_m) for r in box_footprints(camera, boxes) if r]
    kernel = np.ones((2 * settings.black_margin_px + 1,) * 2, dtype=np.uint8)
    tiles = tile_windows(
        camera, gsd_mpp, settings.tile_px, max_off_nadir_deg, settings.tile_coverage
    )
    out = []
    for t, window in enumerate(tiles):
        view = rectify(
            rgb,
            camera,
            window,
            max_off_nadir_deg=max_off_nadir_deg,
            pipeline=NO_PIPELINE,
            supersample=supersample,
        )
        black = cv2.dilate((view.image @ LUMA < BLACK_LUMA).astype(np.uint8), kernel) > 0
        top = window.y0_m + window.side_m
        for r in range(settings.tile_px // p):
            for c in range(settings.tile_px // p):
                rows, cols = slice(r * p, (r + 1) * p), slice(c * p, (c + 1) * p)
                if not view.valid[rows, cols].all() or black[rows, cols].any():
                    continue
                x0, y1 = window.x0_m + c * side, top - r * side
                rect = Rect(x0, y1 - side, x0 + side, y1)
                if any(rect.intersects(k) for k in keep_out):
                    continue
                out.append((f"{t}-{r}-{c}", rect, view.image[rows, cols].copy()))
    return out


@dataclass(frozen=True)
class Crop:
    """An object crop at g0: RGBA, opaque over the pasted region and fading across the ring."""

    rgba: NDArray[np.uint8]
    #: The object's position (the box centre; column, row, pixel centres at integers).
    centre_px: tuple[float, float]
    #: The box footprint, metres.
    size_m: tuple[float, float]
    #: Statistics of the water ring, for colour matching when pasted.
    water: Stats


def object_crop(
    rgb: NDArray[np.uint8],
    camera: TiltedCamera,
    box: Box,
    others_m: NDArray[np.float64],
    *,
    gsd_mpp: float,
    max_off_nadir_deg: float,
    supersample: int,
    settings: ObjectSettings,
) -> tuple[Crop | None, str]:
    """The crop of ``box``, or None and the reason. ``others_m`` holds the ground centres of the
    frame's other boxes: a crop whose pasted region also holds one of them is rejected."""
    (foot,) = box_footprints(camera, [box])
    if foot is None:
        return None, "off the water"
    (centre,) = box_centres_m(camera, [box])
    width, height = foot.x1_m - foot.x0_m, foot.y1_m - foot.y0_m
    pasted = foot.grown(
        max(settings.allowance_min_m, settings.allowance_box_factor * max(width, height))
    )
    others = others_m[~np.isnan(others_m).any(axis=1)]
    if pasted.contains(others[:, 0], others[:, 1]).any():
        return None, "neighbour"
    # A square window centred on the box centre: the pasted region and the ring around it.
    half = max(
        centre[0] - pasted.x0_m,
        pasted.x1_m - centre[0],
        centre[1] - pasted.y0_m,
        pasted.y1_m - centre[1],
    )
    n = math.ceil(2 * half / gsd_mpp) + 2 * settings.ring_px
    window = ViewWindow(centre[0] - n * gsd_mpp / 2, centre[1] - n * gsd_mpp / 2, gsd_mpp, n)
    view = rectify(
        rgb,
        camera,
        window,
        max_off_nadir_deg=max_off_nadir_deg,
        pipeline=NO_PIPELINE,
        supersample=supersample,
    )
    if not view.valid.all():
        return None, "outside the usable footprint"
    if (view.image @ LUMA < BLACK_LUMA).any():
        return None, "blackened"
    cols, rows = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64))
    gx = window.x0_m + (cols + 0.5) * gsd_mpp
    gy = window.y0_m + window.side_m - (rows + 0.5) * gsd_mpp
    inside = pasted.contains(gx, gy)
    # Opaque over the pasted region, fading linearly to nothing across the ring.
    distance = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(1.0 - distance / settings.ring_px, 0.0, 1.0)
    col, row = window.ground_to_view(centre[:1], centre[1:])
    return (
        Crop(
            rgba=np.dstack([view.image, np.rint(alpha * 255).astype(np.uint8)]),
            centre_px=(float(col[0]), float(row[0])),
            size_m=(width, height),
            water=stats(view.image, ~inside),
        ),
        "ok",
    )


@dataclass(frozen=True)
class Library:
    """A built library (``scripts/build_libraries.py``): its index tables and where its images
    are."""

    path: Path
    library_id: str
    split: str
    gsd_mpp: float
    #: Side of a background patch, and width of the fading water ring of a crop, pixels.
    patch_px: int
    ring_px: int
    backgrounds: tuple[BackgroundRow, ...]
    objects: tuple[ObjectRow, ...]
    #: Per appearance condition: patch count and mean Lab statistics (``conditions.json``).
    conditions: tuple[dict[str, float], ...]

    @classmethod
    def load(cls, path: str | Path) -> "Library":
        path = Path(path)
        meta = json.loads((path / "library.meta.json").read_text(encoding="utf-8"))
        conditions = json.loads((path / "conditions.json").read_text(encoding="utf-8"))
        return cls(
            path=path,
            library_id=meta["library_id"],
            split=meta["split"],
            gsd_mpp=meta["config"]["gsd_mpp"],
            patch_px=meta["config"]["background"]["patch_px"],
            ring_px=meta["config"]["objects"]["ring_px"],
            backgrounds=tuple(read_csv(path / "backgrounds.csv", BackgroundRow)),
            objects=tuple(read_csv(path / "objects.csv", ObjectRow)),
            conditions=tuple(conditions["conditions"]),
        )

    def patch_path(self, patch_id: str) -> Path:
        return self.path / "backgrounds" / f"{patch_id}.png"

    def patch_image(self, patch_id: str) -> NDArray[np.uint8]:
        """The patch's RGB image."""
        with Image.open(self.patch_path(patch_id)) as image:
            return np.asarray(image)

    def crop_path(self, crop_id: str) -> Path:
        return self.path / "objects" / f"{crop_id}.png"

    def crop_image(self, crop_id: str) -> NDArray[np.uint8]:
        """The crop's RGBA image."""
        with Image.open(self.crop_path(crop_id)) as image:
            return np.asarray(image)
