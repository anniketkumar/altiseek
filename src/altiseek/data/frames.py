"""SeaDronesSee ODv2 frame and box tables (CLAUDE.md §7, step 1).

Field names and conventions were verified on the ODv2 files (docs/data_notes.md):

- ``meta`` comes in two schemas. Frames from the DJI quadcopters carry flight-log fields
  (``height_above_takeoff(meter)``, ``gimbal_pitch(degrees)``, ``latitude``, ...); the
  MicaSense multispectral frames carry ``altitude``, ``gimbal_pitch``, ``gps_latitude``, ...
  Every other image has an empty ``meta``.
- Gimbal pitch is in degrees below the horizon: 90 is a nadir view, 0 a horizontal one
  (Varga et al. 2022, §3.1). Altitude is barometric height above the take-off point, in metres.
- The image-level ``date_time`` is local time with sub-second precision. Multispectral frames
  only have ``meta.date_time``, whose time zone is not documented.
- Boxes are COCO ``[x, y, w, h]`` in image pixels with the origin at the top-left corner.
"""

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from altiseek.data.config import CameraGroupRule

#: ODv2 category names. ``ignored`` exists but has no boxes in train or val.
CATEGORIES = ("ignored", "swimmer", "boat", "jetski", "life_saving_appliances", "buoy")


@dataclass(frozen=True)
class Frame:
    """One labelled image. Sizes in pixels, altitude in metres, angles in degrees."""

    image_id: int
    file_name: str
    official_split: str
    width_px: int
    height_px: int
    camera_group: str
    drone: str
    source_folder: str | None
    source_frame_no: int | None
    timestamp: datetime | None
    #: ``"image"`` (local time) or ``"meta"`` (multispectral, zone undocumented).
    timestamp_source: str | None
    altitude_m: float | None
    #: Degrees below the horizon; 90 is nadir.
    gimbal_pitch_deg: float | None
    latitude_deg: float | None
    longitude_deg: float | None
    n_ignored: int
    n_swimmer: int
    n_boat: int
    n_jetski: int
    n_life_saving_appliances: int
    n_buoy: int

    @property
    def off_nadir_deg(self) -> float | None:
        """Angle between the optical axis and straight down, in degrees."""
        return None if self.gimbal_pitch_deg is None else 90.0 - self.gimbal_pitch_deg


@dataclass(frozen=True)
class Box:
    """One annotated object; ``x_px``, ``y_px`` is the top-left corner in image pixels."""

    annotation_id: int
    image_id: int
    category: str
    x_px: float
    y_px: float
    w_px: float
    h_px: float


def load_odv2(
    annotation_files: Mapping[str, Path], camera_groups: Sequence[CameraGroupRule]
) -> tuple[list[Frame], list[Box]]:
    """Parse ODv2 COCO files, keyed by official split, into frame and box tables."""
    frames: list[Frame] = []
    boxes: list[Box] = []
    for split, path in annotation_files.items():
        coco = json.loads(Path(path).read_text(encoding="utf-8"))
        names = {category["id"]: category["name"] for category in coco["categories"]}
        unknown = set(names.values()) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"{path}: unknown categories {sorted(unknown)}")
        counts: defaultdict[int, Counter[str]] = defaultdict(Counter)
        for ann in coco["annotations"]:
            category = names[ann["category_id"]]
            counts[ann["image_id"]][category] += 1
            x, y, w, h = (float(v) for v in ann["bbox"])
            boxes.append(Box(ann["id"], ann["image_id"], category, x, y, w, h))
        frames.extend(
            _frame(image, split, counts[image["id"]], camera_groups) for image in coco["images"]
        )
    duplicates = [i for i, n in Counter(f.image_id for f in frames).items() if n > 1]
    if duplicates:
        raise ValueError(f"image ids appear more than once: {sorted(duplicates)[:10]}")
    return frames, boxes


def _frame(
    image: Mapping[str, Any],
    split: str,
    counts: Counter[str],
    camera_groups: Sequence[CameraGroupRule],
) -> Frame:
    source = image["source"]
    if isinstance(source, Mapping):
        drone, folder, frame_no = source["drone"], source.get("folder_name"), source.get("frame_no")
    else:
        drone, folder, frame_no = source, None, None
    width, height = image["width"], image["height"]
    groups = [rule.name for rule in camera_groups if rule.matches(drone, width, height)]
    if len(groups) != 1:
        raise ValueError(
            f"image {image['id']} ({drone}, {width}x{height}) matches camera groups {groups}"
        )
    altitude, pitch, lat, lon, meta_time = _read_meta(image["meta"], image["id"])
    if image.get("date_time"):
        timestamp, timestamp_source = datetime.fromisoformat(image["date_time"]), "image"
    elif meta_time:
        timestamp, timestamp_source = datetime.fromisoformat(meta_time), "meta"
    else:
        timestamp, timestamp_source = None, None
    return Frame(
        image_id=image["id"],
        file_name=image["file_name"],
        official_split=split,
        width_px=width,
        height_px=height,
        camera_group=groups[0],
        drone=drone,
        source_folder=folder,
        source_frame_no=frame_no,
        timestamp=timestamp,
        timestamp_source=timestamp_source,
        altitude_m=altitude,
        gimbal_pitch_deg=pitch,
        latitude_deg=lat,
        longitude_deg=lon,
        **{f"n_{category}": counts[category] for category in CATEGORIES},
    )


def _read_meta(
    meta: Mapping[str, Any], image_id: int
) -> tuple[float | None, float | None, float | None, float | None, str | None]:
    """Return altitude (m), gimbal pitch (deg), latitude, longitude and the meta timestamp."""
    if not meta:
        return None, None, None, None, None
    if "height_above_takeoff(meter)" in meta:  # DJI flight log
        return (
            _positive(meta["height_above_takeoff(meter)"]),
            _finite(meta["gimbal_pitch(degrees)"]),
            _finite(meta.get("latitude")),
            _finite(meta.get("longitude")),
            None,
        )
    if "altitude" in meta:  # MicaSense RedEdge-MX
        return (
            _positive(meta["altitude"]),
            _finite(meta.get("gimbal_pitch")),
            _signed(meta.get("gps_latitude"), meta.get("gps_latitude_ref"), negative="S"),
            _signed(meta.get("gps_longitude"), meta.get("gps_longitude_ref"), negative="W"),
            meta.get("date_time") or None,
        )
    raise ValueError(f"image {image_id}: unrecognised meta keys {sorted(meta)}")


def _finite(value: Any) -> float | None:
    """A finite number as float; anything else (missing, empty string, NaN) as None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _positive(value: Any) -> float | None:
    """Altitudes that are not finite and positive are treated as missing."""
    number = _finite(value)
    return number if number is not None and number > 0 else None


def _signed(value: Any, ref: Any, negative: str) -> float | None:
    number = _finite(value)
    if number is None:
        return None
    return -abs(number) if isinstance(ref, str) and ref.upper() == negative else number
