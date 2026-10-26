"""Compressed per-world arrays, strict identity checks, and a bounded LRU pool.

View keys are (zero-based level, south-west ix, iy, variant). Ragged detections
and visible target truth use offsets, including explicit entries for empty views.
Compressed NPZ members are loaded per world; they cannot be memory-mapped.
"""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from altiseek.cache.config import InferenceSettings
from altiseek.config import Config
from altiseek.world.config import RenderConfig

DETECTION_DTYPE = np.dtype(
    [
        ("class_id", "<i4"),
        ("score", "<f4"),
        ("box_px", "<f4", (4,)),
        ("x_m", "<f8"),
        ("y_m", "<f8"),
        ("ix", "<i4"),
        ("iy", "<i4"),
    ]
)
TRUTH_DTYPE = np.dtype(
    [
        ("x_m", "<f8"),
        ("y_m", "<f8"),
        ("ix", "<i4"),
        ("iy", "<i4"),
        ("col_px", "<f8"),
        ("row_px", "<f8"),
    ]
)


class CacheHeader(Config):
    schema_version: Literal[1] = 1
    renderer_version: Literal["D28-v1"] = "D28-v1"
    world_id: str
    detector_id: str
    render: RenderConfig
    tau_min: float = Field(gt=0, lt=1)
    inference: InferenceSettings
    cache_id: str = ""

    def content_id(self) -> str:
        body = self.model_dump(mode="json", exclude={"cache_id"})
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]

    def identified(self) -> "CacheHeader":
        return self.model_copy(update={"cache_id": self.content_id()})


def view_keys(render: RenderConfig) -> NDArray[np.int32]:
    """Every lattice view and variant, in stable level/row/column/variant order."""
    if render.variants.count < 1 or not render.levels.levels:
        raise ValueError("cache needs at least one level and variant")
    return np.asarray(
        [
            (li, ix, iy, v)
            for li, level in enumerate(render.levels.levels)
            for ix, iy in render.levels.lattice(level)
            for v in range(render.variants.count)
        ],
        dtype=np.int32,
    )


def draw_variant(seed: int, observation: int, count: int) -> int:
    """Method-independent draw by episode seed and observation number (no global RNG)."""
    if seed < 0 or observation < 0 or count < 1:
        raise ValueError("seed/observation must be nonnegative and count positive")
    return int(np.random.default_rng([seed, observation]).integers(count))


class WorldCache:
    def __init__(
        self,
        header: CacheHeader,
        views: NDArray[np.int32],
        offsets: NDArray[np.int64],
        detections: np.ndarray,
        truth_offsets: NDArray[np.int64],
        truth: np.ndarray,
    ) -> None:
        self.header = header
        self.views = views
        self.offsets, self.detections = offsets, detections
        self.truth_offsets, self.truth = truth_offsets, truth
        self.validate()
        self._index = {tuple(int(v) for v in key): i for i, key in enumerate(views)}
        for array in (views, offsets, detections, truth_offsets, truth):
            array.flags.writeable = False

    def validate(self) -> None:
        if self.header.cache_id != self.header.content_id():
            raise ValueError("cache_id does not match header content")
        expected = view_keys(self.header.render)
        if self.views.dtype != np.dtype("int32") or not np.array_equal(self.views, expected):
            raise ValueError("incomplete or unordered view lattice/variants")
        for offsets, rows, dtype in (
            (self.offsets, self.detections, DETECTION_DTYPE),
            (self.truth_offsets, self.truth, TRUTH_DTYPE),
        ):
            if rows.ndim != 1 or rows.dtype != dtype:
                raise ValueError("invalid record dtype/shape")
            if (
                offsets.dtype != np.dtype("int64")
                or offsets.shape != (len(expected) + 1,)
                or offsets[0] != 0
                or offsets[-1] != len(rows)
                or (np.diff(offsets) < 0).any()
            ):
                raise ValueError("invalid ragged offsets")
            for name in dtype.names or ():
                if not np.isfinite(rows[name]).all():
                    raise ValueError("non-finite cache record")
            table = self.header.render.levels
            for axis in ("x", "y"):
                pos, cell = rows[f"{axis}_m"], rows[f"i{axis}"]
                if ((pos < 0) | (pos >= table.area_m)).any() or not np.array_equal(
                    cell, np.floor(pos / table.cell_m).astype(np.int32)
                ):
                    raise ValueError("invalid world position/cell")
        d = self.detections
        if ((d["score"] < self.header.tau_min) | (d["score"] > 1)).any():
            raise ValueError("score outside cached range")
        boxes = d["box_px"]
        if (
            (d["class_id"] < 0).any()
            or (boxes < 0).any()
            or (boxes > self.header.render.camera.resolution_px).any()
            or (boxes[:, 2:] <= boxes[:, :2]).any()
        ):
            raise ValueError("invalid detection class/box")

    def lookup(self, level: int, ix: int, iy: int, variant: int) -> tuple[np.ndarray, np.ndarray]:
        i = self._index[(level, ix, iy, variant)]
        return (
            self.detections[self.offsets[i] : self.offsets[i + 1]],
            self.truth[self.truth_offsets[i] : self.truth_offsets[i + 1]],
        )

    def save(self, path: Path) -> None:
        """Atomic replace: an interrupted world never becomes a completed cache."""
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as fh:
            np.savez_compressed(
                fh,
                header=np.asarray(self.header.model_dump_json()),
                views=self.views,
                offsets=self.offsets,
                detections=self.detections,
                truth_offsets=self.truth_offsets,
                truth=self.truth,
            )
        temp.replace(path)

    @classmethod
    def load(cls, path: Path, *, expected: CacheHeader) -> "WorldCache":
        with np.load(path, allow_pickle=False) as data:
            header = CacheHeader.model_validate_json(str(data["header"].item()))
            if header != expected:
                raise ValueError(
                    "cache header mismatch (world/detector/render/inference/threshold)"
                )
            return cls(
                header,
                *(data[k] for k in ("views", "offsets", "detections", "truth_offsets", "truth")),
            )


class CachePool:
    """An LRU pool bounded by world count; every access requires the full expected header."""

    def __init__(self, capacity: int = 8) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._worlds: OrderedDict[tuple[Path, str], WorldCache] = OrderedDict()

    def get(self, path: Path, *, expected: CacheHeader) -> WorldCache:
        if expected.cache_id != expected.content_id():
            raise ValueError("invalid expected cache_id")
        key = (path.resolve(), expected.cache_id)
        if key not in self._worlds:
            self._worlds[key] = WorldCache.load(path, expected=expected)
        self._worlds.move_to_end(key)
        while len(self._worlds) > self.capacity:
            self._worlds.popitem(last=False)
        return self._worlds[key]
