"""Focal length from the sea horizon (CLAUDE.md §7 step 4).

The files carry no field of view, and box sizes cannot give one: a box's pixel size depends on
the object's unknown physical size times the focal length. The open-sea horizon does not have
that ambiguity. For a camera pitched down by ``a`` at height ``h``, the visible horizon lies
``f * tan(a + bias - dip(h))`` pixels above the principal point. Over the small pitches where
it is in view this is close to linear, ``offset ~ f * tan(a - dip) + f * bias``, so a robust line
through (tan(a - dip), offset) gives the focal length as its slope and a constant gimbal pitch
bias (radians) as intercept / slope.

Logged pitch lags the camera while the gimbal moves (metadata is matched to frames by nearest
timestamp), so only frames whose pitch is steady over a time window are used.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from altiseek.data.frames import Frame
from altiseek.data.horizon import theil_sen
from altiseek.geometry.camera import horizon_dip_deg


@dataclass(frozen=True)
class HorizonSample:
    """A frame with an accepted horizon; ``offset_px`` is the horizon's height above centre."""

    frame: Frame
    offset_px: float

    @property
    def video(self) -> tuple[str, str | None]:
        return self.frame.camera_group, self.frame.source_folder

    @property
    def tan_angle(self) -> float:
        """tan(pitch - dip): the regressor whose slope is the focal length."""
        pitch, altitude = self.frame.gimbal_pitch_deg, self.frame.altitude_m
        if pitch is None or altitude is None:
            raise ValueError(f"image {self.frame.image_id} has no pitch or altitude")
        return math.tan(math.radians(pitch - horizon_dip_deg(altitude)))


@dataclass(frozen=True)
class FocalFit:
    focal_px: float
    focal_ci95_px: tuple[float, float]
    pitch_bias_deg: float
    pitch_bias_ci95_deg: tuple[float, float]
    #: Median absolute residual of the horizon offsets, in pixels.
    residual_mad_px: float
    n_frames: int
    n_videos: int


def steady_frames(
    frames: Sequence[Frame], window_s: float, tol_deg: float, min_neighbours: int = 2
) -> list[Frame]:
    """Frames whose video logs a pitch within ``tol_deg`` of theirs for ``window_s`` either side.

    A frame needs at least ``min_neighbours`` other frames of its video inside the window, and
    at least one on each side of it in time.
    """
    by_video: defaultdict[tuple[str, str | None], list[Frame]] = defaultdict(list)
    for frame in frames:
        if frame.timestamp is not None and frame.gimbal_pitch_deg is not None:
            by_video[frame.camera_group, frame.source_folder].append(frame)
    steady = []
    for members in by_video.values():
        members.sort(key=_time_key)
        times = np.array([_time_key(f)[0] for f in members])
        pitches = np.array([f.gimbal_pitch_deg for f in members], dtype=float)
        for i, frame in enumerate(members):
            near = np.abs(times - times[i]) <= window_s
            near[i] = False
            before = near & (times < times[i])
            after = near & (times > times[i])
            if near.sum() < min_neighbours or not before.any() or not after.any():
                continue
            if np.all(np.abs(pitches[near] - pitches[i]) <= tol_deg):
                steady.append(frame)
    return steady


def spread_per_video(frames: Sequence[Frame], per_video: int) -> list[Frame]:
    """At most ``per_video`` frames per video, evenly spaced in time (deterministic)."""
    by_video: defaultdict[tuple[str, str | None], list[Frame]] = defaultdict(list)
    for frame in frames:
        by_video[frame.camera_group, frame.source_folder].append(frame)
    picked = []
    for key in sorted(by_video, key=lambda k: (k[0], k[1] or "")):
        members = sorted(by_video[key], key=_time_key)
        if len(members) <= per_video:
            picked.extend(members)
        else:
            idx = np.linspace(0, len(members) - 1, per_video).round().astype(int)
            picked.extend(members[i] for i in idx)
    return picked


def fit_focal(samples: Sequence[HorizonSample], bootstrap_reps: int, seed: int) -> FocalFit:
    """Theil-Sen fit of horizon offsets on tan(pitch - dip), with a video-clustered bootstrap."""
    x = np.array([s.tan_angle for s in samples])
    y = np.array([s.offset_px for s in samples])
    intercept, slope = theil_sen(x, y)
    videos = sorted({s.video for s in samples}, key=lambda v: (v[0], v[1] or ""))
    members = {v: np.array([i for i, s in enumerate(samples) if s.video == v]) for v in videos}
    rng = np.random.default_rng(seed)
    focal, bias = [], []
    for _ in range(bootstrap_reps):
        chosen = rng.choice(len(videos), size=len(videos), replace=True)
        idx = np.concatenate([members[videos[k]] for k in chosen])
        if np.unique(x[idx]).size < 2:
            continue
        b0, b1 = theil_sen(x[idx], y[idx])
        focal.append(b1)
        bias.append(math.degrees(b0 / b1))
    return FocalFit(
        focal_px=slope,
        focal_ci95_px=_ci95(focal),
        pitch_bias_deg=math.degrees(intercept / slope),
        pitch_bias_ci95_deg=_ci95(bias),
        residual_mad_px=float(np.median(np.abs(y - (intercept + slope * x)))),
        n_frames=len(samples),
        n_videos=len(videos),
    )


def _time_key(frame: Frame) -> tuple[float, int]:
    """Sort key: POSIX seconds (frames without a timestamp first), then image id."""
    seconds = frame.timestamp.timestamp() if frame.timestamp is not None else -math.inf
    return seconds, frame.image_id


def _ci95(values: Sequence[float]) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)
