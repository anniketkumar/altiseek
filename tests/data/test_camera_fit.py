"""Steady-pitch selection, per-video spreading and the horizon focal-length fit."""

import math
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from altiseek.data.camera_fit import HorizonSample, fit_focal, spread_per_video, steady_frames
from altiseek.data.frames import Frame
from altiseek.geometry.camera import horizon_offset_px

T0 = datetime(2021, 8, 20, 15, 0, 0)

FRAME = Frame(
    image_id=0,
    file_name="0.jpg",
    official_split="train",
    width_px=3840,
    height_px=2160,
    camera_group="mavic_4k",
    drone="mavic",
    source_folder="V",
    source_frame_no=None,
    timestamp=T0,
    timestamp_source="image",
    altitude_m=30.0,
    gimbal_pitch_deg=10.0,
    latitude_deg=53.72,
    longitude_deg=7.16,
    n_ignored=0,
    n_swimmer=0,
    n_boat=0,
    n_jetski=0,
    n_life_saving_appliances=0,
    n_buoy=0,
)


def video(pitches, folder="V", start_id=0, step_s=0.5):
    return [
        replace(
            FRAME,
            image_id=start_id + i,
            source_folder=folder,
            timestamp=T0 + timedelta(seconds=i * step_s),
            gimbal_pitch_deg=p,
        )
        for i, p in enumerate(pitches)
    ]


def test_steady_frames_skip_gimbal_motion_and_the_ends():
    pitches = [10.0] * 12 + list(np.linspace(10.5, 19.5, 10)) + [20.0] * 12
    frames = video(pitches)
    steady = {f.image_id for f in steady_frames(frames, window_s=2.0, tol_deg=0.3)}
    assert 0 not in steady  # nothing before it
    assert set(range(1, 8)) <= steady  # plateau at 10 deg, more than 2 s from the ramp
    assert not steady & set(range(9, 25))  # the ramp and its 2 s neighbourhood
    assert set(range(26, 33)) <= steady
    assert 33 not in steady  # nothing after it


def test_steady_frames_are_judged_per_video():
    a = video([10.0] * 5, folder="A")
    b = video([30.0] * 5, folder="B", start_id=100)
    steady = {f.image_id for f in steady_frames(a + b, window_s=2.0, tol_deg=0.3)}
    assert steady == {1, 2, 3, 101, 102, 103}


def test_spread_per_video_is_even_and_deterministic():
    frames = video([10.0] * 10, folder="A") + video([10.0] * 2, folder="B", start_id=100)
    picked = [f.image_id for f in spread_per_video(frames, per_video=4)]
    assert picked == [0, 3, 6, 9, 100, 101]


def test_fit_recovers_focal_length_and_pitch_bias():
    focal_px, bias_deg = 3000.0, -0.4
    rng = np.random.default_rng(0)
    samples = []
    for v in range(8):
        for pitch in np.linspace(1.5 + v, 9.0 + v, 8):
            frame = replace(
                FRAME,
                image_id=len(samples),
                source_folder=f"V{v}",
                gimbal_pitch_deg=float(pitch),
                altitude_m=20.0 + 5 * v,
            )
            offset = horizon_offset_px(pitch + bias_deg, frame.altitude_m, focal_px)
            samples.append(HorizonSample(frame, offset + rng.normal(0, 4)))
    fit = fit_focal(samples, bootstrap_reps=300, seed=1)
    assert fit.focal_px == pytest.approx(focal_px, rel=0.01)
    assert fit.pitch_bias_deg == pytest.approx(bias_deg, abs=0.1)
    assert fit.focal_ci95_px[0] <= focal_px <= fit.focal_ci95_px[1]
    assert fit.pitch_bias_ci95_deg[0] <= bias_deg <= fit.pitch_bias_ci95_deg[1]
    assert (fit.n_frames, fit.n_videos) == (64, 8)
    assert fit.residual_mad_px < 6


def test_tan_angle_needs_pitch_and_altitude():
    sample = HorizonSample(replace(FRAME, altitude_m=None), 100.0)
    with pytest.raises(ValueError, match="no pitch or altitude"):
        _ = sample.tan_angle
    assert HorizonSample(FRAME, 0.0).tan_angle == pytest.approx(
        math.tan(math.radians(10.0 - 0.17582)), rel=1e-3
    )
