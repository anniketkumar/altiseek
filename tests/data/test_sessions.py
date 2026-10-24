"""Capture-day sessions across camera groups."""

from dataclasses import replace
from datetime import date, datetime

import pytest

from altiseek.data.config import Site
from altiseek.data.sessions import build_sessions, session_of, trusted_day

from ._frames import FRAME

SITES = (
    Site(name="lake_constance", latitude_deg=(47.0, 48.0), longitude_deg=(8.5, 10.0)),
    Site(name="north_sea", latitude_deg=(53.0, 54.5), longitude_deg=(6.5, 8.0)),
)
UNDATED = {"trinity_rgb": "lake_constance_2020", "trinity_multispectral": "lake_constance_2020"}

SEA = FRAME
LAKE = replace(
    FRAME, timestamp=datetime(2020, 8, 27, 14, 0), latitude_deg=47.67, longitude_deg=9.27
)


def frame(image_id, base=SEA, **changes):
    return replace(base, image_id=image_id, **changes)


def test_trusted_days():
    assert trusted_day(SEA) == date(2021, 8, 20)
    meta_with_gps = replace(LAKE, camera_group="trinity_multispectral", timestamp_source="meta")
    assert trusted_day(meta_with_gps) == date(2020, 8, 27)
    meta_without_gps = replace(meta_with_gps, latitude_deg=None, longitude_deg=None)
    assert trusted_day(meta_without_gps) is None
    assert trusted_day(replace(SEA, timestamp=None, timestamp_source=None)) is None


def test_one_session_per_day_across_camera_groups():
    frames = [
        frame(1),
        frame(2, camera_group="m210_1080p", drone="m210", latitude_deg=None, longitude_deg=None),
        frame(3, timestamp=datetime(2021, 8, 21, 13, 0)),
        frame(4, base=LAKE, camera_group="m210_4k", drone="m210"),
    ]
    of, sessions = build_sessions(frames, SITES, UNDATED)
    assert of == {1: "2021-08-20", 2: "2021-08-20", 3: "2021-08-21", 4: "2020-08-27"}
    assert sessions["2021-08-20"].campaign == "north_sea_2021"
    assert sessions["2021-08-20"].n_frames == 2
    assert sessions["2020-08-27"].campaign == "lake_constance_2020"


def test_undated_frames_get_their_configured_campaign():
    rgb = frame(5, camera_group="trinity_rgb", timestamp=None, timestamp_source=None)
    clockless = frame(
        6,
        base=LAKE,
        camera_group="trinity_multispectral",
        timestamp=datetime(2021, 8, 27, 11, 0),
        timestamp_source="meta",
        latitude_deg=None,
        longitude_deg=None,
    )
    of, sessions = build_sessions([frame(1), rgb, clockless], SITES, UNDATED)
    assert session_of(rgb) == of[5] == "undated:trinity_rgb"
    assert of[6] == "undated:trinity_multispectral"
    assert sessions["undated:trinity_rgb"].campaign == "lake_constance_2020"
    assert sessions["undated:trinity_rgb"].day is None


def test_undated_group_without_a_campaign_is_an_error():
    stray = frame(5, camera_group="mystery", timestamp=None, timestamp_source=None)
    with pytest.raises(ValueError, match="no campaign"):
        build_sessions([stray], SITES, UNDATED)


def test_day_spanning_two_sites_is_an_error():
    frames = [frame(1), frame(2, latitude_deg=47.67, longitude_deg=9.27)]
    with pytest.raises(ValueError, match="sites"):
        build_sessions(frames, SITES, UNDATED)


def test_position_outside_every_site_is_an_error():
    with pytest.raises(ValueError, match="matches sites"):
        build_sessions([frame(1, latitude_deg=0.0, longitude_deg=0.0)], SITES, UNDATED)


def test_day_without_any_position_is_an_error():
    lone = frame(1, latitude_deg=None, longitude_deg=None)
    with pytest.raises(ValueError, match="sites"):
        build_sessions([lone], SITES, UNDATED)
