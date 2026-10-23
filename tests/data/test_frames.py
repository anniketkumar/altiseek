"""ODv2 parsing on synthetic COCO files shaped like the real ones (docs/data_notes.md)."""

import json
import math
from datetime import datetime
from pathlib import Path

import pytest

from altiseek.config import load_config
from altiseek.data.config import DataConfig
from altiseek.data.frames import Box, load_odv2

REPO_ROOT = Path(__file__).resolve().parents[2]
GROUPS = load_config(REPO_ROOT / "configs" / "data" / "default.yaml", DataConfig).camera_groups

CATEGORIES = [
    {"supercategory": "ignored", "id": 0, "name": "ignored"},
    {"supercategory": "person", "id": 1, "name": "swimmer"},
    {"supercategory": "boat", "id": 2, "name": "boat"},
    {"supercategory": "boat", "id": 3, "name": "jetski"},
    {"supercategory": "object", "id": 4, "name": "life_saving_appliances"},
    {"supercategory": "object", "id": 5, "name": "buoy"},
]


def dji_meta(altitude_m=35.2, pitch_deg=85.0):
    return {
        "image_name": "DJI_0032_merged.MP4_image0.png",
        "datetime(utc)": "2020-08-25 12:16:22",
        "latitude": 47.673596,
        "longitude": 9.270249,
        "height_above_takeoff(meter)": altitude_m,
        "speed(m/s)": 0.0,
        "xspeed(m/s)": 0.0,
        "yspeed(m/s)": 0.0,
        "zspeed(m/s)": 0.0,
        "compass_heading(degrees)": 175.8,
        "gimbal_heading(degrees)": 154.4,
        "gimbal_pitch(degrees)": pitch_deg,
    }


def micasense_meta(with_gps=True):
    meta = {
        "date_time": "2020-08-27T13:03:39",
        "gps_latitude_ref": "S" if with_gps else math.nan,
        "gps_longitude_ref": "W" if with_gps else math.nan,
        "altitude": 110.5,
        "gimbal_pitch": 90.0,
        "speed": -1,
        "xspeed": -1,
        "yspeed": -1,
        "zspeed": -1,
    }
    if with_gps:
        meta |= {"gps_latitude": 47.67, "gps_longitude": 9.27}
    return meta


def video_source(drone, folder, frame_no):
    return {"drone": drone, "folder_name": folder, "video": f"{folder}.MP4", "frame_no": frame_no}


def image(image_id, width, height, source, meta, **extra):
    return {
        "id": image_id,
        "file_name": f"{image_id}.jpg",
        "height": height,
        "width": width,
        "source": source,
        "meta": meta,
        **extra,
    }


IMAGES = [
    image(
        1,
        3840,
        2160,
        video_source("mavic", "DJI_0032", 7),
        dji_meta(),
        date_time="2020-08-25T14:16:21.467133",
    ),
    image(
        2,
        1920,
        1080,
        video_source("m210", "NorthSea_Sat_Z30_DJI003", 0),
        {},
        date_time="2021-08-21T14:40:09.973333",
    ),
    image(3, 3632, 5456, {"drone": "trinity", "folder_name": "rgb"}, {}, frame="DSC01061.jpg"),
    image(
        4,
        1231,
        933,
        {"drone": "trinity", "folder_name": "multispectral"},
        micasense_meta(),
        frame="IMG_0317.png",
    ),
    image(
        5,
        1230,
        932,
        {"drone": "trinity", "folder_name": "multispectral"},
        micasense_meta(with_gps=False),
        frame="IMG_0318.png",
    ),
    image(
        6,
        3840,
        2160,
        video_source("m210", "DJI_0003", 3),
        dji_meta(altitude_m=0.0),
        date_time="2020-08-25T15:23:22",
    ),
]

ANNOTATIONS = [
    {"id": 10, "image_id": 1, "bbox": [3619, 1409, 75, 38], "area": 2850, "category_id": 1},
    {"id": 11, "image_id": 1, "bbox": [100.5, 200, 10, 12], "area": 120, "category_id": 1},
    {"id": 12, "image_id": 1, "bbox": [50, 60, 300, 120], "area": 36000, "category_id": 2},
    {"id": 13, "image_id": 4, "bbox": [5, 6, 7, 8], "area": 56, "category_id": 5},
    {"id": 14, "image_id": 2, "bbox": [1, 2, 3, 4], "area": 12, "category_id": 0},
]


def write_coco(path, images, annotations=(), categories=CATEGORIES):
    coco = {
        "info": {"version": "2"},
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": list(annotations),
    }
    path.write_text(json.dumps(coco), encoding="utf-8")
    return path


@pytest.fixture
def parsed(tmp_path):
    train = write_coco(tmp_path / "train.json", IMAGES[:4], ANNOTATIONS)
    val = write_coco(tmp_path / "val.json", IMAGES[4:])
    frames, boxes = load_odv2({"train": train, "val": val}, GROUPS)
    return {f.image_id: f for f in frames}, boxes


def test_dji_flight_log_frame(parsed):
    frame = parsed[0][1]
    assert frame.camera_group == "mavic_4k"
    assert frame.official_split == "train"
    assert (frame.drone, frame.source_folder, frame.source_frame_no) == ("mavic", "DJI_0032", 7)
    assert frame.timestamp == datetime(2020, 8, 25, 14, 16, 21, 467133)
    assert frame.timestamp_source == "image"
    assert frame.altitude_m == 35.2
    assert frame.gimbal_pitch_deg == 85.0
    assert frame.off_nadir_deg == pytest.approx(5.0)
    assert (frame.latitude_deg, frame.longitude_deg) == (47.673596, 9.270249)
    assert (frame.n_swimmer, frame.n_boat, frame.n_buoy, frame.n_ignored) == (2, 1, 0, 0)


def test_micasense_frame_uses_meta_time_and_gps_refs(parsed):
    frame = parsed[0][4]
    assert frame.camera_group == "trinity_multispectral"
    assert frame.timestamp == datetime(2020, 8, 27, 13, 3, 39)
    assert frame.timestamp_source == "meta"
    assert (frame.altitude_m, frame.gimbal_pitch_deg) == (110.5, 90.0)
    assert (frame.latitude_deg, frame.longitude_deg) == (-47.67, -9.27)
    assert frame.n_buoy == 1


def test_missing_gps_and_nonpositive_altitude_become_none(parsed):
    frames = parsed[0]
    assert (frames[5].latitude_deg, frames[5].longitude_deg) == (None, None)
    assert frames[5].official_split == "val"
    assert frames[6].camera_group == "m210_4k"
    assert frames[6].altitude_m is None


def test_frames_without_meta(parsed):
    frames = parsed[0]
    assert frames[2].camera_group == "m210_1080p"
    assert (frames[2].altitude_m, frames[2].gimbal_pitch_deg, frames[2].off_nadir_deg) == (
        None,
        None,
        None,
    )
    assert frames[2].n_ignored == 1
    assert frames[3].camera_group == "trinity_rgb"
    assert (frames[3].timestamp, frames[3].timestamp_source, frames[3].source_frame_no) == (
        None,
        None,
        None,
    )


def test_boxes_keep_coco_geometry(parsed):
    boxes = {b.annotation_id: b for b in parsed[1]}
    assert len(boxes) == len(ANNOTATIONS)
    assert boxes[10] == Box(10, 1, "swimmer", 3619.0, 1409.0, 75.0, 38.0)
    assert boxes[11].x_px == 100.5
    assert boxes[14].category == "ignored"


def test_unknown_meta_schema_is_an_error(tmp_path):
    bad = image(1, 3840, 2160, video_source("mavic", "DJI_0032", 0), {"height": 3.0})
    with pytest.raises(ValueError, match="unrecognised meta"):
        load_odv2({"train": write_coco(tmp_path / "a.json", [bad])}, GROUPS)


@pytest.mark.parametrize(("drone", "width", "height"), [("mavic", 1920, 1080), ("dji", 3840, 2160)])
def test_image_outside_every_camera_group_is_an_error(tmp_path, drone, width, height):
    bad = image(1, width, height, video_source(drone, "X", 0), {})
    with pytest.raises(ValueError, match="matches camera groups"):
        load_odv2({"train": write_coco(tmp_path / "a.json", [bad])}, GROUPS)


def test_image_id_in_two_splits_is_an_error(tmp_path):
    files = {s: write_coco(tmp_path / f"{s}.json", IMAGES[:1]) for s in ("train", "val")}
    with pytest.raises(ValueError, match="more than once"):
        load_odv2(files, GROUPS)


def test_unknown_category_is_an_error(tmp_path):
    categories = [*CATEGORIES, {"supercategory": "x", "id": 9, "name": "kayak"}]
    path = write_coco(tmp_path / "a.json", IMAGES[:1], categories=categories)
    with pytest.raises(ValueError, match="kayak"):
        load_odv2({"train": path}, GROUPS)
