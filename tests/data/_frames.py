"""A template frame for tests; vary it with ``dataclasses.replace``."""

from datetime import datetime

from altiseek.data.frames import Frame

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
