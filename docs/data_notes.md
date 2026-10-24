# SeaDronesSee ODv2: verified data notes

Everything here was checked on 2026-10-24 against the downloaded files, unless a line says
otherwise. Counts cover the labelled official splits (train + val), which
`scripts/prepare_data.py` turns into `data/processed/frames.csv` and `boxes.csv`.

## Source and licence

- Download: the "Object Detection v2" button on <https://seadronessee.cs.uni-tuebingen.de/dataset>
  POSTs to `/dataset/ODv2_redirecter`, which redirects to the public Nextcloud share
  `https://cloud.cs.uni-tuebingen.de/index.php/s/ZZxX65FGnQ8zjBP`. The share can be read over WebDAV
  (`https://cloud.cs.uni-tuebingen.de/public.php/webdav/`, user = share token, empty password).
- **Licence:** "Code is MIT license, dataset is Creative Commons CC0 1.0 Universal (CC0 1.0)"
  ([Ben93kie/SeaDronesSee README](https://github.com/Ben93kie/SeaDronesSee)). Derived artifacts may
  be published; images still never go into git (CLAUDE.md §2).
- Share contents:

  | Path | Size |
  |---|---|
  | `Compressed Version/images/{train,val,test}` (JPEG) | 6.34 GB, 1.07 GB, 2.54 GB |
  | `Compressed Version/annotations/instances_{train,val}.json` | 10.0 MB, 1.7 MB |
  | `Uncompressed Version/images/{train,val,test}` (PNG) | 78.9 GB, 13.6 GB, 33.5 GB |
  | `Uncompressed Version/annotations/instances_{train,val,test_nogt}.json` | 10.0 MB, 1.7 MB, 1.4 MB |
  | `Readme.md`, `sample_submission_odv2.json` | |

- The share's `Readme.md` says test annotations are withheld, and that test-set metadata is only in
  the uncompressed version.
- The compressed and uncompressed train/val annotation files are **identical except
  `file_name`** (`<id>.jpg` vs `<id>.png`). Compressed JPEGs keep full resolution (3840×2160 checked
  on 5 frames).
- COCO `info`: year 2022, version "2", date_created 2022-08-15.
- SHA-256 of the annotation files:

  | File | SHA-256 |
  |---|---|
  | compressed/instances_train.json | `5706b042f6b6c50267ed06076b8c27594ff4da76569edbd99ed53ed8bfa1db95` |
  | compressed/instances_val.json | `9e965c832263e6b9b9dd299637833e5b93924ecf9a1780b16fbd7c53e4a63f5e` |
  | uncompressed/instances_train.json | `d7309081bdd84975f1783279b4f02bfff829d8effa6e2a4857630f82c982a75c` |
  | uncompressed/instances_val.json | `9d64e077cc8c5d8c53e8f106ec292123300a6d93fed7f1c5c1464b600f9ce6a9` |
  | uncompressed/instances_test_nogt.json | `d9ee6d4c4da466ac3ab7555807ff8573cb3e7302e4fd2d218d7e5391c09514e5` |

## Counts and categories

- train: 8,930 images, 57,760 boxes. val: 1,547 images, 9,630 boxes. test: 3,750 images, no boxes.
  No image id appears in both train and val.
- Categories (id: name): 0 `ignored`, 1 `swimmer`, 2 `boat`, 3 `jetski`, 4 `life_saving_appliances`,
  5 `buoy`.

  | Category | train boxes | val boxes |
  |---|---|---|
  | swimmer | 37,096 | 6,206 |
  | boat | 13,022 | 2,214 |
  | jetski | 2,330 | 320 |
  | life_saving_appliances | 923 | 330 |
  | buoy | 4,389 | 560 |
  | ignored | 0 | 0 |

- Annotation keys: `id`, `image_id`, `bbox`, `area`, `category_id`. There is no `iscrowd` or ignore
  flag. `bbox` is COCO `[x, y, w, h]` in pixels from the top-left corner.
- **Ignored regions are blackened in the pixels**, not annotated. The paper says ignored regions such
  as boats on land were annotated, and that unwanted regions can be removed "by blackening"
  (Varga et al. 2022, §3.2–4). In ODv2 no box uses category 0, and image 3885 shows solid black
  rectangles. `has_ignore_regions` therefore has to come from a pass over the pixels (not done yet).
- The paper's ODv1 classes (swimmer, floater, life jacket, swimmer†, floater†, boat) differ from
  ODv2's. ODv2's `swimmer` presumably merges people in water with and without life jackets; this is
  unverified.

## Image records

- Image keys: `id`, `file_name`, `width`, `height`, `source`, `meta` (always present, possibly
  empty), `date_time` (7,376 train / 1,288 val) and `frame` (Trinity images only: the original file
  name, e.g. `DSC01420.JPG`, `IMG_0317.png`).
- `source` is `{drone, folder_name, video, frame_no}` in train/val and a bare drone string in test.
  `folder_name` identifies the video (or `rgb` / `multispectral` for the Trinity stills).
- Image-level `date_time` is **local time** with sub-second precision. For example image 3388 has
  `date_time` 14:16:21 and `meta.datetime(utc)` 12:16:22, i.e. CEST = UTC+2. Consecutive extracted
  frames are about 0.5 s apart; the paper says at most 3 fps were extracted from 30 fps video.

## Metadata schemas

Two `meta` schemas exist; every other image has `meta == {}`.

**DJI flight log** (all `mavic_4k` and `m210_4k` frames). Keys: `image_name`, `datetime(utc)`,
`latitude`, `longitude`, `height_above_takeoff(meter)`, `speed(m/s)`, `xspeed(m/s)`, `yspeed(m/s)`,
`zspeed(m/s)`, `compass_heading(degrees)`, `gimbal_heading(degrees)`, `gimbal_pitch(degrees)`.
`image_name` names the flight-log record (consecutive frames can share one), and `datetime(utc)` has
1 s resolution. The paper says metadata is logged at 10 Hz from the barometer, IMU, GPS and gimbal
and matched to frames by nearest neighbour.

**MicaSense** (`trinity_multispectral`). Keys: `date_time`, `gps_latitude_ref`,
`gps_longitude_ref`, `altitude`, `gimbal_pitch`, `speed`, `xspeed`, `yspeed`, `zspeed` (always −1,
i.e. missing). `gps_latitude` and `gps_longitude` are present on only 208 of 475 frames; the refs
are NaN when GPS is missing.

- **Altitude** is barometric height above take-off, in metres. There are no zero or negative values
  (minimum 4.9 m); the parser still treats non-finite or ≤ 0 values as missing.
- **Gimbal pitch** is degrees below the horizon: "α = 90° corresponds to a top-down view, and α = 0°
  to a horizontally facing camera" (paper §3.1). This was confirmed visually: frame 5248 (89.9°, 91 m)
  is top-down, 4472 (45.4°, 9 m) is oblique, and 3885 (1.6°, 17 m) shows the horizon. A few values
  are slightly negative (minimum −7.5°), i.e. looking just above the horizon.

## Camera groups

Assigned by drone and image size (`configs/data/default.yaml`). Cameras come from the paper's
Table 2 unless noted otherwise.

| Group | Drone, size | Camera | Frames train / val | Metadata | Altitude | Swimmer boxes |
|---|---|---|---|---|---|---|
| `mavic_4k` | mavic, 3840×2160 | Hasselblad L1D-20c (Mavic 2 Pro), 30 fps video | 5,122 / 937 | DJI | 5.6–132.0 m | 25,400 |
| `m210_4k` | m210, 3840×2160 | not recorded per frame (paper lists Zenmuse X5 and XT2 at this size) | 678 / 93 | DJI | 4.9–61.0 m | 1,276 |
| `m210_1080p` | m210, 1920×1080 | Zenmuse Z30 zoom (from folder names); not in the paper | 1,576 / 258 | none | — | 13,273 |
| `trinity_rgb` | trinity, 5456×3632 and 3632×5456 | Sony UMC-R10C stills (Trinity F90+) | 1,142 / 196 | none, no time | — | 2,612 |
| `trinity_multispectral` | trinity, 1225–1234 × 926–935 | MicaSense RedEdge-MX, RGB channels only | 412 / 63 | MicaSense | 52.6–259.4 m | 741 |

- Only `mavic_4k`, `m210_4k` and `trinity_multispectral` have altitude and pitch.
- `trinity_multispectral` reports `gimbal_pitch` 90.0 on every frame (a fixed nadir mount). The 267
  frames without GPS are dated 2021-08-27, while the rest of the group is dated 2020-08-27/28; its
  camera clock is probably unreliable.
- Horizontal FOVs are not in the files. They are measured below for `mavic_4k` only.

## GSD model (CLAUDE.md §7 step 4)

`scripts/fit_camera.py` writes `data/processed/camera_models.json` and per-frame detections to
`data/processed/horizon_samples.csv`, and draws [docs/figures/horizon_fit_mavic_4k.png](figures/horizon_fit_mavic_4k.png).

- **Method.** Box sizes cannot give the FOV: a box's pixel size is the object's unknown physical
  size times the focal length over the distance. The open-sea horizon has no such ambiguity. A
  camera pitched down by α at height h sees the horizon f·tan(α + b − dip(h)) pixels above the
  centre, where b is a constant gimbal pitch bias. A robust (Theil–Sen) line through
  (tan(α − dip), offset) gives f as its slope and b as intercept / f. CIs come from a bootstrap
  over videos.
- **Frames used:**
  - `mavic_4k` on the North Sea only; the Lake Constance horizons are shorelines.
  - Logged pitch 1–17°.
  - Pitch steady within 0.3° for ±2 s. Metadata is nearest-neighbour matched at 10 Hz, so a moving
    gimbal logs a lagging pitch; one frame mid-sweep was 5° off.
  - At most 15 frames per video.
- **Horizon detection:**
  - Blackened regions plus an 8 px JPEG halo are masked.
  - At least 8 of 32 strips must agree, and they must be at least 75% of the strips that can see
    the line.
  - |roll| ≤ 3°.
- **Result:** 118 frames examined, 91 accepted, 8 videos on 3 days.

  | Quantity | Estimate | 95% CI |
  |---|---|---|
  | Focal length f | 3029 px | 2932–3225 px |
  | HFOV at 3840 px | 64.7° | 61.5–66.4° |
  | Gimbal pitch bias b | −0.38° | −1.21 to −0.11° |
  | Residual MAD | 8.5 px | |

- **Nadir GSD = altitude × 3.30 × 10⁻⁴ m/px** (k = 1/f; CI 3.10–3.41 × 10⁻⁴). That is 0.99 cm/px at
  30 m and 3.3 cm/px at 100 m.
- **Consistency.** The zero-bias median f per video is 2738–3089 px on all three days, so the
  camera mode did not change. (`NorthSea_Fri_DJI_0110` is left out of that range: at 1.6° pitch its
  zero-bias estimate is dominated by the bias.)
- **Mode check.** The Mavic 2 Pro records 4K either from the full sensor width or as an HQ crop.
  DJI's published lens data (28 mm equivalent, about 77° diagonal) gives f ≈ 2900–2985 px for full
  width and ≈ 4130–4250 px for the crop. These spec numbers are derived, not read from the files.
  The fit confirms the full-width mode.
- Lens distortion is ignored; detection uses the central 80% of the width.
- **Not modelled yet:**
  - `m210_4k`: no open-sea footage. The paper lists Zenmuse X5 and XT2 at 3840×2160.
  - `trinity_multispectral`: the manufacturer's spec for the RedEdge-MX is 47.2° HFOV, unverified
    here.
  - `m210_1080p` and `trinity_rgb`: no altitude.

## Capture days and videos

| Date | Site (median GPS) | Frames by group | Swimmer boxes |
|---|---|---|---|
| 2020-08-25 | Lake Constance (47.673 N, 9.269 E) | mavic_4k 270, m210_4k 75 | 315 |
| 2020-08-27 | Lake Constance | mavic_4k 492, m210_4k 696, multispectral 186 | 3,612 |
| 2020-08-28 | Lake Constance | mavic_4k 290, multispectral 22 | 1,595 |
| 2021-08-19 | North Sea (53.722 N, 7.164 E) | mavic_4k 1,041 | 1,615 |
| 2021-08-20 | North Sea | mavic_4k 2,446 | 10,774 |
| 2021-08-21 | North Sea | mavic_4k 1,520, m210_1080p 1,834 | 22,457 |
| 2021-08-27 (suspect clock) | no GPS | multispectral 267 | 322 |
| none | — | trinity_rgb 1,338 | 2,612 |

- The 4K groups hold 6,830 frames from 35 videos. Within a day, the Mavic and the M210 flew
  interleaved (for example on 2020-08-27), so both filmed the same exercise.
- **The official train/val split is not session-disjoint.** 30 of the 35 videos have frames in both
  (e.g. `NorthSea_Fri_DJI_0112`: 353 train, 60 val), which confirms the concern in CLAUDE.md §7.
- Folder names are not unique on their own: m210 `DJI_0001` (2020-08-25) and `DJI_0001_d3`
  (2020-08-27) are different videos.

## Nadir frames are scarce

Most quadcopter footage is oblique: 4K pitch has a median of about 31° (off-nadir about 59°).
Frames within a tolerance of nadir (off-nadir = 90° − pitch):

| Tolerance | `mavic_4k` frames / swimmer boxes / videos | `m210_4k` frames / swimmer boxes |
|---|---|---|
| ±5° | 109 / 217 / 2 | 0 / 0 |
| ±10° | 296 / 903 / 3 | 24 / 0 |
| ±15° | 348 / 1,081 / 6 | 26 / 0 |
| ±20° | 589 / 2,331 / 9 | 51 / 23 |
| ±30° | 802 / 3,343 / 10 | 59 / 29 |

At the default ±10° (CLAUDE.md §7 step 3), nadir 4K frames with swimmers come from only three videos
on three days: `DJI_0064` (2020-08-27, 19 frames at about 91 m), `NorthSea_Thu_DJI_0107`
(2021-08-19) and `NorthSea_Sat_DJI_0131` (2021-08-21). A session-disjoint A/B/T split restricted to
these frames is not workable. This is open decision "Oblique frames" in CLAUDE.md §25.

A tilted frame does not have one GSD. With f = 3029 px, the along-view GSD from the bottom row to
the top row of a frame tilted off-nadir by the tolerance, relative to the nadir value
(`altiseek.geometry.camera.pixel_gsd_mpp`):

| Off-nadir | Bottom row | Centre | Top row |
|---|---|---|---|
| 10° | 0.91 | 1.03 | 1.17 |
| 20° | 0.89 | 1.13 | 1.50 |
| 30° | 0.92 | 1.33 | 2.11 |

**Resolution (D14):** oblique frames are used through ground-plane rectification into
nadir-equivalent views, with per-pixel GSD from the tilted camera model. `mavic_4k` frames per
off-nadir band (90° − pitch, at the image centre) and split, as frames / swimmer boxes:

| Off-nadir | A | B | T | All frames |
|---|---|---|---|---|
| 0–10° | 19 / 38 | 161 / 634 | 116 / 231 | 296 |
| 10–20° | 129 / 375 | 134 / 996 | 30 / 57 | 293 |
| 20–30° | 94 / 475 | 70 / 505 | 49 / 32 | 213 |
| 30–45° | 326 / 1,347 | 200 / 1,792 | 41 / 72 | 567 |
| 45–60° | 1,285 / 6,435 | 485 / 3,366 | 277 / 546 | 2,047 |
| 60–75° | 954 / 5,587 | 276 / 1,587 | 407 / 596 | 1,637 |
| 75–90° | 560 / 342 | 151 / 295 | 115 / 80 | 826 |
| 90–100° | 131 / 2 | 43 / 9 | 6 / 1 | 180 |

Frames beyond 90° look above the horizon. Most of the footage is 45–75° off nadir, so θ_train (M2)
decides how much of it the detector sees.

## Sessions and splits

`scripts/make_splits.py` writes `data/processed/splits.csv`, `sessions.csv` and `split_tables.md`.
The `split_id` is **783b4403804291a2**.

- **Sessions (D15):** one per capture day across all camera groups, plus undated sessions for
  frames without a trustworthy day. Both undated sessions are assigned to the 2020 Lake Constance
  campaign.

  | Session | Campaign | Frames | Split |
  |---|---|---|---|
  | 2020-08-25 | lake_constance_2020 | 345 | A |
  | 2020-08-27 | lake_constance_2020 | 1,374 | A |
  | 2020-08-28 | lake_constance_2020 | 312 | A |
  | 2021-08-19 | north_sea_2021 | 1,041 | **T** |
  | 2021-08-20 | north_sea_2021 | 2,446 | A |
  | 2021-08-21 | north_sea_2021 | 3,354 | **B** |
  | undated:trinity_multispectral | lake_constance_2020 | 267 | A |
  | undated:trinity_rgb | lake_constance_2020 | 1,338 | A |

- **Assignment (D16):**
  - The search is exhaustive over 4 blocks (81 assignments; 12 feasible).
  - Targets are A/B/T = 0.6/0.25/0.15 of `mavic_4k` frames, stratified by altitude band ×
    swimmer presence.
  - Constraints: B and T come only from `north_sea_2021`, and each has ≥ 100 near-nadir swimmer
    boxes.
  - Loss 0.101; the runner-up has 0.117.
  - Achieved `mavic_4k` shares: A 57.7%, B 25.1%, T 17.2%.
- **Counts:** `mavic_4k` frames and boxes per split × altitude band × class:

  | Split | Altitude | Frames | With swimmers | swimmer | boat | jetski | life_saving_appliances | buoy |
  |---|---|---|---|---|---|---|---|---|
  | A | 0–20 m | 827 | 333 | 969 | 1,787 | 55 | 4 | 677 |
  | A | 20–40 m | 1,305 | 1,280 | 6,711 | 2,099 | 427 | 0 | 1,031 |
  | A | 40–60 m | 1,111 | 1,063 | 5,919 | 2,539 | 443 | 64 | 914 |
  | A | 60–200 m | 255 | 165 | 1,002 | 858 | 146 | 4 | 302 |
  | **A** | all | 3,498 | 2,841 | 14,601 | 7,283 | 1,071 | 72 | 2,924 |
  | B | 0–20 m | 231 | 197 | 1,792 | 272 | 116 | 0 | 40 |
  | B | 20–40 m | 697 | 660 | 4,747 | 991 | 386 | 613 | 362 |
  | B | 40–60 m | 335 | 196 | 1,039 | 693 | 156 | 151 | 205 |
  | B | 60–200 m | 257 | 257 | 1,606 | 421 | 120 | 231 | 91 |
  | **B** | all | 1,520 | 1,310 | 9,184 | 2,377 | 778 | 995 | 698 |
  | T | 0–20 m | 745 | 578 | 1,151 | 77 | 345 | 0 | 79 |
  | T | 20–40 m | 142 | 84 | 168 | 27 | 35 | 1 | 11 |
  | T | 40–60 m | 72 | 68 | 132 | 3 | 0 | 0 | 5 |
  | T | 60–200 m | 82 | 82 | 164 | 0 | 0 | 0 | 0 |
  | **T** | all | 1,041 | 812 | 1,615 | 107 | 380 | 1 | 95 |

  Near-nadir (≤ 30°) `mavic_4k` swimmer boxes: A 888, B 2,135, T 320. All frames by group: A has
  every `m210_4k`, `trinity_rgb` and `trinity_multispectral` frame; B has every `m210_1080p` frame.
- **Caveats:**
  - Only six capture days exist, so the splits are coarse. The same volunteers may appear on
    several days of a campaign; day-level splitting removes shared scenes, not shared people.
  - T is dominated by low altitudes (72% of its frames are below 20 m) and has almost no
    distractors. Distractor-heavy test scenarios will reuse a small set of T distractor crops.
