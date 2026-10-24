# CLAUDE.md — AltiSeek

**Learning where — and how high — to look for people at sea.** A simulated drone searches open water for people within a fixed battery budget. A detector trained on real drone footage turns each simulated camera view into evidence. An explicit, altitude-aware sensor model turns that evidence into a probability map. A policy (classical baselines vs. PPO) then decides where to fly next and at what altitude. Simulation only; no hardware.

This file is the source of truth for the project's goals, design, data contracts and working rules. Read it fully before writing or changing code. When the implementation has to deviate, update the relevant section and add an entry to the Decision Log (§24) in the same change.

> This file is long on purpose. If it gets heavy for session context, move §7–§17 into `docs/spec/*.md` and keep this file as the index plus §0, §2, §5 and §21–§25.

---

## 0. Project status

Update as milestones complete (§21).

| Milestone | Status |
|---|---|
| M0 Repo & tooling | done (2026-10-24) |
| M1 Data | in progress: frame table, camera groups and the `mavic_4k` GSD model done; sessions, splits and leakage check wait on §25 decisions |
| M2 Detector | not started |
| M3 Level design | not started |
| M4 Worlds & renderer | not started |
| M5 Detection cache | not started |
| M6 Sensor models | not started |
| M7 Environment & baselines | not started |
| M8 PPO | not started |
| M9 Main experiments | not started |
| M10 PX4 / Gazebo / ROS 2 | not started |
| M11 Write-up & polish | not started |

---

## 1. What this project is

The core tradeoff is altitude:
- **High:** each image covers far more sea, but a person shrinks to a few pixels and the detector misses most of them.
- **Low:** detection is reliable, but covering the area costs far more battery.

The pipeline:
1. **Detector.** A small object detector is fine-tuned on SeaDronesSee (real drone footage of people in open water, with altitude metadata). It is trained to work at the image resolutions the simulated camera produces.
2. **Worlds.** Search worlds are composed from real sea imagery and real object crops, so every pixel the detector sees in simulation is real imagery.
3. **Detection cache.** Detector outputs for every reachable camera view are precomputed, so RL training never runs the detector.
4. **Sensor model.** Recall and false-alarm rates per altitude, plus calibrated scores, turn detections (and non-detections) into Bayesian updates of a per-cell probability map.
5. **Policy.** A policy reads the map and chooses the next move: a horizontal step, a climb or a descent.
6. **Deployment.** The same pipeline runs closed-loop on PX4 SITL + Gazebo through ROS 2, with the detector running live on the simulated downward camera.

CV and RL are coupled through the belief map. The policy never sees pixels, but everything it sees was produced by the detector and the sensor model. Detector quality and calibration therefore directly change search behaviour. Measuring that coupling is RQ1.

---

## 2. Constraints, scope and non-goals

**Hard constraints**
- **No physical hardware.** Everything runs in simulation or on public datasets. Never claim real-world performance; the strongest transfer claim is Python simulator → PX4/Gazebo.
- **Compute.**
  - Development happens on Windows; WSL2 (Ubuntu) is used for everything ROS 2 / PX4 / Gazebo.
  - Assume no local GPU. Detector training and bulk detector inference (cache generation) run on free cloud GPU notebooks (Kaggle/Colab).
  - Those sessions have limits and can disconnect, so every long job must be resumable from checkpoints.
  - RL training and all evaluation must run on CPU.
- **Data.**
  - SeaDronesSee only, unless something else is added via a Decision Log entry.
  - Never commit dataset images, crops, rendered worlds, caches or checkpoints to git.
  - Check the dataset's terms of use before publishing anything derived from it.

**In scope**
- Detector fine-tuning and evaluation as a function of ground sample distance (GSD).
- Calibration and explicit sensor models.
- Procedurally generated search worlds built from real imagery.
- A Gymnasium search environment backed by cached detections.
- Random, oracle, classical-baseline and PPO policies.
- A frozen, leakage-safe evaluation protocol.
- A PX4/Gazebo/ROS 2 closed-loop demonstration and transfer study.

**Non-goals** (do not drift into these without a Decision Log entry)
- Learning low-level flight control. PX4 flies; policies choose high-level moves.
- Oblique or gimballed cameras. The simulated camera is nadir with fixed yaw. Oblique *source* frames are used only after ground-plane rectification into nadir-equivalent views (D14).
- Targets moving during an episode.
- Multiple drones.
- End-to-end RL from pixels.
- Real-world flight tests.

---

## 3. Research questions and hypotheses

**RQ1 — Does sensor-model fidelity matter for search?**
Every policy is run with four ways of turning detections into belief updates (V0–V3, §10).
- **H1a:** Altitude-aware, calibrated sensor models (V2, V3) find more people per unit battery than a single global model (V1).
- **H1b:** The naive model (V0) produces two measurable failure modes. V0 treats raw detector scores as probabilities and assumes a fixed, altitude-independent recall for negative evidence.
  - *False clears:* cells containing people are driven to low probability after being seen only from high altitude.
  - *False declarations:* overconfident scores trigger declarations on empty cells.
- **Falsified if** V0/V1 fall within the confidence intervals of V2/V3 across scenario families.

**RQ2 — When does a learned policy beat strong heuristics that use identical sensing?**
- **H2:** PPO beats the best tuned heuristic mainly when:
  - (a) the battery is tight relative to the area;
  - (b) the prior is structured (elongated drift, clustered groups);
  - (c) distractor density is high.
  On easy, uniform scenarios it roughly matches the heuristics.
- **Falsified if** the best heuristic matches or beats PPO in every family.
- A negative result is still a result. Report it honestly, and never tune on test worlds to rescue it.

**RQ3 — Does the policy survive the move from cached detections to a live detector in PX4/Gazebo?**
- **H3a:** Performance drops modestly on matched scenarios, and the failures trace to rendering differences (sensor-model shift) and to pose/altitude error.
- **H3b:** Re-fitting the sensor model on Gazebo-rendered views recovers most of the gap — a direct benefit of explicit sensor modelling.

---

## 4. System overview

**Offline stages**
1. **Data:** SeaDronesSee frames + metadata → frame table, camera groups, sessions, session-disjoint splits A/B/T, native GSD per frame.
2. **Detector:** split A real frames, resampled to target GSDs → detector checkpoint (`detector_id`).
3. **Level design:** detector recall-vs-GSD curve → camera spec, altitude levels, cell size, battery.
4. **Libraries & worlds:** split B and split T imagery → background and object-crop libraries per split → world specs for each world set.
5. **Cache:** world spec + camera/level spec → renderer → views → detector (batched on GPU) → detection cache per (world, level, view, variant).
6. **Sensor models:** caches of the sensor-fit worlds → calibration and sensor models V0–V3 (`sensor_model_id`, tied to `detector_id`).

**Online loop (Python simulator)**
7. **Environment:** scenario (ground truth) + cache + sensor model → belief map → observation → policy → action → energy/time model → reward.
8. **Evaluation:** fixed scenario seeds × methods × sensor-model variants → episode logs → metrics, statistics, figures.

**Deployment**
9. **PX4/Gazebo/ROS 2:** world spec → textured Gazebo world; PX4 SITL + ROS 2 nodes (camera → live detector → belief → policy → offboard setpoints) → episode logs → RQ3 comparison.

---

## 5. Conventions

**Units and naming**
- SI units everywhere. Suffix names with units when ambiguous: `_m`, `_s`, `_px`, `_deg`, `_rad`, `_mpp` (metres per pixel).
- Cell indices are integers named `ix`, `iy`. Never mix cell indices and metres in the same variable.
- Docstrings state units and frames.

**Frames** (one implementation, in `src/altiseek/geometry/`, used everywhere)
- **World frame:** origin at the search area's south-west corner; x = east, y = north, z = up; metres.
- **Grid:** cell (ix, iy) covers x ∈ [ix·c, (ix+1)·c) and y ∈ [iy·c, (iy+1)·c), for cell size c.
- **Image:** rows increase downward (southward in a north-up nadir view); columns increase eastward. Converting row → y needs a flip. Unit-test it.
- **PX4:** local frame NED (x north, y east, z down), body frame FRD.
- **ROS:** ENU, body frame FLU.
- **Conversion:** ENU.x = NED.y, ENU.y = NED.x, ENU.z = −NED.z; yaw_ENU = π/2 − yaw_NED. Unit-test it.
- Yaw is held fixed (camera north-up) in every simulator, so footprints stay axis-aligned with the grid.

**IDs** (content-addressed; see §18)
- `detector_id` = hash of the checkpoint file.
- `sensor_model_id` = hash of (detector_id, fit config, fit world-set ID).
- `world_id` = hash of the world spec.
- `cache_id` = hash of (world_id, detector_id, camera spec, level spec, variant spec).
- Every artifact records the IDs it was built from.
- Loaders must refuse mismatches, for example a cache built by a different detector.

**Randomness**
- Every stochastic component takes an explicit seed.
- Evaluation scenario seeds are fixed lists stored in `configs/eval/`.
- A given scenario seed must produce identical worlds, targets and variant draws for every method.

**Code style**
- Type hints throughout; a formatter and linter enforced in tests or pre-commit.
- One Python version across WSL2, cloud notebooks and local runs, recorded in `docs/env.md`.

---

## 6. Repository layout

```text
altiseek/
├── CLAUDE.md
├── README.md
├── pyproject.toml              pinned dependencies (single source)
├── uv.lock                     generated from pyproject.toml; pins transitive dependencies
├── .github/workflows/ci.yml    lint, format and tests on Ubuntu and Windows
├── configs/
│   ├── data/                   split ratios, nadir tolerance, camera groups
│   ├── detector/               model, input size, GSD sampling, augmentations
│   ├── camera/                 resolution, HFOV, resampling, camera-pipeline noise
│   ├── levels/                 level table (footprints, strides)
│   ├── world/                  area, cell size, canonical GSD, scenario families
│   ├── sensor/                 thresholds, variant settings, re-look rule
│   ├── env/                    energy model, rewards, observation options
│   ├── policy/                 baseline hyperparameters, PPO settings
│   ├── eval/                   protocol versions, fixed scenario seed lists
│   └── deploy/                 Gazebo / ROS 2 / PX4 parameters
├── docs/
│   ├── data_notes.md           verified dataset facts (field names, cameras, splits)
│   ├── env.md                  exact tool versions and setup notes
│   └── results.md              result tables and figure index
├── src/altiseek/
│   ├── config.py               hierarchical YAML configs + schema validation
│   ├── geometry/               frames, footprints, GSD, conversions
│   ├── data/                   frame table, sessions, splits, leakage check
│   ├── imaging/                shared resampling + camera pipeline
│   ├── detector/               training-view generation, training wrapper, inference API, evaluation
│   ├── calibration/            cell-level matching, calibrators, sensor models V0–V3
│   ├── world/                  libraries, scenario sampler, world spec, renderer
│   ├── cache/                  view lattice, batch generation, storage, validation
│   ├── belief/                 belief state, updates, re-look rule, derived quantities
│   ├── env/                    Gymnasium env, energy model, rewards, observation builder, masks
│   ├── policies/               random, oracle, B1–B5, PPO training and inference
│   ├── eval/                   protocol runner, metrics, statistics
│   ├── viz/                    episode replays, figures
│   └── registry/               artifact IDs, manifest, run metadata
├── ros2_ws/src/altiseek_ros/   ROS 2 nodes, launch files, Gazebo world exporter and assets
├── notebooks/                  thin cloud-GPU notebooks that call src/ (detector training, cache generation)
├── scripts/                    one CLI entry point per stage
├── tests/
├── data/                       gitignored: raw, processed, libraries, worlds, caches, sensor_models
└── runs/                       gitignored: training and evaluation runs
```

**Stage entry points in `scripts/`**, in pipeline order:
1. prepare-data, fit-camera (focal length from the sea horizon, §7 step 4), make-splits, scan-images (one streaming pass over every image: hashes, ignore regions), check-leakage
2. train-detector, eval-detector
3. design-levels
4. build-libraries, generate-worlds
5. build-cache, validate-cache
6. fit-sensor-models
7. run-baselines, train-ppo
8. evaluate, make-figures
9. export-gazebo-world

Each entry point reads a config, writes its outputs plus a metadata record (§18), and is resumable where it runs long.

---

## 7. Data pipeline (SeaDronesSee)

**Reported facts to verify on download**

Record verified values, exact metadata field names and counts in `docs/data_notes.md`. Never hard-code a field name you haven't seen in the files.

- Object Detection v2 reportedly has about 8.9k train, 1.5k val and 3.7k test images. Only train and val are labelled.
- Classes are reported as swimmer, boat, jetski, life_saving_appliances and buoy. Check for ignore categories or ignore regions.
- Per-image metadata is reported to include altitude, UAV and gimbal angles, GPS, speed and time. An altitude of 0 may mean "missing".
- Images are reported to come from several cameras:
  - 3840×2160 video frames from a quadcopter;
  - 5456×3632 stills from a fixed-wing drone, reportedly often without altitude;
  - roughly 1230×930 multispectral images.

**Steps**
1. **Frame table.** One row per image: image_id, path, width, height, camera_group, altitude_m, gimbal_pitch_deg, GPS, timestamp, per-class annotation counts. `has_ignore_regions` comes from the image scan (`image_scan/stats.csv`), because ODv2 blackens ignore regions in the pixels instead of annotating them.
2. **Camera groups.**
   - Assign by resolution, plus metadata where available.
   - Use only groups with reliable altitude for anything that needs GSD.
   - Default: the 3840×2160 group only. In practice only `mavic_4k`, the one group with a fitted camera model (D12).
3. **Off-nadir handling** (D14; replaces the original ±10° nadir filter, which kept only 296 frames from 3 videos).
   - Every frame with altitude, pitch and a camera model gets per-pixel GSD from the tilted pinhole model over a flat sea (`geometry/camera.py`).
   - Stages that need a uniform GSD rectify the ground plane (a homography) into nadir-equivalent views. They use only pixels within an off-nadir limit (θ_train for detector views in M2, a stricter θ_lib for libraries in M4) and only where the native GSD is ≤ the target.
   - Log frame counts per off-nadir band.
4. **GSD model per camera group.**
   - Native GSD = altitude × k, with k = 2·tan(HFOV/2) / image_width = 1/f, at nadir.
   - The HFOV is not in the files, and box sizes cannot separate object size from focal length. f is therefore fitted from the open-sea horizon (`fit-camera`, D12).
   - Record k, the fit plot and the residuals.
5. **Sessions** (D15). One session per capture day, across all camera groups: the drones of a campaign filmed the same exercise in the same hours. Frames without a trustworthy day form an undated session per camera group, tied to their campaign.
6. **Splits** (D16). Pool all labelled images; the official train/val split is not session-disjoint. Split by session into:
   - **A** — detector training and internal validation (the largest share).
   - **B** — sensor-model fitting, plus imagery for RL training and validation worlds.
   - **T** — test only: test worlds, final calibration evaluation, fidelity check.
   Stratify by altitude band and swimmer presence, using an exhaustive search over session assignments. B and T come only from the open-sea campaign. Report counts per split × altitude band × class.
7. **Leakage check** (D17). A perceptual-hash near-duplicate search across splits, confirmed by thumbnail correlation. Thresholds come from known duplicates (consecutive video frames). It must pass before any training.

**Done when:** `docs/data_notes.md` holds the verified field names, camera groups, GSD model, split statistics and the leakage-check result.

---

## 8. Camera, altitude levels and GSD

The simulated camera models the detector's effective input resolution (an onboard compute budget), not the original 4K sensor. This is a deliberate, defensible design knob. State it in the write-up.

**Camera spec**
- Square resolution R×R px (starting point R = 640).
- Horizontal FOV θ (starting point 80°).
- Nadir, fixed yaw.
- Footprint width W(h) = 2·h·tan(θ/2).
- Simulated GSD g(h) = W(h)/R.

**Altitude levels.** Define a small discrete set of levels by footprint width in whole cells, so views tile the grid exactly. Derive each altitude as h_ℓ = W_ℓ / (2·tan(θ/2)).

**Design procedure** (after M2, before building any cache)
1. Measure swimmer recall vs GSD on real B frames resampled across a sweep (e.g. 2–30 cm/px).
2. Choose 3–5 levels whose recall spans roughly ≥0.8 at the finest level to ≤0.3 at the coarsest, with useful intermediate levels in between.
   - If the coarse levels have near-zero recall, the tradeoff collapses: the agent will simply stay low. Raise R or shrink the coarse footprints.
   - If even the coarsest level has high recall, there is no tradeoff either. Lower R or enlarge the footprints.
3. Choose cell size c so every footprint is an integer number of cells.
4. Choose the battery (§14) so full coverage at the finest level is infeasible but full coverage at the coarsest level is cheap.
5. Record the final table in `configs/levels/` and in the Decision Log.

**Worked example** (illustrative only — the real numbers come from step 1). Area 300 m × 300 m, c = 5 m (60×60 grid), θ = 80°, R = 640:

| Level | Footprint | Altitude | Sim GSD | Stride | Views per world |
|---|---|---|---|---|---|
| L1 | 25 m (5 cells) | ≈14.9 m | ≈3.9 cm/px | 5 cells | 144 |
| L2 | 50 m (10 cells) | ≈29.8 m | ≈7.8 cm/px | 5 cells | 121 |
| L3 | 100 m (20 cells) | ≈59.6 m | ≈15.6 cm/px | 10 cells | 25 |
| L4 | 150 m (30 cells) | ≈89.4 m | ≈23.4 cm/px | 15 cells | 9 |

A swimmer (about 0.3–0.8 m seen from above) spans roughly 8–20 px at L1 but only 1–3 px at L4. L4 is probably too harsh — exactly what the design procedure should catch.

**Canonical world GSD g₀** is the resolution at which worlds are composed.
- It must be finer than the finest simulated GSD, with margin: g₀ ≤ about 0.65·g(L1). Example: g₀ = 2.5 cm/px when g(L1) = 3.9 cm/px.
- Imagery used to build worlds must have native GSD ≤ g₀.

**Resampling.** There is one shared function in `src/altiseek/imaging/`:
- anti-aliased area-averaging downsampling;
- then the camera pipeline (optional mild blur, JPEG re-encoding).

Use the identical function for detector training views, real-frame evaluation and world rendering. Different resampling paths create a hidden domain gap.

---

## 9. Detector

**Goal:** a small, fast detector that finds swimmers (and the distractor classes) in simulated-resolution views.

**Model**
- A small one-stage detector, such as a nano/small YOLO-family model, or another framework. Record the library, version and licence in the README; some popular YOLO implementations are AGPL-licensed.
- Input size = R.
- Train on all classes, which helps separate swimmers from boats and buoys. The search itself uses swimmer detections only.
- If recall is unusable at intermediate levels, try a higher-resolution detection head before changing the camera spec.

**Training data** (split A only)
- **Real frames only**, resampled to target GSDs drawn across the simulator's GSD range, with jitter.
  - Oblique frames are ground-plane rectified into nadir-equivalent views first (D14), using pixels within θ_train of nadir.
  - Use only source pixels whose native GSD (across and along the view) ≤ the target GSD.
  - Prefer source frames whose resampled size covers most of the R×R view.
  - Crop R×R windows; letterbox when the resampled frame is smaller than R in one dimension.
- **Never train on pasted composites.** The detector would learn paste artefacts and inflate recall inside the simulator. Disable copy-paste augmentation.
- **Augmentations:** flips, 90° rotations (nadir views have no "up"), colour jitter, mild blur, JPEG. Mosaic-style augmentation changes scale and context; log whether it's used.
- **Internal validation:** a session-disjoint subset of A, for early stopping and model selection only.

**Evaluation** (real B and T frames, resampled per GSD bin and per level)
- Swimmer AP50.
- Recall and precision at the operating threshold τ.
- False positives per view.
- All of the above broken down per GSD bin / level and per object-size bin.
- Also report centre-distance matching (predicted centre within e.g. 1 m of the ground-truth centre), because IoU is unstable for objects a few pixels wide.
- **Required plots:** recall vs GSD (the project's premise), precision vs GSD, false positives per view vs GSD.

**Inference API (contract)**
- Input: a batch of R×R uint8 RGB views.
- Output per view: detections as (class, raw score, box in view pixels), tagged with `detector_id`.
- Deterministic apart from GPU nondeterminism.

**Training runs on cloud GPUs.** Checkpoint often, make training resumable, and log the config, split IDs and dataset version.

---

## 10. Calibration and sensor models

These are two different things. Keep them separate in code and in the write-up.
- **Calibration:** is a detection's score a probability? P(true swimmer | detection with score s, level ℓ).
- **Sensor model:** how likely is a detection given that a person is, or isn't, in a cell? This needs recall (misses) and false-alarm rates. Calibration alone says nothing about misses.

**Cell-level matching** (use the same definitions when fitting and inside the simulator)
- A detection *lands* in the cell containing its box centre, in world coordinates.
- A cell is *observed* at level ℓ if its centre lies inside the footprint.
- r_ℓ(τ) = P(an observed cell containing a swimmer gets ≥1 swimmer detection with score ≥ τ in that cell).
- f_ℓ(τ) = P(an observed cell with no swimmer gets ≥1 swimmer detection with score ≥ τ). This includes distractor cells at the world's distractor density.
- **Swimmers near cell boundaries:** decide the rule (centre cell only, or centre cell plus neighbours) and log it. Use the same rule everywhere.

**Fit data**
- Cached detections on the *sensor-fit* world set: split-B imagery, with seeds disjoint from the RL worlds.
- This is the same rendered domain the agent experiences.
- Validate on the test worlds and on real T frames (the fidelity check, §11).

**Calibration method**
- Per-level temperature scaling on score logits, fit by minimising the BCE of TP/FP labels on the sensor-fit set.
- Report ECE and reliability diagrams per level, before and after calibration, on the test worlds and on real T frames.
- Optional: an altitude-conditioned calibrator, e.g. logistic regression on score logit, GSD and box size.

**Re-look correlation**
- The detector is deterministic, so re-observing a cell from the same level gives highly correlated evidence.
- Measure the conditional recall P(detected on look 2 | missed on look 1) using different cached variants.
- This number decides the re-look rule (§13).

**Sensor-model variants**

All four share one interface: given a prior p, a level ℓ and the cell's detection outcome, return a posterior p′. The detection outcome is the maximum swimmer score landing in the cell, or "none". Updates happen in log-odds (posterior odds = prior odds × LR), clipped to a safe range.

| Variant | Evidence when the cell gets a detection | Evidence when it doesn't |
|---|---|---|
| **V0** Naive | p′ = max(p, s_raw) | Fixed recall r₀ (e.g. 0.9) at every level, f = 0: p′ = p(1−r₀) / (1 − r₀·p) |
| **V1** Global binary | LR = r/f at threshold τ, pooled over levels | LR = (1−r)/(1−f), pooled |
| **V2** Altitude binary | LR = r_ℓ/f_ℓ | LR = (1−r_ℓ)/(1−f_ℓ) |
| **V3** Altitude soft | LR = [r_ℓ·p(s \| swimmer, ℓ)] / [f_ℓ·p(s \| empty, ℓ)], using calibrated-score histograms at τ_min | LR = (1−r_ℓ)/(1−f_ℓ) at τ_min |

**Deliverable:** sensor-model files, versioned and tied to `detector_id`, containing:
- r_ℓ and f_ℓ with bootstrap CIs;
- τ and τ_min;
- per-level temperatures;
- the score histograms;
- the conditional recall;
- ECE before and after calibration;
- the fit world-set ID.

---

## 11. World generator and renderer

A world is a lightweight **spec**, not a giant image. The renderer composes whatever view is requested on demand.

**Libraries** (built separately per split; never mix splits)
- **Backgrounds.**
  - Sea patches from rectified regions within θ_lib of nadir (D14) with native GSD ≤ g₀.
  - Patches must contain no annotated objects or ignore regions, with a safety margin.
  - Store appearance statistics (Lab mean/std, texture energy).
  - Cluster patches into *appearance conditions*, e.g. calm/dark, glare, choppy.
- **Object crops.**
  - Swimmers (targets), plus boats, buoys, jetskis and life-saving appliances (distractors), from rectified regions within θ_lib of nadir with native GSD ≤ g₀.
  - Each crop stores a small water margin, a feathered soft alpha mask, its physical size (from GSD) and its source session.

**World spec**

| Field | Meaning |
|---|---|
| world_id, seed, split, world_set | Identity (world_set ∈ sensor-fit / rl-train / rl-val / test) |
| area_m, cell_m, g₀ | Geometry |
| appearance_condition | Background cluster used for this world |
| background_layout | Patch placements (patch_id, position, rotation in 90° steps, flip) covering the area, with overlaps for blending |
| objects | (class, crop_id, position_m, rotation, small physically plausible scale jitter) |
| targets | Which objects are swimmers to find (ground truth) |
| prior_given | Prior parameters the agent receives |
| prior_true | Parameters used to sample targets (may differ, for misspecification tests) |
| K, distractor_density, family tags | Scenario descriptors |

**Scenario sampler**
- **Prior.**
  - An anisotropic Gaussian, optionally a 2-component mixture.
  - Its mean lies inside the area; it has major/minor standard deviations and an orientation (the drift direction).
  - Discretise it to cells.
  - Default: targets always lie inside the area, so truncate and renormalise the prior.
- **Targets:** K ∈ [1, K_max].
  - Clustered mode: sample a group centre from the prior, then place individuals within a small radius of it.
  - Independent mode: sample each target from the prior.
- **Distractors:** independent of the prior, either uniform or along a random "lane".
- **Constraints:** minimum spacing between objects; all objects fully inside the area.
- **Family knobs:** prior sharpness, K and clustering, distractor density, appearance condition, battery.

**Renderer**
- Composite the background patches, resampled to g₀, with seam blending (feathered or multi-band) and colour harmonisation to the world's appearance condition.
- Paste objects with soft alpha, colour-matching each crop's water margin to the local background.
- **To render a view:**
  1. Crop the footprint region at g₀.
  2. Resample to R×R with the shared function (§8).
  3. Apply the camera pipeline.
- **Variants:** K_var variants per view.
  - Each variant is a sub-pixel translation (under one simulated pixel) plus photometric jitter (exposure, white balance, glint strength).
  - Variant draws are seeded.

**Fidelity check** (must pass before any RL)
- For each level, compare detector recall and false-alarm rate on rendered test-world views against real T frames resampled to the same GSD.
- Report the gap per level.
- If pasted swimmers are detected markedly more or less often than real ones, fix blending and colour matching before continuing.

---

## 12. Detection cache

**Purpose:** RL needs millions of environment steps, so the detector must never run inside the training loop.

- **Lattice:** per level, view centres on a grid with that level's stride (in cells). Every footprint lies fully inside the area.
- **Generation:**
  - For every (world, level, view, variant): render → detect (batched on GPU) → store every detection with score ≥ τ_min.
  - Each record holds class, raw score, box (view px), centre (world m) and cell index.
  - Calibrated scores are computed at load time from the sensor model.
- **Storage:**
  - One compressed columnar/array file per world.
  - The file header holds world_id, detector_id, camera spec, level spec, variant spec, τ_min and cache_id.
- **Loading:** environment workers keep an LRU pool of worlds in memory and memory-map large files.
- **Validation:** re-run live inference on random views. Outputs must match the cache within a small tolerance (GPU nondeterminism).
- **Staleness:** a new `detector_id` invalidates every cache and sensor model built from the old one. Loaders check this.
- **Generation runs on cloud GPU** in resumable per-world chunks, with a manifest recording which worlds are complete.

---

## 13. Belief map

**State**
- **p[ix, iy]:** probability that the cell contains ≥1 undiscovered target, stored as log-odds.
- **observed[ℓ, ix, iy]:** informative-observation counts per level.
- **used_variants[ℓ, ix, iy]:** cached variants already consumed, for the re-look rule.
- **resolved[ix, iy]:** cells closed by a declaration (verified found, or verified empty).

**Initialisation**
- Let q be the per-target cell distribution (the discretised prior_given) and K the number of targets.
- Then p_i = 1 − (1 − q_i)^K.
- This assumes independence across cells. It is an accepted approximation; name it in the limitations.

**Update after an observation at level ℓ.** For every unresolved cell whose centre lies inside the footprint, compute the cell's detection outcome and apply the active sensor-model variant.

**Re-look rule** (default; configurable)
- **Default:** only the first observation per (cell, level) updates the belief. Later views of that cell from the same level are uninformative.
- **Ablation:** each distinct variant counts, optionally down-weighted using the measured conditional recall.
- **Why it matters:** with a deterministic detector and no rule like this, the agent can hover in place and drive the belief to false certainty — and PPO will find and exploit that.

**Found targets**
- Once a target is found, suppress cached detections whose centres lie within radius ρ of it in all later observations; the rescue boat has picked the person up.
- Otherwise stale detections leak into neighbouring cells.

**Declaration**
- **Auto-declare** (default; used for all main comparisons).
  - Any cell with p > τ_declare is declared.
  - Ground truth resolves it, as if a rescue boat checks: true → target found; false → false declaration.
  - Either way, the cell becomes resolved.
- **Learned declare** (ablation): a DECLARE action declares the highest-p cell in the current footprint.

**Derived quantities** (for observations, baselines and logs): entropy map, Σp over unresolved cells, coverage per level.

---

## 14. Search environment (Gymnasium)

**Episode**
- **Reset:** sample or load a scenario. The drone starts at a configurable launch point (default: a random edge cell) at a default level, and takes an initial observation.
- **Step:**
  1. Apply the action: one lattice step, or a level change that snaps to the nearest lattice point at the new level.
  2. Charge energy and time.
  3. Observe from the cache (seeded variant draw).
  4. Update the belief and run auto-declare.
  5. Compute the reward and build the observation.
- **Termination:** all targets found.
- **Truncation:** battery exhausted, or the step cap reached.

**Actions** (discrete, masked)
- Move N, E, S or W one lattice step at the current level; Ascend; Descend; plus DECLARE in learned-declare mode.
- Invalid actions (leaving the area, going past the lowest or highest level) are masked, not penalised.
- Masks must be supplied during training *and* evaluation.

**Energy/time model** (one implementation, shared with the Gazebo supervisor)
- Horizontal time = distance / v_h.
- Vertical time = |Δh| / v_v.
- Plus a dwell time t_obs per observation.
- Energy = time in hover-equivalent units, with a multiplier on climbing.
- Battery = energy budget.
- **Starting points:** v_h = 10 m/s, v_v = 3 m/s, t_obs = 2 s, climb multiplier 1.5, battery 600 s hover-equivalent. Then adjust using the §8 rule. In the worked example, full L1 coverage costs about 650 s (infeasible) and full L4 coverage under about 120 s (cheap).

**Observation** (dict)
- **Map tensor** (C × G × G, float32):
  - belief p (or clipped log-odds);
  - prior q;
  - one observed-mask per level;
  - resolved mask;
  - current footprint mask;
  - drone position;
  - optional x/y coordinate channels.
- **Vector:** current level (one-hot), battery fraction, elapsed-time fraction, targets found / K, and K itself (known by default; unknown as an ablation).
- Ground truth never enters observations.

**Reward** (configurable; defaults)
- +1 per target found.
- −0.5 per false declaration.
- −(energy used this step / battery) per step, so spending the whole battery costs −1.
- **Optional shaping must be potential-based:** F = γ·Φ(s′) − Φ(s), with Φ computed from the belief (e.g. −entropy, or −Σp over unresolved cells). Report results with and without shaping. Any other shaping can be farmed.

**Engineering requirements**
- Passes Gymnasium's environment checker.
- Fully deterministic given a seed.
- Vectorised NumPy belief updates. Profile so the environment is never the training bottleneck; aim for sub-millisecond steps.
- Episode logging lives in a wrapper (§18), not in core environment logic.

---

## 15. Policies

Every policy implements the same interface: observation + action mask → action.
- Classical baselines may also read the belief state and the active sensor model directly. This is the same information the agent has.
- Only the oracle sees ground truth.

**Bounds**
- **Random:** uniformly random valid action.
- **Oracle:** knows the target cells and flies a short tour (nearest-neighbour or a TSP solver) visiting each one at the finest level. Used to normalise scores.

**Classical baselines** (tune every hyperparameter on rl-val worlds only)
- **B1 Lawnmower:** a boustrophedon sweep at one fixed level, chosen per family on rl-val.
- **B2 Prior-ordered sweep:** visits views at a fixed level in order of prior mass, using nearest-neighbour ordering over high-mass views. This approximates the expanding-square and sector patterns used in SAR practice.
- **B3 Two-phase "scan high, confirm low":**
  - A prior-ordered sweep at a coarse level.
  - Whenever a cell's p exceeds τ_candidate, descend to the confirm level over it, then resume.
  - Tune the coarse level, the confirm level and τ_candidate.
- **B4 Greedy information gain:**
  - Pick the action that maximises expected entropy reduction over the next footprint per unit energy, computed in closed form from the sensor model.
  - Break ties toward higher expected finds.
  - Pitfall: the expected posterior equals the prior (beliefs are a martingale), so "expected reduction in Σp" is not a valid greedy objective. Use entropy or expected-find criteria.
- **B5 Lookahead** (optional; the strongest classical baseline): depth-2/3 rollout or tree search on the same objective.

**PPO**
- **Algorithm:** masked PPO — either sb3-contrib's MaskablePPO with a multi-input policy, or a single-file PPO if you want full control. Record which.
- **Network:**
  - A CNN over the map tensor (a few strided convolutions or small residual blocks).
  - Its output is concatenated with the vector features → MLP → policy and value heads.
  - Ablate coordinate channels vs an egocentric map re-centred on the drone.
- **SB3 caveat:** by default SB3 only treats uint8 Box spaces as images, so a float32 map gets flattened into an MLP by the default extractors. Check how your installed version behaves, and write a custom features extractor that runs the CNN on the map and concatenates the vector features.
- **Training distribution:** a mixture over all scenario families (one generalist policy), evaluated per family. A curriculum (small areas → full size) is optional.
- **Model selection:** mean normalised AUC on rl-val. Test once per frozen checkpoint.
- **Seeds:** at least 3 per configuration. Report all of them, never the best one alone.
- **RQ1:** a separate policy per sensor-model variant. If compute is tight, train RL on V0 and V3 only, and run every baseline on all four variants.
- **Behaviour analysis:**
  - altitude-usage histograms and time spent per level;
  - how often the agent descends after a coarse-level detection;
  - first-find times;
  - a side-by-side comparison with B3.

---

## 16. Evaluation protocol and experiments

**World sets** (disjoint seeds; imagery by split)

| Set | Imagery | Used for |
|---|---|---|
| sensor-fit | B | Calibration and sensor models |
| rl-train | B | PPO training |
| rl-val | B | Baseline tuning, checkpoint selection |
| test | T | Final numbers only |

Test scenario seeds are fixed and shared by every method, so all comparisons are paired.

**Scenario families** (main grid)
- Prior: sharp vs diffuse.
- Targets: single vs clustered group.
- Distractors: low vs high density.
- Battery: default, plus a sweep from tight to generous.

**Metrics** (per episode, aggregated per family)

| Metric | Definition |
|---|---|
| Found fraction | Targets found by the end of the episode / K |
| Success | All targets found |
| Found-vs-time AUC | Area under found fraction vs time, normalised by battery; rewards finding early |
| Time to first find | Censored at the battery budget |
| False declarations | Count per episode |
| Usage | Energy used, path length, time per level |
| Normalised score | (AUC_method − AUC_random) / (AUC_oracle − AUC_random) |

**Statistics**
- At least 200 test scenarios per family (justify any fewer).
- Bootstrap 95% CIs and paired comparisons on shared seeds.
- RL: mean and spread across training seeds, plus per-seed results.

**Experiments**
- **E1 (RQ1):** every policy × V0–V3. Report metric deltas, plus V0 diagnostics:
  - false clears (true target cells driven below e.g. p = 0.01 by coarse-level negative evidence);
  - false declarations.
- **E2 (RQ2):** PPO vs B1–B5 per family, plus the battery sweep.
- **E3 (robustness):**
  - prior misspecification (prior_true ≠ prior_given: shifted, rotated, scaled);
  - higher distractor density than in training;
  - detector swap (a different checkpoint, with its sensor model re-fitted);
  - a camera-resolution change;
  - an appearance condition held out of training.
- **E4 (RQ3):** see §17.
- **Ablations:** re-look rule; learned vs auto declare; known vs unknown K; shaping on/off; coordinate channels vs egocentric map.

**Hygiene**
- Freeze the protocol in `configs/eval/protocol_vN.yaml` before any test-set run.
- Any change means a new version, a Decision Log entry, and rerunning every method.
- If results look too good, suspect leakage first: splits, crops in the wrong split, a cache from the wrong detector, ground truth reaching observations.

---

## 17. PX4 / Gazebo / ROS 2 deployment (RQ3)

**Goal:** run the trained policy closed-loop with PX4 SITL flying a simulated quadrotor in Gazebo, and the detector running live on the simulated downward camera. Use scenarios matched to the test worlds.

**Environment**
- **Versions:** WSL2 Ubuntu with the Ubuntu / ROS 2 / Gazebo combination that PX4's *current* docs recommend. Record the exact PX4, ROS 2, Gazebo and px4_msgs versions in `docs/env.md`.
- **PX4 ↔ ROS 2:** through the uXRCE-DDS bridge. The px4_msgs definitions must match the PX4 version.
- **Gazebo camera ↔ ROS 2:** through the ros_gz bridge.
- **Python environment:**
  - rclpy comes from the system Python.
  - Set things up so ROS nodes can import this package, the detector library and the trained policy — for example, a virtual environment that can see system site-packages.
  - Record the approach in `docs/env.md`.
- **No duplicated logic.** ROS nodes import the same belief, sensor-model, observation-builder and energy-model code as the Python simulator.

**Gazebo world**
- **Export:** turn a world spec into a flat textured plane. Render the composed sea at g₀, split it into tiles that respect GPU texture-size limits, and place them at exact world coordinates.
- Targets and distractors are baked into the texture. This is acceptable for a nadir camera.
- Flat or ambient lighting, no shadows.
- Texture filtering (mipmaps) changes appearance with distance. That is part of the measured sim-to-sim gap, not a bug to hide.
- **Camera:** downward-facing, resolution R, HFOV θ, fixed yaw. Verify the GSD at every level altitude by rendering a checkerboard with a known square size.

**ROS 2 nodes** (one responsibility each)

| Node | Subscribes to | Publishes | Responsibility |
|---|---|---|---|
| camera bridge | Gazebo camera | Image | Bridging only |
| detector | Image + pose (time-synchronised) | Detections in world frame | Live inference; tags messages with detector_id |
| belief | Detections, pose, level | Belief map, footprint | Shared belief + sensor-model code; footprint from the *actual* pose, not the commanded one |
| policy | Belief, pose, battery | Next goal (lattice point + level) | Shared observation builder, trained policy, masks |
| offboard controller | Goals, PX4 status | PX4 offboard setpoints | Arming, takeoff, offboard heartbeat, goal tracking, a "settled" signal |
| supervisor | All of the above | Episode events, logs | Scenario loading, shared energy model, ground-truth resolution of declarations, metrics, rosbag recording |

**Execution: stop-and-shoot**
1. Fly to the goal.
2. Wait until position error and velocity stay below thresholds for a settle time.
3. Capture, detect, update the belief, decide the next move.

This avoids motion blur and synchronisation problems.

**Gotchas**
- PX4 needs a continuous offboard setpoint stream (faster than 2 Hz) *before* switching to offboard mode, and the whole time it is in offboard.
- Gazebo altitude is never exact. Map the actual altitude to the nearest level for the sensor model, compute the actual footprint, and don't update partially covered cells.
- Check whether your PX4/Gazebo setup supports a simulation speed factor (lockstep) to run faster than real time.

**Experiments**
- Run matched scenarios (a subset of test scenarios, e.g. 20–50) with the best PPO policy and the best baseline, both on V3.
- Compare against cached-simulator results on the same scenarios: metric deltas, and per-level recall and false-alarm rates in Gazebo vs the cache.
- **H3b:** export sensor-fit worlds to Gazebo, fly calibration sweeps, re-fit the sensor model on Gazebo detections, and rerun the matched scenarios.
- **Deliverable:** a demo video showing a top-down view with the footprint and belief overlay, plus the drone-camera feed as an inset.

---

## 18. Experiment tracking, artifacts and reproducibility

- **Configs:** hierarchical YAML with schema validation. Every run saves the resolved config, git commit, seeds, package versions and artifact IDs.
- **Runs:** `runs/<stage>/<timestamp>_<name>/` containing config, logs, metrics and checkpoints.
- **Tracking:** pick one of TensorBoard or W&B. Also write results to CSV/Parquet, so figures can be regenerated without the tracker.
- **Artifact registry:** a manifest listing every detector, sensor model, world set and cache, with its ID, parent IDs and storage location (local or cloud).
- **Episode logs:**
  - per step: time, battery, position, level, action, mask, detection summary, declarations, found events;
  - per episode: scenario seed, method, sensor variant, metrics;
  - belief snapshots at a configurable interval, for replays.
- **Figures** are produced by scripts from saved results only, never from hand-edited notebooks.

---

## 19. Testing and sanity checks

**Unit tests**
- **Geometry:** footprint vs altitude; GSD; world ↔ grid ↔ image conversions (including the row flip); NED ↔ ENU, including yaw.
- **Imaging:** every caller of the shared resampling function gets identical output for identical input.
- **Belief:**
  - every variant matches hand-computed Bayes updates;
  - log-odds clipping works and no NaNs appear;
  - a detection raises p and a non-detection lowers p whenever r > f;
  - resolved cells are never touched;
  - the re-look rule is enforced.
- **Sensor model:** probabilities stay in [0,1]; histograms are normalised; loading refuses a mismatched detector_id.
- **World generator:**
  - the same seed gives the same spec;
  - objects stay inside the area and respect spacing;
  - split purity: every crop and background comes from the world's declared split.
- **Cache:** lookups match live inference within tolerance; header IDs are validated.
- **Environment:**
  - passes Gymnasium's checker;
  - deterministic under a seed;
  - masks are correct at boundaries and at the min/max levels;
  - energy accounting is correct;
  - termination and truncation fire correctly;
  - found-target suppression works.

**Integration test:** a tiny world run end-to-end with a scripted policy. Replaying by seed gives identical logs.

**Sanity checks** (run after any major change; a failure blocks progress)
- The oracle is far better than random on every family.
- With unlimited battery, a lawnmower at the coarsest level finds fewer targets than at the finest. This is the premise.
- With V0, a coarse-level-only lawnmower produces measurable false clears.
- A PPO agent whose belief channels are zeroed does no better than a prior-only heuristic. If it does better, it is exploiting something it shouldn't see.

---

## 20. Visualisation and deliverables

**Episode replay:** belief heatmap, prior contours, drone path coloured by level, footprints, detections, declarations and true targets; exportable as GIF/MP4.

**Required figures**
- Recall, precision and false positives per view vs GSD.
- Reliability diagrams per level, before and after calibration.
- r_ℓ and f_ℓ with CIs.
- Found-fraction-vs-time curves per family.
- The normalised-score table.
- Altitude usage, PPO vs B3.
- The robustness table.
- The Gazebo vs cache comparison.

**Repository deliverables**
- A README covering the problem, a method diagram, key results, how to reproduce each stage, limitations, licences and citations.
- `docs/data_notes.md`, `docs/env.md` and the results tables.
- The demo video.
- A paper-style write-up: introduction, related work, method, experiments (RQ1–RQ3), limitations, conclusion.

**The limitations section must state:**
- simulation only;
- nadir camera;
- flat sea;
- static targets;
- composited worlds;
- detector training and world imagery include ground-plane-rectified oblique frames (D14);
- only six capture days, so splits are coarse and the same volunteers may appear on several days;
- the independence approximation in the belief map.

---

## 21. Build order and acceptance criteria

Each milestone depends on the ones before it. Don't start the next one until the current one's "done when" holds.

| Milestone | Scope | Done when |
|---|---|---|
| **M0** Repo & tooling | Package layout, config system, test runner, formatter/linter, .gitignore for data/ and runs/. Add real code as stages need it, not stubs. | Tests run; configs validate. |
| **M1** Data | §7 | data_notes.md verified; A/B/T built by session; leakage check passes; GSD model fitted, with its plot. |
| **M2** Detector | §9 | Recall, precision and FP-per-view vs GSD curves on real B and T frames; the inference API returns detections plus detector_id; checkpoint registered. |
| **M3** Level design | §8 | Level table, camera spec, cell size and battery chosen and logged; footprints are whole cells; coverage-cost check passes. |
| **M4** Worlds & renderer | §11 | Libraries per split; scenario sampler; renderer; fidelity check reported per level and accepted. |
| **M5** Detection cache | §12 | Caches for the sensor-fit, rl-train, rl-val and test sets; validation passes; manifest complete. |
| **M6** Sensor models | §10 | V0–V3 fitted; ECE before/after reported; conditional recall measured; re-look rule confirmed in the Decision Log. |
| **M7** Env + baselines | §13–§15, no RL yet | Env passes the checker and the tests; random, oracle and B1–B5 run on rl-val; sanity checks pass; baseline hyperparameters frozen. |
| **M8** PPO | §15 | Stable learning curves over ≥3 seeds; checkpoint chosen on rl-val; behaviour analysis done. |
| **M9** Main experiments | §16: E1–E3 and the ablations on test worlds, under a frozen protocol | Every table and figure regenerates from saved results with one command each. |
| **M10** Deployment | §17 | Matched scenarios run end-to-end in PX4/Gazebo; RQ3 comparison and the H3b re-fit done; demo video recorded. |
| **M11** Write-up & polish | §20 | All deliverables complete. |

---

## 22. Pitfalls checklist

**Data and leakage**
- The official train/val split may share sessions. Re-split by session.
- Near-duplicate frames across splits inflate both detector and simulator results.
- Crops or backgrounds from the detector's training split inside evaluation worlds inflate recall.
- Training the detector on pasted composites teaches it paste artefacts.
- Missing or zero altitude, oblique frames and mixed cameras all produce wrong GSDs.
- Rectified oblique views are not true nadir views. 3D objects lean away from the camera, and the water reflects more sky at grazing angles. The fidelity check measures the effect; keep θ_lib strict.
- Logged gimbal pitch lags the camera while the gimbal moves, because metadata is matched to frames by nearest timestamp.
- ODv2 blackens ignore regions in the pixels. Their edges, with JPEG ringing, look like strong image edges.
- Using different resampling for training and rendering creates a hidden domain gap.

**Belief, reward and training**
- A deterministic detector without a re-look rule lets the agent farm certainty by re-observing the same cells.
- Found targets left in the cache produce phantom detections in neighbouring cells.
- Shaping that isn't potential-based gets farmed.
- "Expected Σp reduction" is zero in expectation (beliefs are a martingale), so it is not a usable greedy objective.
- Retraining the detector leaves stale caches and sensor models behind.
- Image row vs north axis, NED vs ENU, and yaw conventions are easy to get wrong.
- SB3 flattens float "images" into an MLP by default.
- Forgetting action masks at evaluation time silently changes results.

**Evaluation hygiene**
- Never tune anything on test worlds.
- Never report only the best seed.
- Never change the protocol after seeing test numbers.

**Compute and repository**
- Cloud notebook sessions end mid-job, so every long job must be resumable.
- Never commit data or derived imagery to git.

**Deployment**
- PX4 needs the offboard heartbeat, and px4_msgs must match the PX4 version.
- Gazebo has texture-size limits, and mipmapping blurs the texture at altitude.

---

## 23. Working agreements for Claude

**Before acting**
- Read this file. If a request conflicts with it, say so and propose a spec update instead of silently diverging.
- Inspect, don't assume. Never invent dataset field names, file layouts, library APIs or version-specific behaviour. Check the data, the docs or the installed package first, and record findings in `docs/`.

**How to work**
- Be direct and concise. Flag problems, risks and disagreements immediately; no hedging, no flattery.
- Work in small, reviewable increments: one module or feature at a time, with its tests. Afterwards, summarise what changed, what is verified and what comes next.
- No placeholder scaffolding. Don't generate empty stubs or TODO skeletons across the repo; build real functionality when a stage needs it.
- Ask before adding heavy dependencies or changing the module layout.
- Keep all randomness seeded and all artifact IDs intact. Refuse to mix artifacts with mismatched IDs.
- Put units in names and docstrings, and name frames explicitly.

**Protecting the results**
- Never use split T, test worlds or test scenario seeds for training, tuning or model selection.
- Never edit a frozen evaluation protocol. Create a new version and log it.
- When results look too good, investigate leakage before celebrating.

---

## 24. Decision log

| ID | Decision | Rationale | Status |
|---|---|---|---|
| D1 | Simulation only; no hardware | Owner constraint | Fixed |
| D2 | Nadir camera, fixed yaw | Simple, exact footprint/GSD geometry; matches nadir-filtered data | Fixed |
| D3 | Policies output high-level lattice moves; PX4 flies | The project is about where to look, not how to fly | Fixed |
| D4 | Session-disjoint splits A/B/T | Leakage control | Fixed |
| D5 | Auto-declare at τ_declare for main comparisons | Isolates search behaviour; learned declare is an ablation | Default |
| D6 | Re-look rule: first observation per (cell, level) | Deterministic detector; revisit once conditional recall is measured | Provisional |
| D7 | Camera resolution and level table | Decided at M3 from the measured recall-vs-GSD curve | Pending |
| D8 | Project name: AltiSeek (working title was SeaSearch) | Names the new idea (the drone chooses its own altitude); "Seasearch" is already a UK marine-survey programme | Fixed |
| D9 | Python 3.12 everywhere (`requires-python = ">=3.12,<3.13"`) | PX4 recommends ROS 2 Jazzy on Ubuntu 24.04, whose system Python (hence rclpy) is 3.12; Colab runtime 2026.07 is 3.12.13 and Kaggle builds on the Colab image; local dev has 3.12.10. Evidence in `docs/env.md` | Fixed |
| D10 | Tooling: uv ≥ 0.13.4 with a committed `uv.lock`; hatchling; ruff for lint and format, run inside pytest; GitHub Actions CI on Ubuntu and Windows | `pytest` alone enforces style; pyproject stays the single place dependencies are declared, the lockfile pins transitive ones; uv < 0.13.4 trusts a compromised CA (GHSA-hq85-5cpj-mr62) | Fixed |
| D11 | Config system: PyYAML + pydantic v2 in `altiseek.config`. A `_base_` key gives inheritance and cross-group composition; dotted CLI overrides; schemas forbid unknown keys and are frozen | Lighter and more explicit than Hydra/OmegaConf; misspelt keys fail loudly; the resolved config serialises for run metadata (§18) | Default |
| D12 | `mavic_4k` camera model: f = 3029 px (95% CI 2932–3225), i.e. HFOV 64.7° and nadir GSD = 3.30 × 10⁻⁴ m/px per metre of altitude; gimbal pitch bias −0.38°. Fitted from the open-sea horizon (`fit-camera`), not from box sizes | §7 step 4's box-size fit cannot separate object size from focal length; the horizon gives f directly. The fit confirms the Mavic 2 Pro's full-width 4K mode. Details in `docs/data_notes.md` | Fixed until re-measured |
| D13 | Runtime deps numpy 2.0.2 (Colab's version), Pillow 12.3.0, matplotlib 3.11.2 | Needed for M1 imaging and figures; matching Colab avoids reinstalls in cloud sessions | Fixed |
| D14 | Oblique source frames are used through a tilted pinhole model over a flat sea and ground-plane rectification into nadir-equivalent views. Off-nadir limits are θ_train (detector views, set in M2) and a stricter θ_lib (libraries, set in M4). This replaces the ±10° nadir filter of §7 step 3 | Nadir-only keeps 296 `mavic_4k` frames from 3 videos, too few for session-disjoint splits; the median 4K pitch is about 31°. Rectification is exact for a flat sea with a known camera model (D12). The cost is an appearance gap (lean of 3D objects, sky reflection), which the fidelity check measures | Fixed |
| D15 | Session = capture day across all camera groups. Frames without a trusted day get `undated:<group>` sessions: `trinity_rgb` (no timestamps) and MicaSense frames without GPS (their clock dates them to 2021-08-27, when nothing else flew). Both are assigned to campaign `lake_constance_2020` | The Mavic and M210 filmed the same exercises in the same hours, so a finer unit would leak swimmers, boats and sea state across splits. The paper flew the Trinity only at Lake Constance in 2020 | Fixed |
| D16 | Split by exhaustive search over session blocks: target shares A/B/T = 0.6/0.25/0.15 of usable (`mavic_4k`) frames, stratified by altitude band (0/20/40/60/200 m) × swimmer presence. Undated sessions are tied to their campaign. B and T only from `north_sea_2021`. B and T need ≥ 100 near-nadir (≤ 30°) swimmer boxes. Result: A = Lake Constance 2020 + 2021-08-20, B = 2021-08-21, T = 2021-08-19 (`split_id` 783b4403804291a2) | Sensor models are fitted on B worlds and judged on T worlds, so both must come from the target domain (open sea); the lake adds variety to detector training only. B feeds more stages than T, and unequal targets also break the tie between swapping two days | Fixed |
| D17 | Leakage check: 64-bit DCT perceptual hash of a 32×32 masked-luma thumbnail; candidate if the hash distance ≤ H, near-duplicate if the thumbnail correlation is also ≥ R. H and R are the 95th/5th percentiles over consecutive video frames (known duplicates). False alarms are measured on the full lake × sea search | Open-water scenes look alike, so a hash alone over-reports. Deriving thresholds from known duplicates keeps them independent of the cross-split result | Fixed |
| D18 | Images: the compressed JPEG version (full resolution), streamed. `scan-images` reads each image once and keeps only near-nadir (≤ 30°) `mavic_4k` frames on disk | 7.4 GB vs 92.5 GB as PNG; the laptop has about 23 GB free and no Kaggle account is set up yet. Later stages re-fetch what they need; cloud notebooks fetch from the share directly | Fixed |

---

## 25. Open decisions (resolve early; log answers in §24)

- Detector framework and its licence (decide together with the repository licence).
- θ_train (M2) and θ_lib (M4) off-nadir limits.
- Swimmer class definition: merge any subclasses? Treat life-saving appliances as distractors?
- Boundary rule for swimmers straddling cells.
- Known vs unknown K as the default.
- MaskablePPO vs a custom PPO.
- Allocentric map with coordinate channels vs egocentric map as the default observation.
- TensorBoard vs W&B.

---

## 26. Glossary

| Term | Meaning |
|---|---|
| GSD | Ground sample distance: metres of ground per image pixel |
| Footprint | Ground area covered by one nadir image at a given altitude |
| Level | One of the discrete altitudes, defined by footprint width in cells |
| Lattice | The view-centre positions reachable at a given level |
| Belief map | Per-cell probability of containing an undiscovered target |
| Sensor model | P(detection outcome \| target present or absent, level); turns detections into belief updates |
| r_ℓ, f_ℓ | Cell-level recall and false-alarm probability at level ℓ |
| LR | Likelihood ratio used in the Bayes update |
| Calibration / ECE | Whether scores behave like probabilities; expected calibration error |
| Prior | The agent's initial target-location distribution (drift model, last known position) |
| Distractor | A non-target object that can trigger false detections (boats, buoys, life-saving appliances) |
| False clear | A cell containing a target that the belief has driven near zero |
| Auto-declare | A cell is declared automatically when p > τ_declare; ground truth then resolves it |
| IPP | Informative path planning: planning paths that maximise information under a budget |
| SITL | Software-in-the-loop: the PX4 autopilot running as a program against a simulator |
| Offboard mode | PX4 mode in which an external program streams setpoints |
| uXRCE-DDS | The middleware bridge between PX4 and ROS 2 |

---

## 27. References

**Data and detection**
- Varga, Kiefer, Messmer, Zell. *SeaDronesSee: A Maritime Benchmark for Detecting Humans in Open Water.* WACV 2022. arXiv:2105.01922. Dataset: seadronessee.cs.uni-tuebingen.de
- *Maritime Small Object Detection from UAVs using Deep Learning with Altitude-Aware Dynamic Tiling.* arXiv:2511.19728. Related CV work using SeaDronesSee altitude metadata.
- Akyon, Altinuc, Temizel. *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection.* ICIP 2022.

**Informative path planning and search**
- Meera, Popović, Millane, Siegwart. *Obstacle-aware Adaptive Informative Path Planning for UAV-based Target Search.* ICRA 2019. arXiv:1902.10182. Target search with height-dependent sensor performance.
- Rückin, Jin, Popović. *Adaptive Informative Path Planning Using Deep Reinforcement Learning for UAV-based Active Sensing.* ICRA 2022. arXiv:2109.13570.
- *Deep Reinforcement Learning with Dynamic Graphs for Adaptive Informative Path Planning.* arXiv:2402.04894.
- *Learning-based Methods for Adaptive Informative Path Planning* (survey). arXiv:2404.06940.
- IMO/ICAO. *IAMSAR Manual, Vol. II* — standard search patterns (expanding square, sector, parallel track).
- DTU. *Autonomous aerial robotics for search and rescue in man overboard* (research project). orbit.dtu.dk/en/projects/autonomous-aerial-robotics-for-search-and-rescue-in-man-overboard/ — closely related application work.

**Calibration**
- Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017.
- Küppers, Kronenberger, Shantia, Haselhoff. *Multivariate Confidence Calibration for Object Detection.* CVPR Workshops 2020.

**Reinforcement learning**
- Schulman et al. *Proximal Policy Optimization Algorithms.* 2017.
- Huang, Ontañón. *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.* FLAIRS 2022.
- Ng, Harada, Russell. *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.* ICML 1999.

**Deployment**
- PX4 User Guide (current version): ROS 2 integration, offboard mode, Gazebo simulation.
