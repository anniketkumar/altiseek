# Full M10 continuation (D51)

The owner authorized the full remaining M10 scope on 2026-10-31, lifting the
D50 stop after first flight. Preserve that successful flight and all M8/M9
results. No completed training or evaluation is restarted.

## Work and connectivity rules

Proceed through camera geometry, tiled B-world export, actual-pose projection,
live detector integration, shared belief/policy integration, B flight pilots,
frozen transfer evaluation, B Gazebo sensor refit, comparison and demo video.
M10 is complete only after all of those acceptance requirements are reviewed.
No transfer protocol is frozen yet; B-only pilots must first establish valid
geometry, energy/time accounting, settling gates and achievable throughput.

The owner cannot guarantee continuous internet. Record failed online operations
in `data/m10/continuation/network-events.jsonl`, including the operation, error
and retry state. Continue independent offline work; retry required operations
when connectivity returns. Missing inputs or failed scientific checks remain
pending, never silently waived. Record an inaccessible documentation page as
such, not as proof the whole internet is down. Prefer the pinned installed
source/schema when it answers the question. Commit tested local increments and
record any pending push if GitHub is unavailable.

Long jobs remain detached, bounded/resumable, with locks and local logs. Do not
submit duplicates after a connection loss. Native images, textures, logs and
videos stay on D: inside `/home/aniket/altiseek-m10`; only numeric reports and
source code may enter git. No new heavy dependencies are currently needed.

## Initial findings

- The WSL OpenGL backend is **llvmpipe**, software rendering, not accelerated.
  Camera rendering and CPU inference share the laptop's resources. A requested
  simulation speed factor is not an achieved throughput measurement.
- ROS image/PX4 type support imports successfully alongside pinned NumPy 2.1.3
  and CPU torch 2.11.0 in the existing AltiSeek venv after sourcing ROS and the
  px4_msgs overlay. No package replacement was required.
- Detector `700e0ff3455259e8` loaded locally and ran a blank-frame speed smoke:
  0.112/0.267/0.103 s per warmed inference with two torch threads. This tests
  dependencies, not detection quality or realistic combined throughput.
- The camera gate uses a synthetic 5 m checkerboard, asymmetric NE/SW markers,
  the frozen 640 px / 80 degree camera and all five fixed altitudes. It must
  demonstrate north-up orientation and measured GSD within 3% on both axes.
  Images pass through Gazebo and `ros_gz_bridge`; numerical predictions alone
  do not satisfy this gate. Failed attempts are preserved with logs.

Pointers: `data/m10/camera-check-path.txt`,
`data/m10/continuation/camera-check.log`, and
`data/m10/continuation/runtime-probe.json`. Current stage reports determine
progress, not the historical setup markers. Successful static camera checks
will not by themselves establish flight-camera pose synchronization or transfer.

## Static camera gate passed

Run `20261030T170658Z` passed all five levels in 20.9 s using the installed
OGRE renderer. The north-up NE/SW markers are correct. Measured GSD errors are
0%, 0.78%, 0%, 1.23% and 1.59%; see `m10-camera-check.json`. The rendered L1
image was visually checked. The full suite passed 337 tests in 49.62 s.

Earlier OGRE2 attempts produced invalid materials/shader warnings and incomplete
capture on llvmpipe. The exporter needed explicit mesh normals; the material also
needed unit ambient reflectance to avoid darkening the source texture. These failed
attempts remain in native camera-check storage. Use `--render-engine ogre` for the
verified local configuration. This renderer choice is part of the deployment
configuration, not a change to the frozen Python camera or historical results.

## Flight camera and live B hover integration

Synthetic camera flight `20261030T221404Z` completed takeoff to approximately
30 m, a 20 s hold, landing and disarm. All 23 images matched Gazebo model-pose
timestamps exactly; projected marker error was at most 0.116 m. Visual review
found a small magnetic-heading/world-north offset, despite the north command.
This initial diagnostic is retained in `m10-flight-camera.json`.

The controller now compensates from measured Gazebo world heading and gates
capture on heading error below 1 degree and tilt below 2 degrees, in addition
to the existing position/speed settle checks. Follow-up `20261030T221659Z`
passed 20 image/pose pairs, maximum heading error 0.016 degrees and marker
error 0.135 m; see `m10-flight-camera-north.json`. Both flights landed and
disarmed, and all owned processes stopped. These use simulator ground-truth
pose, not a claim of real-world localization accuracy.

B world `003b0c54f581c65b` was composed with the existing world compositor and
exported without resampling, in 20.7 s. Native assets and checksums are under
`/home/aniket/altiseek-m10/worlds/003b0c54f581c65b`. The exporter rejects T and
checks existing exports before reuse.

Live perception flight `20261030T222009Z` passed: 20 frames processed during
a 25 s hover, 376 belief cells updated, 1–2 raw swimmer detections per frame,
then landing/disarm. See `m10-perception-hover.json`. Detector, camera finishing,
centre-cell projection, sensor and belief updates use existing shared contracts.
Only fully visible cells update or consume a first look. An inference worker
keeps ROS offboard heartbeats independent. Mean processing including image/evidence
logging was 0.610 s, range 0.258–3.487 s including first-inference startup; this is
not full-search throughput. A prior startup failed from a ROS property-name
collision before takeoff; its logs are preserved.

This is a hover integration pilot, not closed-loop search, sensor fitting or
transfer evidence. It uses a B target-centred integration view explicitly; no
performance claim is made from that selection. Shared search evidence application
is shared with the waypoint-policy integration. The final full regression suite
passed 347 tests in 40.85 s, including cached environment regressions, live-source
cache exclusion, partial-cell first-look protection, projection, format and lint.

## First closed-loop policy pilot

B5 pilot `20261030T222617Z` (the exact native path is retained alongside its
launcher logs) observed at L1, selected ASCEND, flew to the L2 goal and observed
again. The shared declaration logic then verified the sole target and terminated
the episode; it correctly stopped before the three-view cap. Actual camera-goal
errors were 0.224 m and 0.265 m. Both images matched pose timestamps exactly.
Landing/disarm completed. See `m10-search-pilot.json` for the numeric trace.

The pilot starts near a known B target for integration only and still uses ideal
M7 energy accounting. It is not a transfer performance result. The live adapter
never loads cached detections, preserves shared policy observations/masks and
verification, and supplies an actual-footprint mask. Measured flight energy,
budget expiry during movement, normal edge launch, ROS component separation,
calibration sweeps and the frozen transfer protocol remain outstanding.

The original generic pilot flight-summary field `hold_s=1000` was the requested
hold timeout, not the observed hold duration; policy completion landed earlier.
The logger now records measured `hold_s` separately from `requested_hold_s`.
The original files are retained; do not interpret that field as a 1000 s hover.

`scripts/run_deploy_pilots.py` runs the same bounded B integration with each of
the three frozen V3 PPO checkpoints. It verifies checkpoint hashes, preserves
all seeds, resumes only verified completions and stops on any failure. Its identity
includes source hashes and the frozen policy manifest. Check
`data/m10/ppo-pilots-path.txt` and that directory's `status.json` before launching.
This suite does not start transfer evaluation or retrain a policy.

## D52 continuation

All three frozen PPO seed pilots (17/29/43) completed successfully in suite
`387e5e2a331307e3`; numeric records are in `m10-ppo-pilots.json`. Do not rerun them.
The normal seeded L5 edge-launch check with a 25 s measured budget also passed,
including landing/disarm (`20261031T041908Z`, `m10-measured-budget.json`). It used
21.61 hover-equivalent seconds over 21.02 simulator seconds and stopped before
an unaffordable nominal move. It did not physically exhaust the battery; exact
cutoff/no-late-evidence behavior is separately covered by regression tests.

The full suite passed **353 tests in 37.56 s**. D52 freezes live accounting,
settling, velocity limits and complete B acquisition in `m10-live-protocol.md`.
`scripts/continue_m10.py` automatically proceeds from a full 500 s B5 check to
the ten-view calibration pilot, all 8,800 B calibration views and the shared
sensor-model fit. It verifies existing stages/views, stops on failure, and ends
at calibration review. Native continuation state is pointed to by
`data/m10/continuation-path.txt`; consult that status instead of assuming a job
is still running. No held-out transfer run is launched by this wrapper.
