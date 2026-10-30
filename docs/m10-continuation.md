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
