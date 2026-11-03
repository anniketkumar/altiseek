# M10 live accounting and B calibration protocol (D52)

This freezes B integration and calibration acquisition. It does **not** authorize
using outcomes to tune on T, and it is not yet the held-out transfer protocol.
The owner's full M10 authorization remains active; successful stages continue
without another permission request.

## Measured search budget

The five levels, 500 hover-equivalent-second battery, known K, north-up maps,
frozen detector and first-look rule remain unchanged. Search begins at the same
seeded edge lattice node selected by the shared environment at L5. Initial
positioning/takeoff and final landing are preparation/recovery, excluded from
search metrics and separately timed. No target coordinates choose that launch.

After first settling begins, every simulator-time interval counts, including
settling, inference and policy computation. Charge elapsed time at the shared
climb multiplier 1.5 when measured vertical speed exceeds +0.2 m/s; otherwise
charge one hover-equivalent second per second. The deadband excludes hover jitter.
The shared `EnergyModel.measured_interval` implements this rule. Sequential
constant-speed movement agrees with the original model; concurrent movements
and actual waiting can differ, which is part of the deployment gap. M7–M9's
ideal accounting and their published results remain unchanged.

The continuous meter runs in pose callbacks independently of inference. Expiry
clips the last interval and prevents subsequent evidence updates; the vehicle
lands. If a proposed nominal move is already unaffordable, stop before flying
and report actual energy consumed, retaining the unused remainder. Do not report
unspent time as measured flight. Report both simulator time and wall time.

PX4 horizontal and vertical limits are 10 m/s and 3 m/s, matching the shared
nominal settings. Settling requires position error <0.5 m, speed <0.3 m/s,
world-heading error <1 degree and tilt <2 degrees, sustained for 2 simulator
seconds. Images must match a measured Gazebo pose within 0.04 s and the actual
camera goal within 1 m. Pose is simulator ground truth; this does not test
real-world localization. Only whole cells within the actual image update belief.

## Calibration acquisition

Use all 40 existing **sensor-fit B** worlds, set `5dc45d0e1ea5afbd`, preserving
their compositions and IDs. Fly every lattice location once at each level:
121/49/25/16/9 views, 220 per world, **8,800 views total**. Order each level in
serpentine rows; progress from L1 through L5. Calibration flights are acquisition
sweeps, not 500-second search episodes, and have a separate bounded runtime.

First fly ten pilot views on the first world: the first and last sweep location
at every level. Verify capture, pose/goal, saved checksums and clean landing.
Then resume the full sweep, reusing these exact views, and continue all worlds.
Each completed view is atomic and checksummed. A failed flight stops the suite;
an interrupted suite resumes verified views without repeating them.

Use actual fully observed cells for occupied/empty trials and maximum raw
swimmer score per centre cell. Fit with the existing sensor fitting code and
unchanged thresholds/hyperparameters, recording a distinct Gazebo acquisition
contract and model ID. Single-view acquisition does not estimate conditional
second-look recall; its zero-denominator diagnostics remain explicitly unmeasured.
No refit is accepted until the complete set passes integrity checks.

## Runtime architecture

Gazebo image and pose bridges and PX4/DDS remain separate processes. On this
CPU-only machine the detector, belief, policy and supervisor are co-located as
separate software components rather than additional ROS transport nodes. ROS
callbacks keep the offboard stream and continuous meter active while an inference/
policy worker performs stop-and-shoot decisions. This changes the transport
layout described in the original §17 table, not the observation or policy rules.
Only the supervisor uses world truth to verify declarations. Native logs, images
and resumable state remain on D:; no dataset imagery enters git.
