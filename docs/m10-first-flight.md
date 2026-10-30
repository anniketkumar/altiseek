# M10 first flight: passed, further work stopped

On 2026-10-30 the owner limited this increment to clearing the message-build
warning and completing one first flight. Both are complete. No camera, detector,
search-policy integration or transfer evaluation was started. Full M10 remains
incomplete, and no further job is queued.

Final regression check: **333 tests passed in 35.53 s**, including lint and
formatting. The flight entry script also passes Bash syntax validation.

## Clock-skew warning

The original px4_msgs build reported a generated C file 0.18 s ahead of the
clock. Before rebuilding, an audit found no future-dated files in the ROS
workspace's source, build or install trees. A clean-object rebuild with
`colcon build --packages-select px4_msgs --executor sequential --cmake-clean-first`
completed in 6 min 35 s without that warning or other stderr. Python type-support
imports passed for status, local position, vehicle command, offboard mode and
trajectory setpoint. Evidence: `data/m10/setup/message-rebuild.log` and the
recorded check script in that directory. No system-clock adjustment or source
timestamp rewriting was needed; this does not claim WSL can never drift again.

## First-flight evidence

The headless X500 flew in the bundled Gazebo default world with the pinned
PX4/ROS/DDS stack in [deployment.md](deployment.md). Live ROS position/status
and landing telemetry drove the check; this was not a cached search episode.

| Check | Observed result |
|---|---|
| Flight run | `20261030T084811Z` |
| Commanded climb | 5 m above initial local position |
| Hold requirement | Offboard, armed, position error <0.5 m and speed <0.3 m/s continuously for 10 s |
| Recorded hold to land command | 10.021 s; 191 hold samples |
| Maximum recorded hold position error | 0.450 m |
| Maximum recorded hold speed | 0.297 m/s |
| Landing | PX4 reported landing, then automatic disarm; final telemetry landed=true, arming state=disarmed |
| Total controller duration | 48.546 s, including startup/telemetry wait |
| Cleanup | All test-owned processes stopped; no PX4, Gazebo or DDS agent left running |

[Numeric evidence and hashes](m10-first-flight.json) identify the preserved
telemetry and logs under
`/home/aniket/altiseek-m10/flights/20261030T084811Z` on D:. The independent
saved-record audit checked hold duration, mode/arming states, error and speed
limits, landing and disarm. PX4's own log confirms takeoff, landing and disarm.
The local audit script is `data/m10/audit_flight.py`.

Two preceding attempts never armed: the first used the wrong PX4 startup-data
directory, and the second lacked the ground-station heartbeat required by PX4.
Their logs remain in `20261030T084539Z` and `20261030T084659Z`. The final runner
passes the build's `etc` directory and owns a local MAVLink GCS heartbeat.
Arming health checks were retained. The remote-control configuration explicitly
uses offboard operation without a physical RC handset.

ROS domain 49, DDS UDP port 8899 and a separate Gazebo partition isolate the
test. PX4's DDS agent address and MAVLink heartbeat target are loopback. A
150 s controller deadline initiates landing, with a further 45 s landing
confirmation limit; the supervisor also bounds execution and cleans up owned
process groups. A process lock prevents a second first-flight supervisor.

Gazebo emitted upstream `gz_frame_id` SDF extension warnings and ROS reported
the supported-but-deprecated `ROS_LOCALHOST_ONLY` setting. These are separate
from the resolved clock-skew warning. They did not prevent the flight. No
camera-rendering quality, physical flight performance or search transfer is
claimed from this check.

## Reproduction (only when another flight is authorized)

From the repository in WSL, `bash scripts/first_flight.sh` starts exactly one
bounded headless flight and cleans up afterward. It uses system ROS Python plus
the built px4_msgs overlay; the heartbeat uses the separate PX4-build venv.
Native sources, assets and libraries are already installed, so this flight
requires no internet or Kaggle quota. Saved logs remain on D: and the latest
path is written to `data/m10/first-flight-path.txt`.

The owner requested stopping after this first flight. Do not proceed to the
remaining M10 acceptance sequence without a new instruction.
