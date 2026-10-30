# M10 deployment work

M10 started on 2026-10-30 after D48 accepted M9. This document tracks setup;
it is not a completed transfer study or a frozen held-out evaluation protocol.
No M8/M9 training or evaluation is restarted.

The initial implementation check passed all **333 tests in 84.80 s**, including
ENU/NED basis, altitude, heading and round-trip tests plus lint/format checks.
All three setup shell scripts pass Bash syntax validation. These checks do not
substitute for building and flying the simulator.

## Stack and storage (D49)

Use the existing Ubuntu 24.04 WSL distro, ROS 2 Jazzy and Gazebo Harmonic.
These match the [PX4 ROS guide](https://docs.px4.io/main/en/ros2/user_guide)
and [Ubuntu setup guide](https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu).
The sources are pinned as follows:

| Component | Version | Commit |
|---|---|---|
| PX4-Autopilot | v1.17.0 | `d6f12ad1c4f70ad3230afd7d86e971421e02fef4` |
| px4_msgs | release/1.17 | `86d8239e962f6939e05c3737784f60c02fa884db` |
| Micro-XRCE-DDS-Agent | v2.4.3 | `73622810d984349b80bbac0ef55fc0b694d62222` |
| ROS apt source package | 1.3.0, noble | Official ros-infrastructure release |

System package patch versions are recorded after installation in
`data/m10/setup/system-packages.txt`; these are not claimed installed until
the setup/build checks complete. Build dependency versions are recorded in
`px4-python-packages.txt` alongside it.

The WSL filesystem is on D:, with about 111 GB host free at preflight. Keep
sources, native builds and simulator assets under `/home/aniket/altiseek-m10`;
C: had only about 4 GB free. WSL exposes 7.6 GiB RAM and 2 GiB swap. Limit native
builds to two workers. WSLg exposes display `:0`, Wayland and a render device;
actual Gazebo rendering support and speed still require a measured check.

`scripts/setup_m10_system.sh` installs simulation dependencies as root. It
uses the official ROS/OSRF repositories, without a general OS upgrade or the
hardware cross-compilation toolchain. Unlike the upstream setup script, Python
packages are not installed into system Python. `scripts/build_m10_stack.sh`
uses `/home/aniket/.venvs/px4-build` with system packages visible. Existing
AltiSeek dependencies and its venv stay separate. The ROS runtime integration
with AltiSeek still needs an explicit import/ABI check before live perception.

## Current jobs and recovery

System installation and source/build jobs are detached, with process locks.
Read these files before launching anything:

- `data/m10/setup/state`, `system.log`, `launcher.pid`;
- `data/m10/setup/build-state`, `build.log`, `build.pid`.

The build downloads sources while installation runs, then waits for
`system_ready`. It stops on a failed prerequisite. Only `ready` means PX4,
the DDS agent and px4_msgs all built successfully. An existing source directory
must match the pinned commit and have no tracked changes. Git submodules and
native build outputs are reusable on recovery; never launch a duplicate.

From an interactive WSL shell in the repository, the underlying commands are:

```bash
sudo bash scripts/launch_m10_setup.sh
bash scripts/build_m10_stack.sh
```

These are foreground commands; long runs must be detached with redirected logs
by the launcher/operator. The initial launch used Python `Popen` with
`start_new_session=True`, closed stdin and file-backed stdout/stderr, through
WSL root for system installation and the normal `aniket` user for builds.
Do not interrupt apt/dpkg mid-install to pause unrelated project work.

## Acceptance sequence

1. Verify exact versions, ROS imports, DDS agent, PX4 build and Gazebo rendering.
2. Complete a basic simulated takeoff, offboard hold and landing, with telemetry
   and bounded timeout. Use the shared ENU/NED conversions and explicit local
   origin; prove north/east/up and yaw conventions before search commands.
3. Add a nadir 640 px camera and check checkerboard GSD at all five altitudes.
   Hold camera north-up and use stop-and-shoot with measured pose/velocity gates.
4. Export B worlds in bounded texture tiles. Verify live camera, detector and
   belief/policy imports. Reuse existing geometry, sensor, energy and policy code.
5. Measure CPU inference and rendering latency. Establish a B-only integration
   pilot and freeze deployment settings, scenario subset and policy selection
   before any new T transfer evaluation. Do not choose policies from M9 T scores.
6. Run matched cache/Gazebo comparisons, then B-only Gazebo sensor refitting and
   the prescribed comparison. Save all metrics, exact versions and a demo video.

Basic flight alone does not complete M10. RQ3, the sensor-refit experiment and
the demonstration remain required. No real hardware is involved. Use fresh
artifact IDs for live-camera/refitted-sensor results and retain all M9 findings.
