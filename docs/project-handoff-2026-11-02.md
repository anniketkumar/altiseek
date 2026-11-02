# Project continuation handoff — 2 November 2026

> **Update (D55, later on 2 November 2026).** The calibration job described below
> completed at 08:33 Singapore time: 40/40 worlds, 8,800 views, no deferred
> worlds, refit sensor `caba50dfc568562d`. It was reviewed and its integrity
> accepted in `docs/m10-calibration-review.md`. No calibration process is
> running, and calibration must not be relaunched. The "immediate state" and
> remaining-work items 1–2 below are historical. `registry/manifest.json` is no
> longer dirty. Transfer status is tracked in the newest AGENTS.md entry.
>
> **Transfer (D56).** The worktree draft is preserved on branch
> `m10-transfer` and finished on main. Protocol v3 `750d20fa9df98142`
> (`configs/eval/m10_transfer_v3.json`) is frozen. v1 was superseded before any
> T run, and v2 after a classifier gap (D57). Runner root: `/home/aniket/altiseek-m10/transfers/<protocol_id>/`
> (`status.json`, `failures.jsonl`, `preflight/`, `cached/`, `live/`). Launch
> with `scripts/launch_m10_transfer.py <protocol>`, adding
> `--stop-after-preflight` for the B gate only. It refuses duplicates and
> reuses verified completions. Never edit `src/`, `configs/` or execution
> scripts while it runs; the hash check stops it.

Snapshot taken around **01:52 Singapore time, 2 November 2026**.
Read `CLAUDE.md` in full, then the latest entries at the top of `AGENTS.md`.
Historical stop/pending notes in those files are superseded by later decisions.
This handoff records the continuation from the completed first-flight boundary
through the current M10 calibration and recovery work. It is not M10 acceptance.

## Immediate state: preserve and inspect before acting

**A detached calibration job is running. Do not launch another or restart it.**

- Main is pushed through **`ec45a6f`**.
- Acquisition identity: **`cf148ad98784906d`**.
- Native root: `/home/aniket/altiseek-m10/continuations/cf148ad98784906d`.
- 31/40 worlds are complete; 6,820 views belong to completed worlds.
- About **6,960/8,800 views are saved**, including partial worlds. Counts change.
- Active world: `a626e23c05e26f80`; latest observed 37/220 views.
- Active flight: `/home/aniket/altiseek-m10/camera-flights/20261101T174808Z`.
- Scheduler PID observed: 396; calibration adapter PID: 1012. These are only
  snapshot identifiers; verify live command lines before any process action.
- Scheduler: `scripts/continue_m10_resilient.py`.
- Actual controller: `scripts/calibration_recovery.py` (D54).
- The current run has progressed beyond this world's earlier failures, but no
  complete D54 world or successful corrective retry had yet been reviewed.
- **M10 transfer evaluation has not started. M10 and M11 are incomplete.**

The owner's request for this document is a handoff, not a request to stop the
detached computation. Leave the job running. The owner wants existing progress
preserved and repeatedly said not to restart completed work.

## Owner instructions and practical expectations

- Complete full M10, including deployment, transfer study, reports and demo.
  The earlier request to stop after the first flight was lifted by D51.
- Internet may be intermittent. Record failed online operations, continue
  independent offline work and retry online operations later.
- Log isolated flight failures, defer those worlds, continue others, then retry
  incomplete worlds. Do not silently accept invalid captures or drop worlds.
- Integrity failures, unknown errors, three consecutive flight failures, or
  exhausted retry passes still require diagnosis. Do not simply relax checks.
- Ask only for genuinely missing input. Existing choices remain authorized.
- Commit and push finished, tested increments to `main`; no attribution trailers
  or generated-by notes. Never commit dataset imagery, caches or checkpoints.
- Keep the owner informed. Earlier optimistic estimates and silent stops caused
  frustration. Give actual status and distinguish calibration ETA from full M10.
- Local jobs do not need internet, but the laptop must stay powered and awake.
  Windows recorded lid-triggered Modern Standby on Nov 1 at 05:30–10:24 and
  15:52–16:04 Singapore time. Sleep paused WSL work; wake generally resumed it.
  Do not equate wall-clock time with active processing time.
- There is no automatic user notification service. A detached recovery process
  is not a continuously supervising assistant. Do not promise unsolicited alerts
  or claim silence means success.

## What was inherited

M0–M9 were already accepted. Preserve their artifacts, findings and qualifications:

- Split `783b4403804291a2`; detector `700e0ff3455259e8` (YOLO11s).
- Original sensor `fcf8cdae58ca3947`; sensor-fit B set `5dc45d0e1ea5afbd`.
- M5 caches, M6 sensor work, M7 baselines, and all M8 training are complete.
- M8 has qualified acceptance: negative PPO results, V0 convergence uncertainty
  and the historical zero-belief gate failure remain documented. No T tuning.
- M9 v1 `341aadf1366d8f05`: 25,920 episodes; v2 `1c19752957aabaa0`:
  18,720 new episodes plus 4,320 reused nominal references. All 24 ablation
  training runs and 320 additional GPU caches are complete. Do not rerun them.
- T worlds contain no distractors. B stress evidence is explicitly B-only.
- D50 first flight passed and originally stopped further work at the owner's
  request. Run `/home/aniket/altiseek-m10/flights/20261030T084811Z` performed
  takeoff, 10 s offboard hold at 5 m, landing and disarm. Preserve it.

## Work completed during this continuation

| Commit | Work |
|---|---|
| `31fcff6` | D51 full continuation; tiled static camera geometry verified |
| `67b4060` | Live camera evidence, coordinate alignment, shared belief and B5 waypoint integration |
| `d814af7` | D52 measured flight budget, normal launch, all-seed pilots and resumable B calibration/refit |
| `f621950` | D53 bounded deferred recovery scheduler |
| `7815de7` | Recovery for rejected image/pose synchronization captures |
| `e7adda0` | Recovery for stale telemetry, only with corroborating flight report |
| `ec45a6f` | D54 bounded same-waypoint correction after rejected captures |

Completed checks and artifacts:

- Static camera tests at all five altitudes passed with OGRE1; largest GSD error
  about 1.59%. OGRE2 had problems; accepted runs use `ogre` and CPU llvmpipe.
- Live pose/orientation checks passed. PX4 local magnetic heading differs from
  Gazebo world north; the controller corrects using measured world heading.
- B hover live detector/shared belief integration passed: 20 frames, 376 cells.
- B5 short closed-loop L1→L2 flight verified a target and landed.
- All three frozen V3 PPO seeds 17/29/43 passed short B live pilots in suite
  `387e5e2a331307e3`. Do not repeat them.
- The measured 25-unit battery check used 21.61 energy units before rejecting an
  unaffordable next move; it was not an actual expiry test. Shared expiry and
  late-evidence rejection have regression tests.
- Full-budget B5 and the ten-view calibration pilot passed. The original
  continuation has their report paths and checksums; do not rerun them.

Read `docs/m10-continuation.md`, `docs/m10-live-protocol.md`,
`docs/m10-measured-budget.json`, `docs/m10-ppo-pilots.json`,
`docs/m10-perception-hover.json`, `docs/m10-search-pilot.json` and the camera
reports. These are integration results, not held-out transfer performance.

## D52 acquisition and scientific contracts

- All 40 original sensor-fit B worlds, 220 lattice views/world: 121/49/25/16/9
  across L1–L5. Total **8,800**, one view per location. Pilot views are reused.
- Calibration contract ID: **`0e81d39c16e288b9`**.
- Saved views live under `<root>/views/<world_id>/`, as per-view JSON + NPZ.
  `load_view` checks key/contract/checksum before reuse. Never overwrite a valid
  view or relabel artifacts merely to satisfy identity checks.
- Full-cell visibility is required for belief updates; partial cells do not
  consume their first-look bit. Cached M7–M9 behavior remains unchanged.
- Actual simulator time costs hover energy; climbing above 0.2 m/s uses the
  existing 1.5 multiplier. Include motion, settling, inference and decisions;
  exclude initial positioning/takeoff and final recovery. Budget is 500.
- Normal seeded L5 edge launch; no target-selected start for scored flights.
- Position settling <0.5 m in PX4 coordinates, speed <0.3 m/s, heading <1°,
  tilt <2°, dwell 2 simulator seconds; image/pose mismatch ≤0.04 s and actual
  camera-to-goal error ≤1 m. Thresholds were not relaxed by recovery fixes.
- Gazebo ground-truth pose is used: do not claim localization robustness.
- Detector/belief/policy/supervisor components are co-located with independent
  control callbacks; this documented D52 deviation is not separate ROS nodes.
- Refit only after all worlds pass and all views verify. The model's contract
  adds deployment provenance to the original sensor contract; it gets a new ID.
- Single views do not estimate conditional re-look recall. Zero denominators
  must not be described as measured zero recall. Calibration quality may remain
  poor; preserve negative findings and uncertainty.

## Failure history, diagnosis and recovery behavior

Read `docs/m10-recovery.md` and `docs/m10-position-diagnosis.md`.

1. World 17 stopped after 26 views when actual camera-goal error was 2.011 m.
   Original controller sources were left unchanged and saved views resumed.
2. D53 added a separate scheduler, initially waiting for the original runner's
   lock. Recognized position/timeout failures defer after verified cleanup.
3. World 24 stopped after 109 views on a 0.060 s image/pose mismatch. Recovery
   initially did not recognize that error; `7815de7` added it without changing
   the 0.04 s acceptance limit.
4. Automatic position-failure deferral then worked, but a stale-telemetry flight
   stopped it. `e7adda0` requires the separate flight report to explicitly say
   `Telemetry became stale or invalid`; generic ended-observation errors do not
   qualify on their own.
5. Three subsequent position errors (1.189, 1.578, 1.152 m) triggered the intended
   consecutive-failure stop at **00:55 Singapore, Nov 2**.
6. D54 diagnosis: the last failed flight had PX4 hold error <0.291 m despite
   actual camera error >1 m. A one-shot conversion from world goal to local
   setpoint does not correct estimated-to-world offset changes during travel.
   Saved views showed vertical offsets around 0.57–0.67 m. Stream timing versus
   estimator drift is not uniquely resolved by the existing logs.
7. D54 adapter retries only position/image-pose rejected captures, at the same
   waypoint, at most three attempts. Each call reuses the original `sense`,
   recomputes the setpoint from latest poses and resets settling. Rejected
   frames never enter the calibration collection. Exhaustion still defers the
   world; three consecutive failed flights still stop.

New adapter files: `scripts/{retry_m10_capture,calibration_recovery,run_m10_capture}.py`.
The launch adapter reuses the original lifecycle/cleanup and narrowly redirects
only the calibration controller. Per-flight `capture-recovery.json` records
adapter hashes and `capture-retries.jsonl` records corrections when exercised.
Do not claim all calibration views used one controller revision. D54 preserves
the observation acceptance contract, while documenting changed execution.

Scheduler caveats for the next engineer:

- `status.completed_views` counts only completed worlds; include per-view files
  for partial progress. Counting files is a progress estimate, not a full audit.
- Resume temporarily resets status counters while revalidating completed stages.
  This is not recomputation. Explain it rather than reporting lost progress.
- `deferred_worlds` is not a complete global history across invocations; consult
  `failures.jsonl` and actual missing views. On restart the current failed world
  goes last, while earlier partial worlds can be encountered in original order.
- Recovery logs append, so old tracebacks can remain at the end until new output
  appears. Check current processes, timestamps and status together.
- Manual relaunch after a three-failure stop resets invocation counters. Do not
  use repeated relaunching to defeat the safeguard without reviewing causes.
- If a flight saves every view but fails before completion/landing, a subsequent
  all-skipped controller may have no airborne flight and fail its completion
  gate. This edge case was identified but not fixed; preserve all views and
  review actual completion evidence if encountered.

## How to inspect and resume

Windows repository: `C:\Users\Aniket Kumar\Downloads\AltiSeek`.
WSL distro/user: `Ubuntu-24.04`, `aniket`. Native files live on D: through WSL,
under `/home/aniket/altiseek-m10`. Python is
`/home/aniket/.venvs/altiseek/bin/python`.

Local pointers/logs:

```text
data/m10/continuation-path.txt
data/m10/camera-flight-path.txt
data/m10/continuation/recovery.pid
data/m10/continuation/recovery.log
```

Under the native continuation root inspect `status.json`, `definition.json`,
`contract.json`, `scheduler.json`, `failures.jsonl`, stage JSON/logs and views.
Under the current flight inspect `controller.log`, `status.json`,
`calibration-summary.json`, `flight-summary.json`, `provenance.json`,
`capture-recovery.json` and `capture-retries.jsonl` if present.

Example read-only check from PowerShell:

```powershell
wsl.exe -d Ubuntu-24.04 -u aniket --exec cat /home/aniket/altiseek-m10/continuations/cf148ad98784906d/status.json
wsl.exe -d Ubuntu-24.04 -u aniket --exec pgrep -af calibration_recovery.py
```

**Only if no scheduler/owned flight is active and the failure has been reviewed**:

```powershell
wsl.exe -d Ubuntu-24.04 -u aniket --cd 'C:\Users\Aniket Kumar\Downloads\AltiSeek' --exec /home/aniket/.venvs/altiseek/bin/python data/m10/continuation/launch_recovery.py
```

That helper starts detached `flock -n .../recovery.lock bash .../recover_m10.sh`.
The shell sources `/opt/ros/jazzy/setup.bash` and
`/home/aniket/altiseek-m10/ros2_ws/install/setup.bash`, then runs the resilient
scheduler. The scheduler also holds `continuation.lock`; the flight lifecycle
holds its simulator lock. Do not launch the original continuation as a second job.

Completion automatically fits the B sensor and stops at
`awaiting_calibration_review`. **No transfer watcher is installed.** Further
transfer work is unfinished and will not start by itself after calibration.

## Tests, repository state and unfinished transfer work

D52 full suite: 353 passed. At D54, 368 functional/format tests passed and the
lint test initially failed for annotations; after correction all 18 targeted
capture/recovery/style checks passed. Do not invent a fresh single-invocation
369-pass record. Relevant tests: `tests/deploy/test_m10_capture_retry.py` and
`test_m10_recovery.py`; these cover bounded retry, integrity rejection, deferred
world order, frozen fit ordering and repeated-flight stopping.

Full suite command:

```powershell
wsl.exe -d Ubuntu-24.04 -u aniket --cd 'C:\Users\Aniket Kumar\Downloads\AltiSeek' --exec /home/aniket/.venvs/altiseek/bin/python -m pytest
```

Main has a pre-existing dirty `registry/manifest.json`. Do not stage or discard
it blindly. Windows Git can report permission denied hashing that file after
WSL tests; use WSL to inspect it. Finished increments were committed with explicit
file lists. Windows Git needs write permission to `.git`; shell tools may need
an approved escalation. Never weaken Windows application-control policy.

An **uncommitted** isolated worktree exists:

```text
C:\Users\Aniket Kumar\Downloads\AltiSeek\.worktrees\m10-transfer
branch: m10-transfer; base: d814af7
```

It contains draft transfer authorization, sensor-contract handling, live episode
metrics, `freeze_m10_transfer.py`, `run_m10_transfer.py`, tests and
`docs/m10-transfer-design.md`. The `data` and `runs` entries are WSL symlinks to
main's ignored directories and appear untracked to Windows Git: **never stage
them**. This worktree is not fully tested, committed or ready to execute.

Draft design: 24 T worlds selected by fixed metadata-only hashing, three from
each original family; B5 plus all three frozen V3 PPO seeds; original/refitted
sensor → 192 live episodes, plus 96 paired cached references. No T tuning.
This is a proposal in unfinished code, not a frozen protocol or completed study.
Report/demo scripts called by the runner do not yet exist; neither does an
automatic post-calibration transfer continuation. Account for D54 provenance
and current main commits when finishing this worktree.

Use explicit `PYTHONPATH=<worktree>/src` with the existing venv for worktree
tests. Do not `uv sync` or repoint its editable installation while main runs.
Do not edit frozen acquisition sources in main during an active flight: its
source hashes are checked. New unrelated scripts/docs do not alter that contract.

## Remaining work in order

1. Monitor the current corrected calibration. Verify real corrective attempts
   and a completed world; investigate repeated failures rather than broadening
   allowlists indefinitely. Preserve every prior failed flight and valid view.
2. Complete all 40 worlds, including deferred ones; verify 8,800 records and the
   B sensor fit. Review model quality and provenance, preserving limitations.
3. Finish and test the isolated transfer implementation; reconcile D54 source
   history, freeze protocol/artifact hashes, pass B refit smoke for B5 and all
   three PPO seeds, then run held-out comparison. Do not start T early.
4. Implement numeric reporting with paired whole-world intervals and all seeds;
   compare original/refit/cached performance without tuning on outcomes.
5. Produce a predetermined local demo, respecting the no-imagery-in-Git rule.
6. Review and record M10 disposition honestly. M11 write-up follows.

The remaining live comparison is potentially multi-day on this machine; no
credible total M10 ETA has been measured. Refit/reporting are lighter. No new
detector or PPO training is planned. Avoid quoting calibration time as total time.

## GPU discussion, not an implemented change

Hardware inspection found **Intel Iris Xe**, Windows driver `31.0.101.4032`.
The current Gazebo run uses CPU llvmpipe rendering. WSLg GPU OpenGL, another GPU
PC, or a paid cloud GPU machine were discussed. No GPU configuration was changed
or benchmarked and no cloud resources were purchased. Investigate the existing
Intel GPU first if useful, but do not switch the running renderer mid-collection.
Rendering acceleration does not automatically accelerate CPU detector inference;
validate image/geometry differences before reusing calibration with another
backend. All current M10 dependencies are local; tokens are not needed. Never
print or copy credentials into this handoff.
