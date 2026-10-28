# Running and resuming M8

The [initial protocol](ppo-protocol.md) uses CPU MaskablePPO, a custom CNN and local
TensorBoard. All code/configuration is committed; caches, checkpoints, TensorBoard
events and episode logs remain gitignored. Numeric results include per-world
metrics; full selected trajectories live in `runs/ppo/m8-initial/selected-results.json`.
All six initial runs and the failure diagnosis
are complete; M8 remains in progress because the configuration needs revision.
Read [the diagnosis](ppo-diagnosis.md) before launching further training.

## Completed initial suite

The initial suite was launched from `321a1c9` on 2026-10-28 and has completed.
It ran V3 then V0, each with seeds 17, 29 and 43, serially. Each seed had 131,072
transitions and evaluated all 40 rl-val worlds every 16,384 steps. Preserve these
runs; they do not need restarting. The commands below document recovery for an
interrupted run, not an instruction to repeat the completed suite.

`runs/ppo/m8-initial/status.json` records suite progress. `suite.log` and
`suite.err` capture the launcher; `V3-seed17.log` (and the other seed logs) record
individual runs. Each seed subdirectory contains a content-addressed run folder
with its plan, input hashes, checkpoints, numeric evaluations and metadata.

Read progress in PowerShell:

```powershell
Get-Content runs/ppo/m8-initial/status.json
Get-Content runs/ppo/m8-initial/V3-seed17.log -Tail 12
```

Do not launch a second copy while the suite is running. If it was interrupted,
check its error log and relaunch the same command from the repository root:

```powershell
uv run python -u scripts/run_ppo_suite.py
```

This preserves completed runs and resumes unfinished runs from complete
checkpoints. Source/configuration changes create new identities; never combine
them. Checkpoints preserve policy, optimizer, worker worlds/beliefs, all RNGs and
the last observations. Do not load untrusted checkpoint or state files.

For a long resumed run, use a hidden detached process with distinct launcher logs:

```powershell
Start-Process -FilePath (Get-Command uv).Source -ArgumentList @('run','python','-u','scripts/run_ppo_suite.py') -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput 'runs/ppo/m8-initial/resume.log' -RedirectStandardError 'runs/ppo/m8-initial/resume.err'
```

## Inspect saved results

```powershell
uv run tensorboard --logdir runs/ppo/m8-initial --host localhost
uv run python scripts/report_ppo.py --partial
```

TensorBoard stays on the local machine. The partial report goes inside the suite
directory as `progress-report.md`, `.json` and `learning-progress.png`; incomplete
runs are visible but cannot be mistaken for completed results.

After all six runs finish:

```powershell
uv run python scripts/report_ppo.py
```

The detached `scripts/watch_ppo_report.py` companion can queue this command for
the existing suite. It only waits and reports; it cannot launch or restart
training. Its outcome is written to `report-status.json`, with stdout/stderr in
`report.log`/`report.err`. It stops without a final report if any seed fails.

This checks matching sources/inputs across seeds, validation-world completeness,
checkpoint hashes and selection rules before writing `docs/ppo-results.md`,
`.json` and `docs/figures/m8_learning.png`. It reports every seed, including poor
runs. The selected checkpoint paths/hashes are recorded in the JSON. No test
worlds are loaded by training or reporting.

Review finite losses, learning-curve variation, comparisons with B3/B5 and the
zero-belief diagnostic before declaring M8 complete. A diagnostic failure must
be investigated; additional training requires a new logged configuration. A
short successful smoke run establishes engineering correctness, not stable
learning or superior search performance.
