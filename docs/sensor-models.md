# Sensor models (M6)

The fitted model is `fcf8cdae58ca3947`, tied to detector `700e0ff3455259e8`
and sensor-fit world set `5dc45d0e1ea5afbd`. The registry points to
`data/sensor_models/fcf8cdae58ca3947.json`. Its `.fit.json` sidecar contains raw
rates, cluster-bootstrap intervals, conditional recall and reliability summaries;
`.meta.json` records resolved settings, source hashes, code revision and packages.
The model JSON contains the smoothed runtime rates, temperatures and histograms.

The owner confirmed centre-cell matching (D34), the existing five levels and
500 s battery, and first observation per cell/level only (D35). See the frozen
[protocol](sensor-protocol.md) and the [numeric results](sensor-results.md).

## Reproduce

From the repository root, with the accepted M5 caches and world specs present:

```powershell
uv run python scripts/fit_sensor_models.py
uv run python scripts/fit_sensor_models.py --evaluate data/sensor_models/fcf8cdae58ca3947.json
uv run python scripts/evaluate_sensor_real.py --model data/sensor_models/fcf8cdae58ca3947.json
uv run python scripts/report_sensor_models.py --model data/sensor_models/fcf8cdae58ca3947.json --real data/sensor_models/real-T/cb1a5bb437b21ea3/report.json
```

Fitting reads only sensor-fit B. `--evaluate` loads an existing frozen model and
reads test caches without refitting. The real-T script uses the historical M4
view selection, reconstructs its exact geometry and runs inference on CPU. Its
per-view files are atomic and checked against the plan, model and source-image
hashes on resume. Re-run the same command after interruption. A changed plan or
matching version gets a separate directory. Downloaded images remain gitignored.

Run the real-T command as a hidden detached process on Windows, with stdout and
stderr redirected under `data/sensor_models/real-T/`, for session independence.
The source-frame fetch skips existing images. No new Kaggle token or GPU quota
is needed for M6.

## M7 integration contract

```python
from pathlib import Path
from altiseek.calibration.model import SensorModel

sensor = SensorModel.load(
    Path("data/sensor_models/fcf8cdae58ca3947.json"),
    detector_id="700e0ff3455259e8",
    fit_world_set_id="5dc45d0e1ea5afbd",
)
posterior = sensor.posterior(prior=0.01, level=4, score=0.7, variant="V3")
```

Levels are zero-based, so `4` means L5. Pass raw scores, reduced to the maximum
swimmer score in each observed cell. Pass `None` or zero when no retained
detection lands there. V0–V2 apply raw tau; V3 uses all scores at raw tau_min
and calibrates internally. Array inputs broadcast. The return value is clipped
away from zero and one; invalid probabilities and unknown levels/variants fail.

Use `calibration.outcomes.cell_outcomes` for score reduction and
`observed_bounds` for cached footprints. Full world target centres define
occupied cells for evaluation. The model loader can additionally check
`cache_contract` against the cache header with `world_id` and `cache_id` removed.
Never skip detector/geometry compatibility checks when wiring the environment.

The posterior function is stateless. The environment must gate unresolved cells,
enforce first observation per `(cell, level)`, suppress detections from already
found targets, and track variant draws. Those stateful behaviours belong to M7.
Do not interpret repeated calls as independent evidence automatically.

V3 uses smoothed conditional histograms; high raw score need not imply higher
occupancy likelihood when distractors produce confident false detections. A
calibrated score is not a replacement for the prior or the likelihood ratio.
No claim that V3 improves policy performance is possible before M9.
