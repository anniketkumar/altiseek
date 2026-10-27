# Detection cache (M5)

M5 is complete (2026-10-27): all 320 worlds are cached, live-validated, imported and registered.
D31 records storage and inference choices, D32 fixes cloud input transfer, and D33 fixes counts.

## Production run after the successful pilot

```powershell
uv run python scripts/prepare_cache_notebook.py --production
```

Import `data/kaggle/altiseek-cache-production.ipynb` into Kaggle. Keep it private, select GPU
T4 x2 and enable Internet. Attach the existing `altiseek-m5-inputs` dataset and
`altiseek-detector-v3` notebook output. Optionally attach the successful pilot notebook's
saved output to reuse its two caches. No new dataset upload is needed.

Use Save Version → Save & Run All. Generation stops between worlds after eight hours; validation
then checks 20 seeded random views per completed world with the unchanged pilot tolerances.
Expected total runtime is roughly 7–8 hours, extrapolated from two sensor-fit worlds, not a
guarantee. The reported remaining quota before preparation was 22 h 36 min.

Download all outputs, including `production-summary.json`, `cache-production.log` and `caches/`.
Status `complete` requires all 320 worlds to have matching metadata, registry entries and live
validation records. Status `partial` saves the completed subset: attach that saved notebook output
to a subsequent production run. The runner checks restored headers and checksums before copying,
refuses conflicting versions, and registers restored caches. It revalidates all completed caches.
Collect and verify the actual output files locally before marking M5 done and updating the
committed artifact registry; a summary alone is insufficient.

## Manual Kaggle pilot

The installed Kaggle CLI is blocked by a Windows code-integrity policy. Smart App Control is
off; changing it will not address the observed block. Browser automation also could not reliably
identify the browser URL. Use Kaggle's website manually; this requires no security-setting changes
and no token in the notebook.

After pushing a tested commit:

```powershell
uv run python scripts/prepare_cache_notebook.py
uv run python scripts/prepare_cache_inputs.py
```

This creates `data/kaggle/altiseek-cache-pilot.ipynb`, pinned to that commit.
The second command creates `data/kaggle/altiseek-m5-inputs.zip` containing byte-preserved
libraries and all 320 world specs, after checking their IDs and hashes. It verifies every
packaged file against the original before reporting success.

1. Create a notebook at <https://www.kaggle.com/code/new> and import the `.ipynb` file.
2. Keep it **private**. Enable **Internet** and select **GPU T4 x2**.
3. **Add Input**: attach the saved output of your `altiseek-detector-v3` notebook. The pilot
   locates `best.pt` under `/kaggle/input` and checks detector ID `700e0ff3455259e8`.
   Also upload `altiseek-m5-inputs.zip` as a **private dataset** and attach it. Its input tree
   must expose `m5-inputs.json`, `libraries/` and `worlds/`. The notebook refuses to proceed
   without this bundle; uploading a ZIP without unpacking it into the dataset is insufficient.
4. **Save Version → Save & Run All** starts a background run and saves its outputs.
5. Download `pilot-summary.json` and `cache-pilot.log` from the saved output for review.

Kaggle documents [accelerators, notebook inputs and saved outputs](https://www.kaggle.com/docs/notebooks).
The notebook contains only code and a commit hash; it does not contain an API credential or imagery.

The manual notebook now uses the original library/world input bundle (D32), verifying its split,
library IDs, every image hash and all world-set IDs before detection. It skips M1 reconstruction
and library/world generation. It also checks the imported NumPy version is the pinned 2.1.3 and
writes `runtime.json`; distribution metadata alone was ambiguous in the first pilot. Direct
`cache_pilot.py` without `--require-input-bundle` retains the rebuild path when no bundle is
attached. Mismatched IDs are always refused.

On matching inputs it caches two sensor-fit worlds (1,760 views), validates 20 seeded views per
world against fresh GPU inference, and reports composition/render/inference time, output bytes,
and estimated cache hours for each set. These two worlds are a preliminary cost estimate, not
performance evaluation. D33 retains D29's counts based on the measured cost. The pilot does not
launch the full build automatically.

## Generation and resume

```bash
python scripts/build_cache.py --sets sensor-fit --limit 2 checkpoint=/kaggle/input/.../best.pt out_dir=/kaggle/working/caches
python scripts/validate_cache.py --sets sensor-fit --limit 2 checkpoint=/kaggle/input/.../best.pt out_dir=/kaggle/working/caches
```

Use `configs/cache/default.yaml` overrides for the actual library/world paths. GPU inference is
required. CPU tests use a fake detector. `--max-seconds` stops generation between worlds; allow
headroom for one complete world and output saving before Kaggle's session limit.

Each world produces `<world_id>.npz` and `.meta.json`, followed by an atomic `completed.json`
update. The set's manifest is reconstructed from verified files on resume. Each file must match
the expected header and its metadata SHA-256. A complete file without metadata is checked for
identity and regenerated; mismatched files are refused. Partial `.tmp` files are not complete.
The cloud registry is `caches/artifacts.json`; it does not modify the checked-in manifest.

For a new cloud session, attach the previous saved notebook output and copy its cache directory
to `/kaggle/working/caches`. Reuse verified libraries/worlds or rebuild and check them again.
Completed worlds are skipped. Copying the previous output is necessary: `/kaggle/working` is
not automatically shared between sessions. An abruptly killed, unsaved session may lose its
output; chunk runs and save versions before the session cap.

Live validation defaults are fixed at score tolerance 0.0001 and box-coordinate tolerance
0.05 px. Counts and classes must match exactly, and matches must be one-to-one irrespective of
output ordering. Stored world projections and target truth are checked too. Threshold/NMS
boundary differences fail visibly; do not silently loosen tolerances based on test worlds.

M5 is complete only after all four sets exist, live validation passes, and the committed registry
contains the verified cache artifacts with durable locations. M6 must wait for that gate.

## First cloud pilot: library identity diagnosis (D32)

The first pilot stopped before inference. Split and detector IDs matched. Library counts matched,
but B rebuilt as `98ad44c968c05f24` and T as `e20c6b1f8e3dfde2`. The owner supplied the saved
`results.zip`; every library image and both index tables were compared with the originals:

| Split | Images compared | Different PNG bytes | Different decoded pixels | Invalid cloud image hashes |
|---|---:|---:|---:|---:|
| B | 2,157 | 2,157 | 0 | 0 |
| T | 650 | 650 | 0 | 0 |

Every CSV field except `sha256` matched. Parsed B conditions matched exactly; T differed only
in its `fitted_library` parent ID. Windows JSON used CRLF and Kaggle JSON used LF. A sample PNG's
IDAT payload sizes differed while its decoded pixels matched; the precise encoder/runtime cause
was not established. Both sides reported the same pinned Pillow/OpenCV versions, but cloud
metadata reported NumPy 1.26.4 and local metadata 2.1.3. The first log did not record the imported
module version, so this is not evidence that the cloud actually imported NumPy 1.26.4.

The accepted originals remain authoritative. Transfer their exact bytes privately rather than
renaming rebuilt artifacts or changing their IDs. The dataset's [official repository](https://github.com/Ben93kie/SeaDronesSee/blob/main/README.md)
states that the data are CC0 1.0; terms were checked before preparing this private transfer.
No imagery or transfer bundle is committed to git.

## Successful pilot with original inputs (D33)

The second pilot passed with all frozen library/world-set IDs and imported NumPy 2.1.3,
PyTorch 2.11.0+cu128. Both worlds have 880 views; all 40 randomly sampled live checks passed
at the fixed tolerances. Downloaded cache files were checked locally against their expected
headers and metadata SHA-256. They are retained under
`data/kaggle/altiseek-cache-pilot-passed/`.

Per-world totals were 74.89 and 65.30 seconds (mean 70.10). Composition averaged 6.56 seconds,
rendering 48.32 seconds, and detection 15.12 seconds; the first world includes detector warm-up.
Cache sizes were 23,213 and 17,382 bytes. These sizes are for two low-distractor pilot worlds
and should not be extrapolated as an upper bound. End-to-end pilot time was 416.39 seconds,
including 239.87 seconds of preparation/input checking.

| Set | Worlds | Projected generation hours |
|---|---:|---:|
| sensor-fit | 40 | 0.78 |
| rl-train | 160 | 3.12 |
| rl-val | 40 | 0.78 |
| test | 80 | 1.56 |
| Total | 320 | 6.23 |

Counts and seeds stay unchanged. Generation estimates exclude full validation and startup.
The final build will provide measured production timings and sizes.

## Completed production and collection (2026-10-27)

Production at commit `c6c1e43` completed all 320 worlds in 24,194.46 seconds (6 h 43 min 14 s).
Summed per-world generation time was 20,591.08 seconds (5.72 h, 64.35 s/world). All 6,400 seeded
live comparisons passed the fixed tolerances. The imported NumPy was 2.1.3 and torch 2.11.0+cu128.
The output contains 281,600 cached views and 9,804,608 compressed cache bytes (9.35 MiB).

The archive is retained at `data/kaggle/altiseek-cache-production/results.zip`; its SHA-256 is
`f7aa9d9802498f7bf1cfe6e5d99724dcc01970b865c52d5c87c9b38e714870a5`.
To reproduce collection:

```powershell
uv run python scripts/collect_cache.py data/kaggle/altiseek-cache-production/results.zip
```

Collection verifies every expected world, complete lattice/variant keys, cache file checksum,
header IDs/settings, local world/checkpoint input hashes, metadata configs and counts, exact
seeded validation sample indices, unchanged tolerances, and the cloud artifact registry.
Only after all checks pass does it copy the exact cache/metadata bytes into `data/caches/` and
register the 320 cache artifacts with local locations in `registry/manifest.json`. Existing
different files or registry entries are refused. Collection does not rerun inference locally.
The report and collection provenance are `data/caches/collection-summary.json` and
`data/caches/collect.meta.json`.

M6 may now use the sensor-fit caches. No test detection-performance statistics were used to
choose counts, thresholds, tolerances or any sensor-model rule during M5.
