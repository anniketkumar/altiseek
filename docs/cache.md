# Detection cache (M5)

The code is implemented; the cloud pilot, cost decision and production cache build are pending.
No GPU inference has run for M5 yet. D31 records storage and inference choices.

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
performance evaluation. D29 counts remain provisional until the measured cost is reviewed and
the count decision is logged. The full build is not launched automatically.

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
