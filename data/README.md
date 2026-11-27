# UAVibes Dataset

The full dataset is hosted on Zenodo. The eight ZIP files total ~2.5 GB compressed (~6.3 GB uncompressed).

**Download:** https://zenodo.org/records/24504582  
**DOI:** 10.5281/zenodo.24504582

---

## Which ZIP do I need?

You do not have to download everything. The table below maps each use case to the minimum set of ZIPs required.

| Goal | ZIPs needed |
|------|-------------|
| Reproduce all paper results (`python reproduce.py`) | All eight |
| Dashboard Tab 2 — enroll and test key authentication | `fixed_rig_boot.zip` |
| Dashboard Tab 3 — simulate on-device key exchange | `fixed_rig_boot.zip` |
| Dashboard Tab 4/5/6/7 — offline analysis and spectrograms | `fixed_rig.zip` |
| Board-fingerprint study (`python -m analysis.board`) | `fixed_rig_board.zip` |
| Motor-swap study (`python -m analysis.swap`) | `fixed_rig_swap.zip` |
| Tamper-detection evaluation (`python -m analysis.conditions`) | `fixed_rig_boot.zip` + `fixed_rig_props.zip` + `fixed_rig_payload.zip` |
| Validate the motor model | `calibration.zip` |

---

## Extracting the archives

Each ZIP expands to a single directory. Extract it directly into this `data/` folder so that the resulting path is `data/<directory-name>/`.

**Linux / macOS:**
```bash
cd data/
unzip ../fixed_rig.zip
unzip ../fixed_rig_boot.zip
# repeat for each ZIP you downloaded
```

**Windows (PowerShell):**
```powershell
cd data
Expand-Archive ..\fixed_rig.zip -DestinationPath .
Expand-Archive ..\fixed_rig_boot.zip -DestinationPath .
# repeat for each ZIP you downloaded
```

After extraction the layout should be:

```
data/
├── fixed_rig/            ← from fixed_rig.zip
├── fixed_rig_test/       ← from fixed_rig_test.zip
├── fixed_rig_boot/       ← from fixed_rig_boot.zip
├── fixed_rig_board/      ← from fixed_rig_board.zip
├── fixed_rig_swap/       ← from fixed_rig_swap.zip
├── fixed_rig_props/      ← from fixed_rig_props.zip
├── fixed_rig_payload/    ← from fixed_rig_payload.zip
└── calibration/          ← from calibration.zip
```

---

## Directory structure

### `fixed_rig/` — enrollment sessions

```
fixed_rig/
└── drone{A|B|C|D}/
    ├── raw/              # SD-card binary logs (logNN.raw)
    ├── timelines/        # JSON phase timelines (one per stage)
    ├── converted/        # 1 kHz IMU CSVs (*_1khz.csv) ← analysis uses these
    └── session_NN.json   # session manifest (attempt tracking)
```

Each drone has 5 enrollment sessions × 15 stages = 75 CSV files per drone.

### `fixed_rig_test/` — held-out test sessions

Same layout as `fixed_rig/`. Two sessions per drone, recorded on different days. Never used to build an identity; used only to verify one.

### `fixed_rig_boot/` — boot-protocol sessions

```
fixed_rig_boot/
└── sNN_drone{X}/
    ├── raw/
    ├── timelines/
    ├── converted/        # single CSV per boot: rest_hold + 4 motor holds
    └── session.json      # board params (self-test, gyro bias, accScale)
```

Each boot session is one power cycle: 10 s rest then each motor alone at 58 000 PWM for 5 s. These are the sessions used for key enrollment and authentication.

### `fixed_rig_board/` — board-fingerprint sessions

One session per power cycle, motors off throughout.  
`session.json` contains the IMU self-test readings, gyro bias, accelerometer scale factor and temperature read directly from the firmware at boot.

### `fixed_rig_swap/` — motor-swap experiment

```
fixed_rig_swap/
└── sNN_{pre|swap|post}_frame{A|B}_motors{A|B}/
    ├── raw/ timelines/ converted/
    ├── session.json
    └── analysis/verdict.{json,md}   # per-session identity verdict
```

Drones A and B exchange all four motors. `summary.md` gives the cross-session conclusions.

### `fixed_rig_props/`, `fixed_rig_payload/` — altered-drone sessions

Same layout as `fixed_rig_boot/`. `session.json` describes the modification (which propeller model, payload mass and position).

### `calibration/` — laser-tachometer data

CSV files from PWM→RPM calibration sessions used to fit `pipeline/motor_model.py`. Not needed for any analysis script; included for reproducibility of the motor model itself.

---

## File naming convention

Files under `fixed_rig/` and `fixed_rig_test/` follow the pattern:

```
{YYYYMMDD}_{HHMMSS}_drone{A|B|C|D}_{s|t}{NN}_{purpose}[_1khz].csv
```

`purpose` is one of:  
`sweep-m1..m4`, `sweep-diag_13`, `sweep-diag_24`, `sweep-front_12`, `sweep-rear_34`, `sweep-left_14`, `sweep-right_23`, `sweep-all4`, `stepped-m1..m4`.

Sessions tagged `_sNN_` are enrollment; `_tNN_` are held-out test.

---

## Loading a recording in Python

```python
from pipeline.preprocessing import load_session

# load one stage, keeping only the hold phase
df, fs = load_session(
    "data/fixed_rig/droneA/converted/20261001_120000_droneA_s01_sweep-m1_1khz.csv",
    phase_filter="m1_hold_58000"
)
# df has columns: timestamp_s, acc.x, acc.y, acc.z, gyro.x, gyro.y, gyro.z, motor.m1-m4
# fs ≈ 1000  (Hz)
```

```python
from pipeline.features import extract_features

feat_df = extract_features(df, fs, window_s=1.0, hop_s=0.5)
# 120-D feature vector per 1 s window
```

---

## Notes

- **Only the `converted/` CSVs are needed** to reproduce all results. The `raw/` binary logs (~1 GB total) are included for completeness; they can be omitted if disk space is tight.
- `analysis_cache/`, `keystore/`, `models/`, and `old/` from the development tree are not included in this release.
- `reproduce.py` does not require the session manifests for most steps; it uses them only for session-level evaluation (`boot_eval`, `conditions`).
