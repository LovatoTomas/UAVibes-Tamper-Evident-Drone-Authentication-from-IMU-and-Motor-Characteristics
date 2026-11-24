# UAVibes Dataset

The full dataset is hosted separately due to its size (~7 GB).

**Download:** https://zenodo.org/records/24504582  
**DOI:** 10.5281/zenodo.24504582

---

## Dataset layout

After extracting the archive, place the contents of `data/` from the archive under this directory, so that the layout matches:

```
data/
├── fixed_rig/            # enrollment sessions (drones A, B, C, D)
│   ├── droneA/
│   │   ├── raw/          # SD-card binary logs (logNN.raw)
│   │   ├── timelines/    # JSON phase timelines (one per stage)
│   │   ├── converted/    # 1 kHz IMU CSVs (one per stage, _1khz.csv)
│   │   └── session_NN.json   # session manifest (attempt tracking)
│   ├── droneB/
│   ├── droneC/
│   └── droneD/
│
├── fixed_rig_test/       # held-out test sessions (same layout as fixed_rig/)
│   └── droneA/ … droneD/
│
├── fixed_rig_boot/       # boot-protocol sessions (rest + four single-motor holds)
│   └── sNN_droneX/       # one session per power cycle
│
├── fixed_rig_board/      # board-fingerprint sessions (IMU at rest, one per boot)
│   └── sNN_droneX/
│
├── fixed_rig_swap/       # motor-swap experiment (A↔B, pre/swap/post)
│   ├── sNN_{stage}_frame{X}_motors{Y}/
│   └── summary.md
│
├── fixed_rig_props/      # altered-drone sessions: propellers changed
│   └── sNN_droneX/
│
├── fixed_rig_payload/    # altered-drone sessions: payload added
│   └── sNN_droneX/
│
└── calibration/          # laser-tachometer PWM→RPM measurements (validation only)
```

The `analysis_cache/`, `keystore/`, `models/`, and `old/` directories from the development tree are **not** included in this release.

---

## File naming convention

Files under `fixed_rig/` and `fixed_rig_test/` follow the pattern:

```
{YYYYMMDD}_{HHMMSS}_drone{A|B|C|D}_{s|t}{NN}_{purpose}[_1khz].csv
```

where `purpose` is one of:
`sweep-m1`, `sweep-m2`, `sweep-m3`, `sweep-m4`,  
`sweep-diag_13`, `sweep-diag_24`, `sweep-front_12`, `sweep-rear_34`, `sweep-left_14`, `sweep-right_23`, `sweep-all4`,  
`stepped-m1`, `stepped-m2`, `stepped-m3`, `stepped-m4`.

Sessions tagged `_sNN_` are enrollment sessions; `_tNN_` are held-out test sessions.

---

## Session manifests

Each `session_NN.json` file is the authoritative record of a recording session. It lists every stage attempt with its status (`completed`, `interrupted`, `discarded`, `lost`), the matched raw-file name, and metadata (battery voltage, board parameters). The analysis scripts load converted CSVs directly by name and do not require the manifests; `reproduce.py` uses them only for the session-level evaluation steps.

---

## Board-session data

`fixed_rig_board/sNN_droneX/session.json` contains the IMU self-test results, gyro bias, accelerometer scale factor, and barometer temperature read from the drone's `uavibe` firmware parameter group at every boot. These are the board-fingerprint quantities; the matching `converted/` CSV holds the 30 s IMU rest recording.

---

## Reproducing results without the raw files

The `raw/` subdirectories hold the SD-card binary logs before conversion. All analysis scripts operate on the `converted/` CSVs. If disk space is a constraint, the `raw/` directories (~1 GB total) can be omitted; `reproduce.py` will work without them. Conversion from raw to CSV uses `usd_convert.py`.
