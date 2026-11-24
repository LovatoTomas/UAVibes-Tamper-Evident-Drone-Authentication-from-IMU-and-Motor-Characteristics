# UAVibes — Drone Physical Identity from IMU and Motor Characteristics

UAVibes is a proof-of-concept system that derives a cryptographic key from the physical characteristics of a drone's motors and inertial sensor (IMU), without ever storing the key on the device. At every power-up the drone measures its own vibration signature and IMU self-test response, reconstructs the key with a fuzzy extractor, and authenticates to a ground station with a password-authenticated key exchange (CPace). With the drone switched off, its memory holds only public helper data.

This repository accompanies the paper:

> *UAVibes: Tamper-Evident Drone Authentication from IMU and Motor Characteristics*  
> Anonymous Submission — USENIX Security 2027

---

## Repository layout

```
UAVibe-public/
├── README.md
├── requirements.txt          # Python dependencies
├── pytest.ini
├── reproduce.py              # regenerates every result and figure in the paper
├── run_dashboard.py          # launches the Streamlit dashboard
├── auto_bench.py             # unattended bench-test runner (requires a real drone)
├── main.py                   # legacy CLI enrollment runner
├── usd_convert.py            # converts raw SD-card binary logs to CSV
├── usd_motor_analysis.py     # motor speed analysis on SD recordings
├── rpm_calibration.py        # laser-tachometer calibration session
├── model_validation.py       # validates motor-model predictions
├── tap_test.py / tap_test_analysis.py  # tap-test tool (optional)
│
├── crazyflie/                # Crazyflie drone connection and telemetry
├── enrollment/               # data-collection protocol (micro-sweep, session manifest)
├── pipeline/                 # signal processing: preprocessing, features, motor model
├── analysis/                 # all study scripts and figure generators
├── keyproto/                 # fuzzy extractor, CPace key exchange, boot protocol
├── tests/                    # automated test suite
├── ui/                       # Streamlit dashboard (8 tabs)
│
├── firmware/                 # firmware patches and on-device C core
│   ├── FIRMWARE.md           # full description of every firmware variant and patch
│   ├── patches/              # FW1 and FW2 patches against crazyflie-firmware
│   └── uavibe_key/           # out-of-tree Crazyflie app (FW2): portable C core + build
│
└── data/
    └── README.md             # dataset structure and download link
```

---

## Installation

Python 3.10 or later is required.

```bash
pip install -r requirements.txt
```

The dashboard and analysis scripts do not require a physical drone; a real drone is needed only for `auto_bench.py` and for new data collection.

On **Windows**, `cflib` requires the **WinUSB** driver for the Crazyflie USB device and the CrazyRadio PA dongle (install with [Zadig](https://zadig.akeo.ie/)). The connection modules monkey-patch `cflib`'s device-finding routine to use `libusb-1.0`; this patch is applied automatically at import time.

---

## Reproducing all paper results

Download the dataset from **https://zenodo.org/records/24504582** and extract each ZIP under `data/` so the layout matches `data/README.md`. Then run:

```bash
python reproduce.py           # all steps, reusing cached intermediate results
python reproduce.py --fresh   # also rebuilds derived caches
python reproduce.py --list    # print all available steps
python reproduce.py board key fig-key   # run specific steps by name
```

A full run takes roughly 4 hours on a modern laptop. Every figure and table in the paper is regenerated; output goes to `analysis/figures/` and `analysis/results_*/`.

---

## Streamlit dashboard

```bash
python run_dashboard.py       # opens http://localhost:8501
```

The dashboard is the primary interface for data collection, enrollment and authentication testing. All tabs work without a physical drone when the dataset is present under `data/`; a real drone is required only for live recording.

### Tab 1 — New Rig Session (data collection with a real drone)

This tab implements the standard bolted-rig recording protocol. Each session consists of 15 fixed stages recorded independently as separate SD-card logs, so a dropped connection costs only the stage in progress.

**Workflow:**
1. Connect the drone over USB or CrazyRadio (URI shown at the top of the page).
2. Select drone identifier (A / B / C / D) and session type (Enrollment / Test / Boot / Board).
3. Click **Avvia prossima** to run the next pending stage, or **🤖 Registrazione automatica** to run all remaining stages unattended (90 s battery-recovery pause between stages).
4. When all 15 stages are done, copy the SD card's log files to the PC.
5. Click **Carica file SD** and select all log files from the card; the dashboard matches them to the correct stages automatically.
6. Repeat from step 2 for every new session.

The 15 stages are: four single-motor holds (`sweep-m1`–`sweep-m4`), six motor-pair holds, one all-four hold (`sweep-all4`), and four stepped-PWM ladders (`stepped-m1`–`stepped-m4`). PWM is fixed at 58 000; timings are not configurable to keep sessions comparable.

For **boot sessions** (session type Boot) the protocol is instead: 10 s rest (all motors off) followed by each motor alone at 58 000 PWM for 5 s. These are the sessions used for key enrollment and authentication.

### Tab 2 — Verifica identità (PC-side key enrollment and authentication)

This tab enrolls a drone from its recorded boot sessions and tests authentication — no drone needed after recording.

**Enroll a drone:**
1. Open the tab and select the drone identifier.
2. Click **Enroll from boot sessions**: the tab reads the first three boot sessions of that drone from `data/fixed_rig_boot/` and computes the helper data.
3. The enrollment is saved to `data/keystore/` (helper file + station entry).

**Test authentication:**
1. Select the drone and a boot session to replay.
2. Click **Avvia verifica**: the tab re-extracts features from the session, runs the fuzzy extractor Rep, and runs the PC-side CPace exchange against the local station.
3. The result shows accepted / rejected, the number of wrong symbols, and whether the speed alarm fired.

### Tab 3 — Chiave a bordo (on-device key exchange)

This tab drives the FW2 on-device protocol. It works in two modes:

**Simulation (no drone required):**
1. Select **Drone simulato**.
2. Pick the drone identifier and a set of recorded boot sessions to feed the simulated drone.
3. Click **Avvia**: the simulated drone replays the recorded measurements and runs the full CPace exchange with the local station in memory.

**Real drone (FW2 required):**
1. Connect the drone running FW2.
2. Click **Enroll** (first time) or **Autentica** (subsequent boots): the drone runs its boot protocol autonomously and exchanges keys with the station over the radio.
3. The tab shows the result, speed shifts per motor, and any alarm.

### Tabs 4–7 — Analysis and visualization (dataset required)

These tabs operate on the downloaded dataset under `data/`.

| Tab | Purpose |
|-----|---------|
| **Offline Analysis** | Load a session, run the pipeline, visualize features and authentication result for any stage |
| **Spectrogram** | Time-frequency view of a recording; overlay motor-model frequency predictions |
| **Motor Calibration** | Estimate per-motor speed gains from single-motor stages; compare to the canonical model |
| **Resonances** | Spectral shape comparison across drones and sessions |

---

## Automated tests

```bash
cd <repo root>
python -m pytest
```

The test suite (~5 s) covers the fuzzy extractor, the key exchange, the session-manifest logic, the motor model, the boot protocol, and byte-exact agreement of the firmware C core with the Python reference implementation (requires WSL with `make` and `arm-none-eabi-gcc`).

---

## Firmware

`firmware/FIRMWARE.md` documents every variant and modification. Two variants are used:

- **FW1** — stock Crazyflie firmware with `CONFIG_DECK_FORCE=bcUSD` (enables the custom SD deck) and a patch that exposes IMU self-test results at boot.
- **FW2** — FW1 plus the out-of-tree key application (`firmware/uavibe_key/`), which runs the full boot protocol and CPace on the drone itself.

Build in WSL:
```bash
# FW1
cd crazyflie-firmware   # sibling repo
make

# FW2
cd firmware/uavibe_key
make
make cload              # flash (drone in bootloader mode)
```

---

## Modules

Each subdirectory has its own README with per-module documentation and code examples.

| Directory | README | Purpose |
|-----------|--------|---------|
| `pipeline/` | [pipeline/README.md](pipeline/README.md) | Signal processing: highpass filter, comb filter, 120-D feature extraction, motor model, ManifoldModel |
| `enrollment/` | [enrollment/README.md](enrollment/README.md) | Data-collection protocol: 15-stage micro-sweep, session manifest, SD-card timeline |
| `keyproto/` | [keyproto/README.md](keyproto/README.md) | Fuzzy extractor, CPace key exchange, boot protocol, SPAKE2 |
| `analysis/` | [analysis/README.md](analysis/README.md) | One script per study; figure generators; helper modules |
| `firmware/` | [firmware/README.md](firmware/README.md) | How to apply patches, build FW1/FW2, generate station key |

---

## Using UAVibes on a different drone

The pipeline and crypto modules are hardware-agnostic. Only the motor model constants and the data-collection scripts are Crazyflie-specific. To port to another platform:

1. **Produce IMU recordings** in the standard CSV format (`timestamp_s, phase, acc.x, acc.y, acc.z, gyro.x, gyro.y, gyro.z, motor.m1, …`) at ≥ 1 kHz. The `phase` column must label the excitation stages.

2. **Recalibrate the motor model**: fit `RPM = a × (PWM × vbat) + b` on your hardware with a tachometer or from the dominant spectral line of single-motor recordings. Update `MOTOR_GAIN` and the base coefficients in `pipeline/motor_model.py`.

3. **Re-run enrollment**: collect at least 3 boot sessions per drone with the same rest + single-motor-hold protocol and enroll via `python -m keyproto.boot` or the dashboard.

4. **Adapt the firmware** (optional, for on-device key generation): the portable C core in `firmware/uavibe_key/src/core/` has no dependency on the Crazyflie SDK; only `uv_app.c` contains Crazyflie-specific glue code.

---

## Data format

IMU recordings are CSV files with columns:

```
timestamp_s, phase, acc.x, acc.y, acc.z, gyro.x, gyro.y, gyro.z, motor.m1, motor.m2, motor.m3, motor.m4
```

Sampling rate: ~1 kHz (SD-deck path) or ~100 Hz (CRTP telemetry). The `phase` column labels protocol segments (e.g. `rest_hold`, `m1_hold_58000`). Load a recording with:

```python
from pipeline.preprocessing import load_session
df, fs = load_session("data/fixed_rig/droneA/converted/20261001_120000_droneA_s01_sweep-m1_1khz.csv",
                      phase_filter="m1_hold_58000")
```

---

## Citation

```bibtex
@inproceedings{uavibes2027,
  title     = {{UAVibes}: Tamper-Evident Drone Authentication from {IMU} and Motor Characteristics},
  author    = {Anonymous},
  booktitle = {Proceedings of the 36th USENIX Security Symposium},
  year      = {2027}
}
```

---

## License

The code is released under the MIT License. See `LICENSE`.
