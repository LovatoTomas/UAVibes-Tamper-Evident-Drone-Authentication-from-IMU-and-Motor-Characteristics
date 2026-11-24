# enrollment/

Data-collection infrastructure for the bolted-rig recording protocol. The standard entry point is the dashboard's **New Rig Session** tab (`run_dashboard.py`), which drives this module automatically. The modules here can also be used directly for scripted or unattended recording (`auto_bench.py`).

## Protocol overview

Each session consists of **15 independent stages**, recorded as separate SD-card logs:

| Stage group | Stages | PWM | Duration |
|-------------|--------|-----|---------|
| Single-motor holds | `sweep-m1`, `sweep-m2`, `sweep-m3`, `sweep-m4` | 58 000 | ~15 s each |
| Motor-pair holds | `sweep-diag_13`, `sweep-diag_24`, `sweep-front_12`, `sweep-rear_34`, `sweep-left_14`, `sweep-right_23` | 58 000 | ~15 s each |
| All-four hold | `sweep-all4` | 58 000 | ~15 s |
| Stepped ladders | `stepped-m1`, `stepped-m2`, `stepped-m3`, `stepped-m4` | 10–58 000 (7 steps) | ~15 s/step × 7 |

**Boot-protocol sessions** (session type Boot) use a different fixed sequence: 10 s rest (all motors off), then `m1_hold_58000` / `m2_hold_58000` / `m3_hold_58000` / `m4_hold_58000` (5 s each). These are the sessions used for key enrollment and authentication.

## Modules

### `micro_sweep.py`

Defines the fixed protocol constants and runs one stage blocking on the drone.

```python
from enrollment.micro_sweep import STANDARD_PROTOCOL, BOOT_PURPOSES, run_stage

# STANDARD_PROTOCOL: list of Stage objects (name, pwm, timings)
for stage in STANDARD_PROTOCOL:
    print(stage.name, stage.pwm, stage.hold_s)

# Run a single stage (drone must be connected)
run_stage(cf, timeline_recorder, stage_name="sweep-m1")
```

### `session_manifest.py`

Tracks every recording attempt for a drone/session. The manifest JSON is the authoritative record — the dashboard reads and writes it; the analysis scripts do not need it.

```python
from enrollment.session_manifest import SessionManifest

# Load or create a session manifest
manifest = SessionManifest.load_or_new(
    drone_id="droneA", session_number=1,
    base_dir="data/fixed_rig", kind="enrollment"
)

# After uploading SD card files, match them to stages
manifest.ingest_upload_batch(uploaded_file_paths)

# Check what still needs recording
for stage in manifest.pending_stages():
    print(stage.name, stage.status)
```

### `recorder.py`

Records live CRTP telemetry (100 Hz) to a CSV file.

```python
from enrollment.recorder import DataRecorder

recorder = DataRecorder("output.csv")
recorder.start(cf)          # cf: SyncCrazyflie
# ... run motors ...
recorder.stop()
```

### `usd_timeline.py`

Records the phase timeline (start/stop events, motor commands, timestamps) alongside an SD-card recording. The timeline is what `usd_convert.py` uses to attribute phases to converted CSV rows.

```python
from enrollment.usd_timeline import UsdPhaseRecorder

tl = UsdPhaseRecorder("timeline.json")
tl.event("usd_start", log_index=4)
tl.event("phase_start", phase="m1_hold_58000", pwm=58000)
# ...
tl.save()
```

## Converting SD card logs to CSV

After copying logs off the SD card:

```bash
python usd_convert.py <timeline.json> <logNN.raw> -o converted/output_1khz.csv
```

Or use the dashboard's upload button, which calls this automatically.
