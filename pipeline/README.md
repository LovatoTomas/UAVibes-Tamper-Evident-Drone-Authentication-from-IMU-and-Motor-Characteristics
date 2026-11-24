# pipeline/

Signal-processing pipeline from raw IMU CSV to authentication-ready feature vectors. All modules are drone-agnostic: they only require the standard IMU CSV format (see root README). The motor model is the single source of truth for PWM→RPM conversion and must be imported from `motor_model.py` — never hard-code the constants elsewhere.

## Quick start

```python
from pipeline.preprocessing import preprocess
from pipeline.features import extract_features
from pipeline.motor_model import pwm_to_hz

# Load and filter a recording
df, fs = preprocess("data/fixed_rig/droneA/converted/20261001_120000_droneA_s01_sweep-m1_1khz.csv")
# df has columns: acc.x/y/z, gyro.x/y/z (raw) and noise.acc.x/y/z, noise.gyro.x/y/z (filtered)

# Extract features (1 s windows, 0.5 s hop)
feat_df = extract_features(df, fs, window_s=1.0, hop_s=0.5)
# feat_df has 120 columns: 6 channels × 19 features + 6 cross-axis coherence pairs

# Motor model
hz = pwm_to_hz(pwm=58000, vbat=4.1, motor_id=1)   # shaft frequency for motor M1
```

## Modules

### `preprocessing.py`

Two filtering stages applied in order:

1. **Macro-deconvolution** (`highpass`): Butterworth highpass at 5 Hz removes gravity and low-frequency flight dynamics. Output columns: `noise.acc.*`, `noise.gyro.*`.

2. **Harmonic deconvolution** (`harmonic_deconvolution`, optional): adaptive comb filter that notches the motor shaft frequency and its harmonics, keyed on the per-motor PWM→RPM conversion. Each motor is notched at its own frequency (they differ by several percent). Q factor: 22.

```python
from pipeline.preprocessing import preprocess, load_session

df, fs = preprocess("recording.csv")
# or, to load a specific phase only:
df, fs = load_session("recording.csv", phase_filter="m1_hold_58000")
```

### `features.py`

Extracts a 120-dimensional feature vector per time window from the `noise.*` columns:

- **Per channel** (6 channels: acc.x/y/z, gyro.x/y/z): variance, skewness, kurtosis, spectral centroid, spectral rolloff, spectral flatness, 13 LFCC coefficients → 19 features × 6 = 114
- **Cross-axis coherence**: 6 pairs → 6 features

```python
from pipeline.features import extract_features, feature_matrix

feat_df = extract_features(df, fs, window_s=1.0, hop_s=0.5)
X = feature_matrix(feat_df)   # numpy array, shape (n_windows, 120)
```

### `motor_model.py`

Canonical PWM→RPM map for Crazyflie 2.1. **Single source of truth** — import from here, never hard-code.

```
RPM = (0.0934 × PWM × vbat + 4321) × MOTOR_GAIN[motor_id]
f₀  = RPM / 60          (shaft frequency in Hz)
f_blade = 2 × f₀        (blade-pass frequency, 2-blade props)
```

```python
from pipeline.motor_model import pwm_to_rpm, pwm_to_hz, blade_pass_hz, MOTOR_GAIN

rpm = pwm_to_rpm(58000, vbat=4.1)                    # base model, no per-motor gain
rpm = pwm_to_rpm(58000, vbat=4.1, motor_id=1)        # with per-motor gain
hz  = pwm_to_hz(58000, vbat=4.1, motor_id=1)         # shaft frequency
bp  = blade_pass_hz(58000, vbat=4.1, motor_id=1)     # blade-pass frequency

print(MOTOR_GAIN)   # {1: 0.973, 2: 0.990, 3: 1.041, 4: 1.049}
```

The model is calibrated on 67 laser-tachometer points (PWM 8 000–55 000, R² = 0.995). Per-motor gains are fit on 47 points in the flight-relevant regime (35 000–55 000 PWM). Valid for Crazyflie 2.1 with standard 40 mm propellers; recalibrate with `rpm_calibration.py` for other hardware.

### `manifold.py`

One-class authenticator: StandardScaler → UMAP (2D visualisation) → Mahalanobis distance vs. enrollment baseline.

```python
from pipeline.manifold import ManifoldModel

model = ManifoldModel()
model.fit_enrollment(X_enroll)       # X: (n_windows, 120) numpy array
model.save("models/droneA.pkl")

model = ManifoldModel.load("models/droneA.pkl")
distances, is_auth = model.authenticate(X_test)
```

**Note:** the ManifoldModel performs poorly when enrollment data is multimodal (mixed excitation stages). Always split by stage before fitting and authenticating — see `analysis/experiments.py` for the correct usage.

### `calibration.py`

Spectral cross-validation tool: given a single-motor recording, finds the PWM→RPM gain that best aligns the dominant spectral line with the model prediction.

```python
from pipeline.calibration import estimate_motor_gain
gain = estimate_motor_gain("data/fixed_rig/droneA/converted/..._sweep-m1_1khz.csv",
                           motor_id=1, pwm=58000, vbat=4.1)
```
