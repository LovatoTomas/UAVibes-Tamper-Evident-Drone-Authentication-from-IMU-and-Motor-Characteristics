# analysis/

One Python module per study. `reproduce.py` (in the repo root) runs all of them in order and regenerates every figure and table in the paper. Each module writes its results to a `results_*/` subdirectory and its figures to `analysis/figures/` (both created automatically).

## Study scripts

| Script | Paper section | What it produces |
|--------|--------------|-----------------|
| `experiments.py` | §5.1 | A/B drone separation: LDA, Mahalanobis, kNN on all feature sets; AUC curves |
| `four_drones.py` | §5.1 | Same study on drones A, B, C, D; cross-day stability |
| `swap.py` | §5.2 | Motor-swap experiment analysis: does the identity follow the motors or the frame? |
| `motor_features.py` | §5.2 | Per-motor physical features (speed gain, broadband floor) extraction and visualisation |
| `motor_identity.py` | §5.2 | Session-level authentication on motor features; fusion with spectral features |
| `board.py` | §5.3 | IMU self-test board fingerprint: temperature compensation, day-to-day stability |
| `combined_key.py` | §5.3 | Motor + board combined key: symbol-level fuzzy extractor simulation |
| `keygen.py` | §5.3 | Drift analysis, helper refresh policies, transplant attack simulation |
| `boot_protocol.py` | §5.4 | Boot-protocol design: feature quality vs. hold duration, protocol selection |
| `boot_eval.py` | §5.4 | Evaluation on real boot sessions: FAR, FRR, impostor and cloning tests |
| `conditions.py` | §5.5 | Altered-drone boots (changed propellers, payload): tamper detection |
| `onboard_eval.py` | §5.5 | On-device vs. PC feature comparison; FRR on the drone hardware |
| `onboard_log.py` | §5.5 | Real FW2 authentication log analysis: per-drone FRR, speed-shift distribution |
| `longitudinal.py` | §5.4 | Five-day longitudinal series (31 Oct – 4 Nov): on-board and replay evaluation |
| `sketch_attack.py` | §6 | Sketch leakage analysis: offline guessing bound from helper data |

## Figure scripts

| Script | Figures |
|--------|---------|
| `figures.py` | Figs 1–9: motor signature, PCA embedding, spectral shape, self-test, temperature |
| `figures4.py` | Figs 2–6 extended to four drones |
| `figures_key.py` | Figs 10–14: key configurations, boot protocol, architecture, real boots, stability |
| `figures_evaluation.py` | Figs 15–16, 20–21: speed symbols, longitudinal, memory layout, tampering |
| `figures_motor_model.py` | Figs 17–19: motor model fit, spectrogram ladder, manual calibration |

## Helper modules

| Module | Purpose |
|--------|---------|
| `dataset.py` | Window-feature cache (loads converted CSVs, caches per-window feature vectors as `.pkl`) |
| `palette.py` | Color palette for all figures: `palA`–`palD` per drone, colorblind-safe |
| `rotation.py` | IMU rotation-direction check (attempted, failed on bolted rig — kept for reference) |
| `motor_signature.py` | Low-level per-motor spectral feature extraction shared by several studies |
| `selftest_all.py` | Batch re-extraction of board self-test results across all board sessions |
| `sketch_count.py` | Helper for counting sketch leakage bits |

## Running individual studies

```bash
# From the repo root:
python -m analysis.board          # board fingerprint study only
python -m analysis.boot_eval      # boot evaluation only
python reproduce.py board         # same, via reproduce.py
python reproduce.py --list        # see all step names
```

Results land in `analysis/results_<study>/`; figures in `analysis/figures/`.
