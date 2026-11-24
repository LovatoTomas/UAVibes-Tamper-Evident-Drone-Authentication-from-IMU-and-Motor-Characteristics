"""
Regenerate every result and figure of the thesis from the recorded data.

    python reproduce.py                 # all steps, reusing caches
    python reproduce.py --fresh         # rebuild the derived caches too (slow)
    python reproduce.py --list          # show the steps
    python reproduce.py board key       # only the named steps (and nothing else)

Each step runs as its own process (`python -m <module>`), in the order of
the reports in paper/INDICE.md; a failing step is reported and the others
still run. Outputs: auth_client/analysis/results*/ and analysis/figures/.
Input: the recordings under data/fixed_rig*/ (not versioned).
Duration of a full run (2026-10-09, 22-core PC): ~4 h, mostly the four-drone
study (~2.6 h) and the boot-protocol candidates (~40 min).
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# name -> (report, module, description)
STEPS = [
    ("motor-model", "-", "analysis.figures_motor_model", "motor model: laser fit, per-motor gains, ladder spectrograms (thesis ch. 4)"),
    ("dataset",   "-",  "analysis.dataset",         "window features of every recording (cache)"),
    ("ab",        "1",  "analysis.experiments",     "A vs B identification study (E1-E9)"),
    ("ab-motor",  "1",  "analysis.motor_signature", "A vs B motor signature (E7)"),
    ("fig-ab",    "1",  "analysis.figures",         "figures 1-3"),
    ("4drones",   "2",  "analysis.four_drones",     "four-drone study (X1, E1-E11)"),
    ("swap",      "3",  "analysis.swap",            "motor-swap verdicts"),
    ("fig-4d",    "2-3", "analysis.figures4",       "figures 4-7"),
    ("motor",     "4",  "analysis.motor_features",  "physical per-motor features"),
    ("motor-id",  "4",  "analysis.motor_identity",  "motor identity, fusion, key feasibility"),
    ("keygen",    "5",  "analysis.keygen",          "drift, entropy, transplant"),
    ("board",     "7",  "analysis.board",           "board fingerprint over two days"),
    ("key",       "8",  "analysis.combined_key",    "combined motors + board key"),
    ("boot",      "10", "analysis.boot_protocol",   "boot-protocol candidates"),
    ("demo",      "9",  "keyproto.demo",            "end-to-end key prototype"),
    ("boot-eval", "11", "analysis.boot_eval",       "real boot sessions: key, stability, tilt"),
    ("conditions", "12", "analysis.conditions",     "altered boots: propellers changed, payload added"),
    ("onboard",   "FW2", "analysis.onboard_eval",   "on-board feature computation vs offline, real boots"),
    ("onboard-log", "FW2", "analysis.onboard_log",  "on-board key on the drones: station log, FRR, exact speed symbols"),
    ("selftest-all", "13", "analysis.selftest_all", "accelerometer self-test alone over every recorded boot"),
    ("longitudinal", "13", "analysis.longitudinal", "five-day series: replay of the SD boots (genuine, impostors, transplant), speed window"),
    ("fig-key",   "7-11", "analysis.figures_key",   "figures 8-14"),
]

# derived caches rebuilt by --fresh (raw/converted recordings are never touched)
CACHES = [
    "data/analysis_cache",
    "analysis/results_4drones/motor_signatures.csv",
    "analysis/results_motor/motor_features.csv",
    "analysis/results_boot/boot_features.csv",
    "data/keystore/system.json",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="*", help="run only these steps (default: all)")
    ap.add_argument("--fresh", action="store_true", help="delete derived caches first")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    a = ap.parse_args()

    if a.list:
        for name, rep, mod, desc in STEPS:
            print(f"{name:10s} report {rep:5s} {mod:28s} {desc}")
        return
    unknown = set(a.steps) - {s[0] for s in STEPS}
    if unknown:
        sys.exit(f"unknown step(s): {', '.join(sorted(unknown))} (see --list)")

    if a.fresh:
        for c in CACHES:
            p = ROOT / c
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
            print(f"removed {c}")

    todo = [s for s in STEPS if not a.steps or s[0] in a.steps]
    results = []
    t_all = time.time()
    for name, rep, mod, desc in todo:
        print(f"\n=== [{name}] {desc}  (python -m {mod})", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", mod], cwd=ROOT)
        dt = time.time() - t0
        results.append((name, r.returncode, dt))
        print(f"=== [{name}] {'OK' if r.returncode == 0 else f'FAILED ({r.returncode})'} in {dt / 60:.1f} min",
              flush=True)

    print("\nSummary")
    for name, rc, dt in results:
        print(f"  {name:10s} {'OK    ' if rc == 0 else 'FAILED'} {dt / 60:6.1f} min")
    print(f"  total {(time.time() - t_all) / 60:.1f} min")
    if any(rc for _, rc, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
