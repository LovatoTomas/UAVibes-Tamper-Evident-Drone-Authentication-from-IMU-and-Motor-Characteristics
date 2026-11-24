"""
UAVibes — Real-time Dashboard
=============================
Single-page Streamlit app for drone identity enrollment and live authentication.

Launch via:
  python run_dashboard.py          (from auth_client/)
  streamlit run ui/app.py          (from auth_client/)
"""

# ── path setup (must precede local imports) ───────────────────────────────────
import sys
from pathlib import Path
_AUTH_CLIENT = Path(__file__).resolve().parent.parent
if str(_AUTH_CLIENT) not in sys.path:
    sys.path.insert(0, str(_AUTH_CLIENT))

# ── stdlib ────────────────────────────────────────────────────────────────────
import json
import re
import time
from datetime import datetime

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st

# ── page config MUST be the first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="UAVibes",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── local imports ─────────────────────────────────────────────────────────────
from ui.state import (
    DroneWorker, init_state,
    preprocess_csv, list_saved_models, import_model_file, list_saved_csvs,
    list_sd_logs, detect_hold_segments, append_calibration_row,
)
from enrollment.sweep import run_static_sweep, DEFAULT_MAX_PWM, DEFAULT_SWEEP_DURATION
from ui.plots import (
    spectrogram_figure, spectrum_snapshot_figure, calibration_candidates_figure,
    resonance_bar_figure, auth_timeseries_figure, manifold_overlay_figure,
)
from enrollment.recorder import DataRecorder
from pipeline.features import extract_features, feature_matrix
from pipeline.manifold import ManifoldModel
from pipeline.motor_model import (
    MOTOR_GAIN, MotorModel, get_active_model,
    list_saved_models as list_saved_motor_models,
    save_model as save_motor_model,
    delete_model as delete_motor_model,
    set_active_model as set_active_motor_model,
)
from pipeline.preprocessing import (
    pwm_to_hz, MOTOR_COLS, MIN_MOTOR_PWM,
    load_session, infer_fs, highpass, harmonic_deconvolution,
)
from usd_motor_analysis import (
    rank_candidates, collect_segments, attribute_single_drive,
    single_drive_motor_lines, attribute_all_drive, build_vote_segs,
    find_resonance_candidates, KNOWN_CANDIDATES,
)
from usd_convert import convert as convert_usd_log
from enrollment.usd_timeline import Timeline, UsdPhaseRecorder
from test_usd_motors import run_motor_sweep_sd, sweep_duration_s
from enrollment.session_manifest import (
    SessionManifest, manifest_path, drone_dirs, micro_stem, rig_dir_for,
    ingest_upload_batch, unique_path,
    find_swap_session, swap_session_dir, swap_session_dirs, board_session_dir,
    BOOT_LIKE_KINDS, CONDITION_KINDS,
)
from crazyflie.battery import read_vbat, read_log_float, read_board_params
from enrollment.micro_sweep import (
    purposes_for, MAX_PWM as RIG2_MAX_PWM, COOLDOWN_S, run_micro,
    auto_pause_after, AUTO_PAUSE_S, AUTO_PAUSE_AFTER_ALL4_S,
    VBAT_MIN_REST_V, VBAT_WAIT_MAX_S, VBAT_RECHECK_S, micro_duration_s, BOOT_REST_S,
)
from ui.sound import alert

# ── constants ─────────────────────────────────────────────────────────────────

FEATURE_WIN_S      = 1.0
FEATURE_HOP_S      = 0.5
DEFAULT_FS         = 100.0
DEFAULT_OUTPUT_DIR = _AUTH_CLIENT / "data"
DEFAULT_MODEL_DIR  = _AUTH_CLIENT / "models"
DEFAULT_USD_DIR    = _AUTH_CLIENT / "data" / "usd"
CH_OPTIONS         = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]
CONNECTION_POLL_S  = 2.0   # how often the sidebar connection badge refreshes
                            # itself while a drone is connected — see the
                            # auto-refresh block at the bottom of this file.
DEFAULT_LOCAL_MAX_PWM = 40_000   # individual-motor / motor-pair PWM cap in the
                            # enrollment sweeps below — see run_static_sweep's
                            # local_max_pwm docstring (rig-contact artifact
                            # confirmed 2026-09-08 above ~50000 PWM on
                            # single-motor/adjacent-pair drive; the combined
                            # stage is unaffected and keeps its own slider).

# Vibration-ground-truth calibration ladder (Motor Calibration tab):
# single-drive so each hold has exactly one unambiguous spectral line.
# Capped at 48k, NOT the base laser calibration's full 55k — confirmed
# 2026-09-06: above ~50k PWM with only one motor spinning, the drone tilts
# hard enough to press the chassis against the rig's vertical support
# wires, and that contact turns into broadband stick-slip vibration that
# swamps the motor's own clean tone (operator confirmed: motor audibly
# faster and warm, not stalled or overheating — every m{N}_hold_55000
# segment recorded this way came back completely flat at the predicted
# frequency, ~60 dB below the spectrum's own peak). This is a single-motor
# RIG artifact, not a property of the motor or the comb filter: the base
# model's laser calibration is unaffected (optical, doesn't go through the
# chassis vibration path) and neither is the all-drive flight regime
# (balanced thrust, no net tilt against anything).
CALIB_DEFAULT_PWMS = [8_000, 16_000, 24_000, 32_000, 40_000, 48_000]
CALIB_HOLD_S       = 10.0   # shorter than test_usd_motors.HOLD_S's 15s default —
                            # recordings here are already per-motor (see below),
                            # so keeping each one quick matters more than
                            # squeezing out the last bit of Welch-averaging
                            # precision; still 10x nperseg=2048 at 1 kHz.


# ── initialise session state ─────────────────────────────────────────────────

init_state()


# Speed alarm of the key (identity and on-board key tabs)

def _speed_alarm_warning(r: dict) -> None:
    """Speed alarm of a boot whose key was regenerated (keyproto.fe.SPEED_ALARM_SIGMA)."""
    from keyproto.fe import SPEED_ALARM_SIGMA
    sh = r.get("speed_shift_sigma") or {}
    txt = ", ".join(f"M{m} {v:+.1f}σ" for m, v in sorted(((int(k), v) for k, v in sh.items()))
                    if v is not None and v == v)
    if r.get("speed_alarm"):
        st.warning(f"⚠️ **Speed alarm** on motor(s) {', '.join(f'M{m}' for m in r['speed_alarm'])}: "
                   f"the key was regenerated (it is the right drone), but the speed moved more than "
                   f"±{SPEED_ALARM_SIGMA:.0f}σ from the reference ({txt}): propellers or motors were touched, so "
                   "**access is denied**. The reference was NOT updated: the alarm repeats at every boot "
                   "until the drone is enrolled again (maintenance = new enrollment).")
    elif txt:
        st.caption(f"Speed against the reference: {txt} (alarm beyond ±{SPEED_ALARM_SIGMA:.0f}σ).")


# ─────────────────────────────────────────────────────────────────────────────
# Blocking sweep runner with live progress UI
# ─────────────────────────────────────────────────────────────────────────────
#
# Sweeps run synchronously in the main script thread rather than in a
# background thread. Reason: Streamlit's auto-refresh reruns were racing
# with the sweep's precise 100 ms motor-command loop for the GIL, starving
# cflib's own link-reading thread and causing "too many packets lost" /
# dead CRTP link. A single blocking script run has no such contention, and
# st.empty() placeholders still update live over the existing websocket
# connection without needing a rerun — which is what gives us the countdown
# timer and per-motor RPM display below.

def _run_sweep_blocking(worker, recorder, max_pwm: int, dur: float,
                        individual: bool = False, pairs: bool = False,
                        local_max_pwm: "int | None" = None) -> "str | None":
    """
    Run a static sweep, blocking until it finishes, updating a countdown
    timer and per-motor PWM/RPM readout live. Returns an error message
    string, or None on success.

    individual / pairs enable the extra excitation stages (single motors,
    then motor pairs) before the combined sweep — see enrollment/sweep.py
    module docstring for the modal-testing rationale. local_max_pwm caps
    those two stages separately from max_pwm (which then only bounds the
    balanced combined stage) — see run_static_sweep's docstring for why
    (rig-contact artifact confirmed 2026-09-08 above ~50000 PWM on
    single-motor/adjacent-pair drive).
    """
    if worker is None or worker.motors is None:
        return "Drone not connected"

    progress_ph = st.empty()
    motor_ph    = st.empty()

    def _progress_cb(phase, elapsed_s, total_s, remaining_s, motor_pwm):
        frac = 0.0 if total_s <= 0 else min(1.0, elapsed_s / total_s)
        progress_ph.progress(
            frac,
            text=f"{phase}  —  {remaining_s:0.0f}s remaining  "
                 f"({elapsed_s:0.0f}s / {total_s:0.0f}s)",
        )
        cols = motor_ph.columns(4)
        for i, mid in enumerate((1, 2, 3, 4)):
            pwm = motor_pwm.get(mid, 0)
            rpm = pwm_to_hz(pwm) * 60.0
            cols[i].metric(f"M{mid}", f"{pwm} PWM", f"{rpm:,.0f} RPM",
                           delta_color="off")

    error = None
    try:
        run_static_sweep(
            motors        = worker.motors,
            recorder      = recorder,
            max_pwm       = max_pwm,
            duration_s    = dur,
            individual    = individual,
            pairs         = pairs,
            local_max_pwm = local_max_pwm,
            progress_cb   = _progress_cb,
        )
    except Exception as exc:
        error = str(exc)
    finally:
        progress_ph.empty()
        motor_ph.empty()

    return error


# ─────────────────────────────────────────────────────────────────────────────
# SD-deck per-motor sweep — used by the Motor Calibration tab
# ─────────────────────────────────────────────────────────────────────────────
# Same blocking-in-main-thread rationale as _run_sweep_blocking above: this
# drives motors on a tight time.sleep()-based schedule too, so it must not
# race Streamlit's auto-refresh for the GIL.

def _run_calibration_sweep_blocking(
    worker, pwms: "list[int]", hold_s: float, motor_order: "list[int]",
    output_dir: Path,
) -> "tuple[str | None, Path | None]":
    """
    Run test_usd_motors.run_motor_sweep_sd for the given motor_order (all-
    four segment always skipped — this tab only needs one clean motor line
    per segment), live progress via a Streamlit progress bar + per-motor
    PWM/RPM readout.

    motor_order is normally a single motor: one recording per motor (each
    with its own usd.logging start/stop and its own timeline file) rather
    than one long four-motor recording, so a dropped connection or a dead
    battery partway through only costs that one motor's time, not the
    others already safely on the card.

    Returns (error_message_or_None, timeline_csv_path_or_None). The SD
    recording itself stays ON the drone's card — this only writes the PC-side
    timeline; converting still needs the card pulled and copied (see the
    Convert section right below in the tab), same as the CLI flow.
    """
    if worker is None or worker.cf is None or worker.motors is None:
        return "Drone not connected", None

    cf = worker.cf

    def get_param(name: str) -> "str | None":
        try:
            return cf.param.get_value(name, timeout=3)
        except Exception:
            return None

    if str(get_param("deck.bcUSD")) not in ("1", "1.0"):
        return "SD deck not detected.", None
    if str(get_param("usd.canLog")) not in ("1", "1.0"):
        return ("SD deck present but not ready (card FAT32? config.txt? "
                "card inserted before power-on?).", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    motor_tag = "-".join(f"m{m}" for m in motor_order)
    tl_path = output_dir / f"timeline_{motor_tag}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    tl = Timeline(tl_path)

    progress_ph = st.empty()
    motor_ph    = st.empty()

    def _progress_cb(phase, elapsed_s, total_s, remaining_s, motor_pwm):
        frac = 0.0 if total_s <= 0 else min(1.0, elapsed_s / total_s)
        progress_ph.progress(
            frac,
            text=f"{phase}  —  {remaining_s:0.0f}s remaining  "
                 f"({elapsed_s:0.0f}s / {total_s:0.0f}s)",
        )
        cols = motor_ph.columns(4)
        for i, mid in enumerate((1, 2, 3, 4)):
            pwm = motor_pwm.get(mid, 0)
            rpm = pwm_to_hz(pwm, motor_id=mid) * 60.0
            cols[i].metric(f"M{mid}", f"{pwm} PWM",
                           f"{rpm:,.0f} RPM" if pwm else "off",
                           delta_color="off")

    error = None
    try:
        run_motor_sweep_sd(
            cf, worker.motors, tl,
            pwms=pwms, hold_s=hold_s, motor_order=motor_order,
            skip_all_hold=True, progress_cb=_progress_cb,
        )
    except Exception as exc:
        error = str(exc)
    finally:
        tl.close()
        progress_ph.empty()
        motor_ph.empty()

    return error, tl_path


def _run_usd_sweep_blocking(
    worker, max_pwm: int, dur: float, individual: bool, pairs: bool,
    output_dir: Path, local_max_pwm: "int | None" = None,
) -> "tuple[str | None, Path | None]":
    """
    SD-deck (~1 kHz onboard) counterpart to _run_sweep_blocking above: runs
    the exact same enrollment.sweep.run_static_sweep() protocol — same
    excitation stages, same phase labels — but records via the microSD deck
    (UsdPhaseRecorder + Timeline) instead of a 100 Hz DataRecorder, so the
    resulting recording gets the harmonic comb filter actually active
    (f0 ~ 250-300 Hz needs > 500 Hz Nyquist — 100 Hz telemetry can't carry
    it, see CLAUDE.md). Same blocking-in-main-thread rationale as above.

    Returns (error_message_or_None, timeline_csv_path_or_None). The SD
    recording itself stays ON the drone's card — converting still needs the
    card pulled and copied by hand, same as the CLI's --usd flow and the
    Motor Calibration tab's SD path (no software can skip that step).
    """
    if worker is None or worker.cf is None or worker.motors is None:
        return "Drone not connected", None

    cf = worker.cf

    def get_param(name: str) -> "str | None":
        try:
            return cf.param.get_value(name, timeout=3)
        except Exception:
            return None

    if str(get_param("deck.bcUSD")) not in ("1", "1.0"):
        return "SD deck not detected.", None
    if str(get_param("usd.canLog")) not in ("1", "1.0"):
        return ("SD deck present but not ready (card FAT32? config.txt? "
                "card inserted before power-on?).", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    tl_path = output_dir / f"timeline_static_{datetime.now():%Y%m%d_%H%M%S}.csv"
    tl = Timeline(tl_path)
    phase_recorder = UsdPhaseRecorder(tl)

    progress_ph = st.empty()
    motor_ph    = st.empty()

    def _progress_cb(phase, elapsed_s, total_s, remaining_s, motor_pwm):
        frac = 0.0 if total_s <= 0 else min(1.0, elapsed_s / total_s)
        progress_ph.progress(
            frac,
            text=f"{phase}  —  {remaining_s:0.0f}s remaining  "
                 f"({elapsed_s:0.0f}s / {total_s:0.0f}s)",
        )
        cols = motor_ph.columns(4)
        for i, mid in enumerate((1, 2, 3, 4)):
            pwm = motor_pwm.get(mid, 0)
            rpm = pwm_to_hz(pwm, motor_id=mid) * 60.0
            cols[i].metric(f"M{mid}", f"{pwm} PWM",
                           f"{rpm:,.0f} RPM" if pwm else "off",
                           delta_color="off")
        tl.log("pwm", max(motor_pwm.values()))

    error = None
    try:
        cf.param.set_value("usd.logging", "1")
        tl.log("usd_start")
        run_static_sweep(
            motors        = worker.motors,
            recorder      = phase_recorder,
            max_pwm       = max_pwm,
            duration_s    = dur,
            individual    = individual,
            pairs         = pairs,
            local_max_pwm = local_max_pwm,
            progress_cb   = _progress_cb,
        )
    except Exception as exc:
        error = str(exc)
    finally:
        try:
            cf.param.set_value("usd.logging", "0")
            tl.log("usd_stop")
        except Exception:
            pass
        tl.close()
        progress_ph.empty()
        motor_ph.empty()

    return error, tl_path


def _load_sd_recording(log_path: Path, timeline_path: Path,
                       force_reconvert: bool = False) -> "tuple[pd.DataFrame | None, str, str | None]":
    """
    Load a raw SD-deck recording for the Motor Calibration tab, converting
    it on the fly if needed.

    The converted CSV is cached to disk as ``<logfile>_1khz.csv`` (usd_
    convert.py's own default naming) rather than recomputed on every
    Streamlit rerun: decoding a 250k+-row binary log and reconstructing
    phase/PWM from the timeline is real work, and the calibration review
    loop below reruns the whole script on every button click. If that file
    already exists it's reused as-is unless force_reconvert is set (e.g.
    the timeline was corrected and the old conversion is stale).

    Returns (df, out_csv_path_str, message_or_None) — message is an info
    string worth surfacing ("used cached conversion" / "converted N
    samples") on success, or None; on failure df is None and message
    carries the error.
    """
    out_csv = log_path.with_name(log_path.name + "_1khz.csv")
    if out_csv.exists() and not force_reconvert:
        try:
            return load_session(str(out_csv)), str(out_csv), (
                f"Using cached conversion `{out_csv.name}` "
                "(check 'Force reconvert' to redo it)."
            )
        except Exception as exc:
            return None, str(out_csv), f"Cannot load cached CSV: {exc}"

    try:
        df, stats = convert_usd_log(log_path, timeline_path)
    except Exception as exc:
        return None, str(out_csv), f"Conversion failed: {exc}"
    df.to_csv(out_csv, index=False)
    return df, str(out_csv), f"Converted {stats['n_samples']:,} samples → `{out_csv.name}`."


# ─────────────────────────────────────────────────────────────────────────────
# Shared "load a recording" picker — SD log + timeline (auto-convert) or an
# already-converted CSV, both with a directory browser + OS-upload fallback.
# Used by the Spectrogram, Motor Calibration AND (2026-09-08) Offline
# Analysis tabs so there's one implementation of this UI, not several copies
# drifting apart.
# ─────────────────────────────────────────────────────────────────────────────

def _list_csvs_multi(data_dirs: "list[Path]") -> "list[dict]":
    """list_saved_csvs() merged across several directories, newest first.
    Each entry gets a 'dir' field (the directory it came from) so the UI can
    disambiguate filenames when more than one directory is searched."""
    out = []
    for d in data_dirs:
        for f in list_saved_csvs(d):
            out.append({**f, "dir": d})
    out.sort(key=lambda f: f["mtime"], reverse=True)
    return out


def _recording_picker(
    key_prefix: str,
    data_dirs: "list[Path] | None" = None,
    multi: bool = False,
) -> "tuple[pd.DataFrame, str] | list[str] | None":
    """
    Renders the picker widgets.

    multi=False (default, used by Spectrogram/Motor Calibration): returns
    (df, source_label) the render where "Load recording" was clicked and
    succeeded, else None — a single file, one-shot load.

    multi=True (used by Offline Analysis, for pooling several sessions into
    one enrollment): renders checkboxes instead of a load button, keeps the
    selection in session_state[f"{key_prefix}_selected"], and returns the
    live list[str] of selected CSV paths on every render (so callers just
    read the return value, no separate click needed) — SD-log conversions
    get an "Convert & add to selection" button instead of "Load recording".

    data_dirs controls which directories the "Already-converted CSV" browser
    searches (default: just data/usd/, unchanged from before); raw SD logs
    and timelines always live in data/usd/ regardless, so that browsing
    stays pointed there. key_prefix keeps each caller's widget keys
    independent (e.g. "spec" vs "calib" vs "off_enroll").
    """
    data_dirs = data_dirs or [DEFAULT_USD_DIR]
    sel_key = f"{key_prefix}_selected"
    if multi:
        st.session_state.setdefault(sel_key, [])
        selected = set(st.session_state[sel_key])

    load_mode = st.radio(
        "Input", ["SD log + timeline (auto-convert)", "Already-converted CSV"],
        horizontal=True, key=f"{key_prefix}_load_mode",
    )

    if load_mode == "SD log + timeline (auto-convert)":
        sd_logs  = list_sd_logs(DEFAULT_USD_DIR)
        tl_files = [f for f in list_saved_csvs(DEFAULT_USD_DIR)
                   if f["name"].startswith("timeline_")]
        st.session_state.setdefault(f"{key_prefix}_sd_log_input", "")
        st.session_state.setdefault(
            f"{key_prefix}_tl_input", st.session_state.get("calib_last_timeline", ""))

        if sd_logs:
            with st.expander(f"📂 Browse SD logs in `data/usd/` ({len(sd_logs)})",
                             expanded=not st.session_state[f"{key_prefix}_sd_log_input"]):
                up = st.file_uploader(
                    "…or upload a logNN file from your computer",
                    key=f"{key_prefix}_log_uploader",
                    help="Opens your OS's file picker. It can't be pre-pointed "
                         "at data/usd/ (browsers don't allow a web page to set "
                         "a native dialog's starting folder) — the list below "
                         "is the quicker way to pick a file that's already there.",
                )
                if up is not None and st.button("Save & use this file",
                                                key=f"{key_prefix}_log_upload_use_btn"):
                    DEFAULT_USD_DIR.mkdir(parents=True, exist_ok=True)
                    save_path = DEFAULT_USD_DIR / up.name
                    save_path.write_bytes(up.getvalue())
                    st.session_state[f"{key_prefix}_sd_log_input"] = str(save_path)
                    st.rerun()
                st.divider()
                for f in sd_logs:
                    lc1, lc2, lc3 = st.columns([3, 2, 1])
                    lc1.text(f["name"])
                    lc2.caption(f'{f["size_kb"]:.0f} KB')
                    if lc3.button("Use", key=f"{key_prefix}_use_log_{f['name']}"):
                        st.session_state[f"{key_prefix}_sd_log_input"] = str(f["path"])
                        st.rerun()
        else:
            st.caption(f"No SD log files (logNN) in `{DEFAULT_USD_DIR}` yet.")

        sd_log_path = st.text_input(
            "...or type/paste the logNN path", key=f"{key_prefix}_sd_log_input",
            placeholder="data/usd/log07",
        )

        if tl_files:
            with st.expander(f"📂 Browse timelines ({len(tl_files)})",
                             expanded=not st.session_state[f"{key_prefix}_tl_input"]):
                for f in tl_files:
                    tc1, tc2, tc3 = st.columns([3, 2, 1])
                    tc1.text(f["name"])
                    tc2.caption(f'{f["size_kb"]:.0f} KB · {f["mtime"]:%Y-%m-%d %H:%M}')
                    if tc3.button("Use", key=f"{key_prefix}_use_tl_{f['name']}"):
                        st.session_state[f"{key_prefix}_tl_input"] = str(f["path"])
                        st.rerun()
        else:
            st.caption(f"No timeline CSVs in `{DEFAULT_USD_DIR}` yet.")

        tl_input = st.text_input(
            "Matching timeline CSV", key=f"{key_prefix}_tl_input",
            placeholder="data/usd/timeline_m1_....csv",
            help="Whichever timeline was recorded together with this logNN file.",
        )
        force_reconvert = st.checkbox(
            "Force reconvert (ignore cached CSV)", value=False,
            key=f"{key_prefix}_force_reconvert",
        )

        btn_label = "Convert & add to selection" if multi else "Load recording"
        if st.button(btn_label, key=f"{key_prefix}_load_sd_btn",
                     disabled=not (sd_log_path and tl_input)):
            df, out_csv, msg = _load_sd_recording(
                Path(sd_log_path.strip()), Path(tl_input.strip()), force_reconvert,
            )
            if df is None:
                st.error(msg)
            elif multi:
                st.session_state[sel_key] = sorted(set(st.session_state[sel_key]) | {out_csv})
                st.success(msg)
                st.rerun()
            else:
                st.info(msg)
                return df, out_csv
        return st.session_state[sel_key] if multi else None

    else:
        usd_files = [f for f in _list_csvs_multi(data_dirs)
                    if not f["name"].startswith("timeline_")]
        browse_label = ("`" + "`, `".join(str(d) for d in data_dirs) + "`"
                        if len(data_dirs) > 1 else f"`{data_dirs[0]}`")

        if multi:
            st.caption(f"Select one or more recordings from {browse_label} to pool.")
            if not usd_files:
                st.caption("No CSVs there yet.")
            else:
                show_dir = len(data_dirs) > 1
                for f in usd_files:
                    path_str = str(f["path"])
                    label = f'{f["name"]}  ·  {f["size_kb"]:.0f} KB  ·  {f["mtime"]:%Y-%m-%d %H:%M}'
                    if show_dir:
                        label = f'[{f["dir"].name}] {label}'
                    checked = st.checkbox(label, value=(path_str in selected),
                                          key=f"{key_prefix}_pick_{f['dir'].name}_{f['name']}")
                    if checked:
                        selected.add(path_str)
                    else:
                        selected.discard(path_str)
            st.session_state[sel_key] = sorted(selected)

            with st.expander("⬆ Upload a CSV from your computer"):
                up = st.file_uploader("Upload", type=["csv"],
                                      key=f"{key_prefix}_uploader",
                                      label_visibility="collapsed")
                if up is not None and st.button("Save & add to selection",
                                                key=f"{key_prefix}_upload_use_btn"):
                    data_dirs[0].mkdir(parents=True, exist_ok=True)
                    save_path = data_dirs[0] / up.name
                    save_path.write_bytes(up.getvalue())
                    st.session_state[sel_key] = sorted(
                        set(st.session_state[sel_key]) | {str(save_path)})
                    st.rerun()

            return st.session_state[sel_key]

        st.session_state.setdefault(f"{key_prefix}_path_input", "")
        if usd_files:
            with st.expander(f"📂 Browse {browse_label} ({len(usd_files)} file(s))",
                             expanded=not st.session_state[f"{key_prefix}_path_input"]):
                up = st.file_uploader(
                    "…or upload a CSV from your computer", type=["csv"],
                    key=f"{key_prefix}_uploader",
                    help="Opens your OS's file picker. It can't be pre-pointed "
                         "at a specific folder (browsers don't allow a web page "
                         "to set a native dialog's starting folder) — the list "
                         "below is the quicker way to pick a file that's "
                         "already there.",
                )
                if up is not None and st.button("Save & use this file",
                                                key=f"{key_prefix}_upload_use_btn"):
                    data_dirs[0].mkdir(parents=True, exist_ok=True)
                    save_path = data_dirs[0] / up.name
                    save_path.write_bytes(up.getvalue())
                    st.session_state[f"{key_prefix}_path_input"] = str(save_path)
                    st.rerun()
                st.divider()
                show_dir = len(data_dirs) > 1
                for f in usd_files:
                    fc1, fc2, fc3 = st.columns([3, 2, 1])
                    fc1.text(f'[{f["dir"].name}] {f["name"]}' if show_dir else f["name"])
                    fc2.caption(f'{f["size_kb"]:.0f} KB · {f["mtime"]:%Y-%m-%d %H:%M}')
                    if fc3.button("Use", key=f"{key_prefix}_use_{f['dir'].name}_{f['name']}"):
                        st.session_state[f"{key_prefix}_path_input"] = str(f["path"])
                        st.rerun()
        else:
            st.caption(f"No CSVs in {browse_label} yet.")

        csv_path = st.text_input(
            "...or type/paste a CSV path", key=f"{key_prefix}_path_input",
            placeholder="data/usd/log06_1khz.csv",
        )

        if st.button("Load recording", disabled=not csv_path, key=f"{key_prefix}_load_btn"):
            try:
                return load_session(csv_path.strip()), csv_path.strip()
            except Exception as exc:
                st.error(f"Cannot load: {exc}")
                return None
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Model explorer — list / load / export / import saved ManifoldModel files
# ─────────────────────────────────────────────────────────────────────────────

def _model_explorer(model_dir: Path, key_prefix: str):
    """
    Render a compact list of saved .pkl models (name, size, date) with a
    Load button and a download (export) button per row, plus an uploader
    to import a model file from elsewhere into ``model_dir``.

    Returns a freshly loaded ManifoldModel if the user clicked Load this
    render, else None.
    """
    models = list_saved_models(model_dir)
    loaded = None

    if not models:
        st.caption(f"No saved models in `{model_dir.name}/` yet.")
    else:
        for m in models:
            c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
            c1.text(m["name"])
            c2.caption(f'{m["size_kb"]:.0f} KB · {m["mtime"]:%Y-%m-%d %H:%M}')
            if c3.button("Load", key=f"{key_prefix}_load_{m['name']}"):
                try:
                    loaded = ManifoldModel.load(m["path"])
                    st.session_state[f"_{key_prefix}_loaded_path"] = str(m["path"])
                except Exception as exc:
                    st.error(f"Cannot load {m['name']}: {exc}")
            c4.download_button(
                "⬇", data=m["path"].read_bytes(), file_name=m["path"].name,
                key=f"{key_prefix}_dl_{m['name']}", help="Export this model file",
            )

    with st.expander("⬆ Import a model file"):
        up = st.file_uploader("Upload a .pkl model", type=["pkl"],
                              key=f"{key_prefix}_uploader",
                              label_visibility="collapsed")
        if up is not None and st.button("Save to models folder",
                                         key=f"{key_prefix}_import_btn"):
            saved = import_model_file(up.getvalue(), up.name, model_dir)
            st.success(f"Imported: {saved.name}")
            st.rerun()

    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# Auto-disconnect dead workers (battery death / packet loss / any crash)
# ─────────────────────────────────────────────────────────────────────────────
# Must run BEFORE the sidebar renders: the sidebar's Connect/Disconnect
# buttons and status badge read st.session_state.worker directly. If a
# worker's background thread already died but hadn't been cleared yet, the
# sidebar kept showing "connected" (Connect button disabled) until some
# later rerun caught up — forcing the user to click Disconnect, wait, then
# Connect instead of just Connect once.

_w = st.session_state.worker
if _w is not None and not _w.is_alive():
    reason = _w.error or "Connection lost (drone disconnected or battery died)"
    st.session_state["_disconnect_msg"] = reason
    st.session_state.worker = None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚁 UAVibes")
    st.caption("Physical Identity Extraction — Crazyflie 2.1")
    st.divider()

    # ── Connection (the only thing left in the sidebar — 2026-09-07: Mode,
    # channel/rate selectors, and Enrollment/Authentication controls were
    # all specific to the removed Live tab; Offline Analysis has its own
    # independent record/build-model flow and never touched these) ────────
    st.subheader("Connection")

    # Preset buttons
    p1, p2 = st.columns(2)
    with p1:
        if st.button("📡 USB  (usb://0)", width='stretch',
                     disabled=(st.session_state.worker is not None)):
            st.session_state.uri_value = "usb://0"
            st.rerun()
    with p2:
        if st.button("📻 Radio  (80/2M)", width='stretch',
                     disabled=(st.session_state.worker is not None)):
            st.session_state.uri_value = "radio://0/80/2M/E7E7E7E7E7"
            st.rerun()

    uri_input = st.text_input(
        "Custom URI",
        key="uri_value",
        help="usb://0  or  radio://0/80/2M/E7E7E7E7E7",
    )

    c1, c2 = st.columns(2)
    with c1:
        connect_btn = st.button(
            "Connect",
            disabled=(st.session_state.worker is not None),
            width='stretch',
        )
    with c2:
        disconnect_btn = st.button(
            "Disconnect",
            disabled=(st.session_state.worker is None),
            width='stretch',
        )

    if connect_btn:
        worker = DroneWorker(st.session_state.uri_value.strip())
        worker.start()
        st.session_state.worker             = worker
        st.session_state["_disconnect_msg"] = ""
        st.rerun()

    if disconnect_btn and st.session_state.worker:
        st.session_state.worker.stop()
        st.session_state.worker = None
        st.rerun()

    # Status badge — kept current in real time by the unconditional
    # (while-connected) rerun loop at the bottom of this file.
    worker = st.session_state.worker
    if worker is None:
        st.info("○  Disconnected", icon="⚪")
    elif worker.error:
        st.error(f"✗  {worker.error}", icon="🔴")
    elif worker.connected.is_set():
        st.success(f"●  Connected — {worker.uri}", icon="🟢")
    else:
        st.warning("⟳  Connecting…", icon="🟡")


# Show last disconnect reason as a banner (cleared on next successful connect)
if st.session_state.get("_disconnect_msg"):
    st.warning(
        f"Drone disconnected: {st.session_state['_disconnect_msg']}  "
        "— just click Connect again below to retry.",
        icon="⚠️",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main panel — tabs
# ─────────────────────────────────────────────────────────────────────────────

(tab_offline, tab_models, tab_spectro, tab_calib, tab_resonances, tab_rig2, tab_identity,
 tab_onboard) = st.tabs(
    ["📂  Offline Analysis", "⚙️  Motor Models", "🔬  Spectrogram",
     "🎯  Motor Calibration", "🔔  Resonances", "🆕  New Rig Session", "🔑  Identity check",
     "📡  On-board key"]
)


# ==============================================================================
# (Live tab removed 2026-09-07, along with the Mode/channel/rate sidebar controls
# it alone used -- one operator, one drone; Offline Analysis + the SD-deck tabs
# cover the workflow now. The live-buffer helpers it used (signal_figure,
# fft_figure, feature_bar_figure, manifold_figure, preprocess_buffer,
# buffer_to_df) are still in ui/plots.py and ui/state.py if ever revived.)
# ==============================================================================

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Offline Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab_offline:

    st.markdown("### Offline Analysis — CSV enrollment & testing")
    st.caption(
        "Build a manifold model from a recorded CSV, then test other sessions "
        "against it to verify same-drone acceptance and impostor rejection."
    )

    # ── Shared helpers ────────────────────────────────────────────────────────

    phase_opts = ["(all phases)", "all_four_ramp", "all_four_hold",
                  "all_four_ramp + all_four_hold", "hover", "hover_manual"]

    def _resolve_phase(sel: str):
        if sel == "all_four_ramp":           return "all_four_ramp"
        if sel == "all_four_hold":           return "all_four_hold"
        if sel == "all_four_ramp + all_four_hold": return ["all_four_ramp", "all_four_hold"]
        if sel == "(all phases)":            return None
        return sel

    def _usd_sweep_section(role: str, last_key: str) -> str:
        """
        SD-deck (~1 kHz onboard) variant of the single-sweep record flow
        used by _sweep_section below — record here, then (hardware step,
        can't be automated) power off / pull the card / copy the file over,
        then convert via the same shared _recording_picker used everywhere
        else in the app. Returns the converted CSV path once available, and
        remembers it in session_state[last_key] exactly like the 100 Hz
        path does, so the rest of _sweep_section's contract is unchanged.
        """
        live_worker = st.session_state.get("worker")
        drone_ok    = (live_worker is not None
                       and live_worker.is_alive()
                       and live_worker.connected.is_set()
                       and live_worker.motors is not None)

        if not drone_ok:
            st.warning("Connect to the drone in the sidebar first.", icon="⚠️")
        else:
            st.success("Drone connected — ready to record.", icon="🟢")

        max_pwm = st.slider("Max PWM (combined stage)", 10_000, 55_000, DEFAULT_MAX_PWM,
                            step=1_000, key=f"off_{role}_usd_max_pwm",
                            help="Drone must be restrained (e.g. the test rig) — "
                                 "this drives real props at flight-relevant PWM.")
        local_max_pwm = st.slider(
            "Max PWM (individual + pairs)", 10_000, 55_000, DEFAULT_LOCAL_MAX_PWM,
            step=1_000, key=f"off_{role}_usd_local_max_pwm",
            help="Kept lower than the combined stage — single-motor/adjacent-"
                 "pair drive tilts the drone into the rig's support wires "
                 "above ~50000 PWM (confirmed 2026-09-08, log15).",
        )
        tc1, tc2 = st.columns(2)
        inc_individual = tc1.checkbox("+ Individual motors", value=True,
                                      key=f"off_{role}_usd_individual",
                                      help="Match the excitation richness used for enrollment.")
        inc_pairs      = tc2.checkbox("+ Motor pairs", value=True,
                                      key=f"off_{role}_usd_pairs")

        if st.button("Record via SD", width='stretch',
                     disabled=not drone_ok, key=f"off_{role}_usd_rec_btn"):
            err, tl_path = _run_usd_sweep_blocking(
                live_worker, max_pwm, DEFAULT_SWEEP_DURATION,
                inc_individual, inc_pairs, DEFAULT_USD_DIR,
                local_max_pwm=local_max_pwm,
            )
            if err:
                st.error(f"SD recording failed: {err}")
            else:
                # Prefill the picker's timeline field below so, once the
                # card is copied over, converting is one click.
                st.session_state[f"off_{role}_tl_input"] = str(tl_path)
                st.success(f"Recording done — timeline saved: {tl_path.name}. "
                          "Power off, pull the SD card, copy the newest logNN "
                          "into `data/usd/`, then convert it below.")
                st.rerun()

        st.markdown("###### Convert a recording")
        picked = _recording_picker(
            f"off_{role}", data_dirs=[DEFAULT_OUTPUT_DIR, DEFAULT_USD_DIR],
        )
        if picked is not None:
            st.session_state[last_key] = picked[1]

        last = st.session_state.get(last_key, "")
        if last:
            st.success(f"Last recording saved: `{last}`")
        return last

    def _sweep_section(role: str) -> "str | None":
        """
        Render the 'record new sweep OR pick existing CSV' widget.
        Returns the selected CSV path (or empty string).
        """
        last_key  = f"off_{role}_last_csv"
        drone_key = f"off_{role}_drone_id"
        pwm_key   = f"off_{role}_max_pwm"

        src = st.radio(
            "Data source",
            ["Use existing CSV", "Record new sweep from drone",
             "Record via SD (1 kHz onboard)"],
            horizontal=True,
            key=f"off_{role}_src",
        )

        if src == "Use existing CSV":
            picked = _recording_picker(
                f"off_{role}", data_dirs=[DEFAULT_OUTPUT_DIR, DEFAULT_USD_DIR],
            )
            if picked is not None:
                st.session_state[last_key] = picked[1]
            last = st.session_state.get(last_key, "")
            if last:
                st.success(f"Selected: `{last}`")
            return last

        if src == "Record via SD (1 kHz onboard)":
            return _usd_sweep_section(role, last_key)

        # ── Record new sweep ──────────────────────────────────────────────────
        live_worker = st.session_state.get("worker")
        drone_ok    = (live_worker is not None
                       and live_worker.is_alive()
                       and live_worker.connected.is_set()
                       and live_worker.motors is not None)

        if not drone_ok:
            st.warning("Connect to the drone in the sidebar first.", icon="⚠️")
        else:
            st.success("Drone connected — ready to record.", icon="🟢")

        d_id    = st.text_input("Drone ID", value="cf_001", key=drone_key)
        max_pwm = st.slider("Max PWM (combined stage)", 10_000, 45_000, DEFAULT_MAX_PWM,
                            step=1_000, key=pwm_key)
        local_max_pwm = st.slider(
            "Max PWM (individual + pairs)", 10_000, 45_000, DEFAULT_LOCAL_MAX_PWM,
            step=1_000, key=f"off_{role}_local_max_pwm",
            help="Kept lower than the combined stage — single-motor/adjacent-"
                 "pair drive tilts the drone into the rig's support wires "
                 "above ~50000 PWM (confirmed 2026-09-08, log15).",
        )
        st.caption(f"Combined-sweep duration: fixed at {DEFAULT_SWEEP_DURATION:.0f}s "
                  "(standardised — must match enrollment for comparisons to be valid).")
        tc1, tc2 = st.columns(2)
        inc_individual = tc1.checkbox("+ Individual motors", value=True,
                                      key=f"off_{role}_individual",
                                      help="Match the excitation richness used for enrollment.")
        inc_pairs      = tc2.checkbox("+ Motor pairs", value=True,
                                      key=f"off_{role}_pairs")

        if st.button("Record Sweep", width='stretch',
                     disabled=not drone_ok, key=f"off_{role}_rec_btn"):
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = DEFAULT_OUTPUT_DIR / f"{d_id}_static_{ts}.csv"
            DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)

            rec = DataRecorder()
            rec.start(out, phase="start")

            # IMU samples only ever reach a recorder that is wired onto
            # worker.recorder (DroneWorker._on_sample writes there) — a
            # recorder merely passed to run_static_sweep() never receives
            # any data on its own. Attach it for the duration of the sweep.
            prev_recorder      = live_worker.recorder
            live_worker.recorder = rec
            try:
                err = _run_sweep_blocking(live_worker, rec, max_pwm, DEFAULT_SWEEP_DURATION,
                                          individual=inc_individual, pairs=inc_pairs,
                                          local_max_pwm=local_max_pwm)
            finally:
                live_worker.recorder = prev_recorder
                rec.stop()

            if err:
                st.error(f"Sweep failed: {err}")
            elif rec.sample_count == 0:
                st.error("Sweep completed but 0 samples were recorded — "
                         "check the drone connection.")
            else:
                st.session_state[last_key] = str(out)
                st.success(f"Recording saved: {out.name}  ({rec.sample_count:,} samples)")

        last = st.session_state.get(last_key, "")
        if last:
            st.success(f"Last recording saved: `{last}`")
        return last

    def _enrollment_data_section() -> "list[str]":
        """
        Multi-session enrollment data picker — the shared _recording_picker
        (multi=True) browses both data/ (100 Hz) and data/usd/ (1 kHz SD)
        recordings and keeps a pooled selection, plus two 'record a new
        sweep' controls (100 Hz / SD) that feed fresh recordings straight
        into the same browser above.

        Pooling several independent sweep sessions — each ideally covering
        combined + individual-motor + motor-pair excitation — gives the
        manifold fit a much better estimate of a drone's true intra-session
        variability than a single recording can, which is what a properly
        calibrated authentication threshold needs.
        """
        st.caption("Select one or more recordings to pool into a single "
                  "enrollment — multiple independent sessions build a more "
                  "robust model than any single sweep.")
        selected = _recording_picker(
            "off_enroll", data_dirs=[DEFAULT_OUTPUT_DIR, DEFAULT_USD_DIR], multi=True,
        )

        live_worker = st.session_state.get("worker")
        drone_ok    = (live_worker is not None
                       and live_worker.is_alive()
                       and live_worker.connected.is_set()
                       and live_worker.motors is not None)

        if not drone_ok:
            st.warning("Connect to the drone in the sidebar first.", icon="⚠️")
        else:
            st.success("Drone connected — ready to record.", icon="🟢")

        st.markdown("###### Record a new sweep session")
        d_id    = st.text_input("Drone ID", value="cf_001", key="off_enroll_drone_id")
        max_pwm = st.slider("Max PWM (combined stage)", 10_000, 45_000, DEFAULT_MAX_PWM,
                            step=1_000, key="off_enroll_max_pwm")
        local_max_pwm = st.slider(
            "Max PWM (individual + pairs)", 10_000, 45_000, DEFAULT_LOCAL_MAX_PWM,
            step=1_000, key="off_enroll_local_max_pwm",
            help="Kept lower than the combined stage — single-motor/adjacent-"
                 "pair drive tilts the drone into the rig's support wires "
                 "above ~50000 PWM, injecting rig noise instead of chassis "
                 "signal into exactly the phases meant to isolate local "
                 "asymmetries (confirmed 2026-09-08, log15).",
        )
        st.caption(f"Combined-sweep duration: fixed at {DEFAULT_SWEEP_DURATION:.0f}s "
                  "(standardised across every recording — a duration mismatch between "
                  "enrollment and test sessions was found to confound comparisons).")
        ec1, ec2 = st.columns(2)
        inc_individual = ec1.checkbox("+ Individual motors (+~90s)", value=True,
                                      key="off_enroll_individual")
        inc_pairs      = ec2.checkbox("+ Motor pairs (+~140s)", value=True,
                                      key="off_enroll_pairs")

        if st.button("Record Sweep Session", width='stretch',
                     disabled=not drone_ok, key="off_enroll_rec_btn"):
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = DEFAULT_OUTPUT_DIR / f"{d_id}_static_{ts}.csv"
            DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)

            rec = DataRecorder()
            rec.start(out, phase="start")
            prev_recorder         = live_worker.recorder
            live_worker.recorder  = rec
            try:
                err = _run_sweep_blocking(live_worker, rec, max_pwm, DEFAULT_SWEEP_DURATION,
                                          individual=inc_individual, pairs=inc_pairs,
                                          local_max_pwm=local_max_pwm)
            finally:
                live_worker.recorder = prev_recorder
                rec.stop()

            if err:
                st.error(f"Sweep failed: {err}")
            elif rec.sample_count == 0:
                st.error("Sweep completed but 0 samples were recorded — "
                         "check the drone connection.")
            else:
                st.session_state["off_enroll_selected"] = sorted(
                    set(st.session_state.get("off_enroll_selected", [])) | {str(out)}
                )
                st.success(f"Recorded and added to selection: {out.name}  "
                          f"({rec.sample_count:,} samples)")
                st.rerun()

        st.markdown("###### Or record via SD deck (~1 kHz onboard)")
        st.caption(
            "Same protocol, recorded onboard at ~1 kHz instead of 100 Hz CRTP "
            "telemetry — needed for the harmonic comb filter to actually "
            "notch the motor lines out (f0 ~250-300 Hz is above the 100 Hz "
            "link's 50 Hz Nyquist, see CLAUDE.md). After recording: power "
            "off, pull the SD card, copy the file over, then use the "
            "'SD log + timeline' browser above to convert & add it — that "
            "hardware step can't be automated."
        )
        max_pwm_usd = st.slider("Max PWM (combined stage)", 10_000, 55_000, DEFAULT_MAX_PWM,
                                step=1_000, key="off_enroll_usd_max_pwm",
                                help="Drone must be restrained (e.g. the test "
                                     "rig) — this drives real props at "
                                     "flight-relevant PWM, not a bare desk sweep.")
        local_max_pwm_usd = st.slider(
            "Max PWM (individual + pairs)", 10_000, 55_000, DEFAULT_LOCAL_MAX_PWM,
            step=1_000, key="off_enroll_usd_local_max_pwm",
            help="See the combined-path slider above for why this is lower.",
        )
        uc1, uc2 = st.columns(2)
        inc_individual_usd = uc1.checkbox("+ Individual motors (+~90s)", value=True,
                                          key="off_enroll_usd_individual")
        inc_pairs_usd = uc2.checkbox("+ Motor pairs (+~140s)", value=True,
                                     key="off_enroll_usd_pairs")

        if st.button("Record via SD", width='stretch',
                     disabled=not drone_ok, key="off_enroll_usd_rec_btn"):
            err, tl_path = _run_usd_sweep_blocking(
                live_worker, max_pwm_usd, DEFAULT_SWEEP_DURATION,
                inc_individual_usd, inc_pairs_usd, DEFAULT_USD_DIR,
                local_max_pwm=local_max_pwm_usd,
            )
            if err:
                st.error(f"SD recording failed: {err}")
            else:
                # Prefill the picker's timeline field above so, once the SD
                # card is copied over, "Convert & add to selection" is a
                # one-click step there instead of retyping the path.
                st.session_state["off_enroll_tl_input"] = str(tl_path)
                st.success(f"Recording done — timeline saved: {tl_path.name}. "
                          "Power off, pull the SD card, copy the newest logNN "
                          "into `data/usd/`, then convert it in the browser above.")
                st.rerun()

        return selected or []

    # ══════════════════════════════════════════════════════════════════════════
    col_enroll, col_test = st.columns(2, gap="large")

    # ── Left column: Enrollment ───────────────────────────────────────────────
    with col_enroll:
        st.markdown("#### 1. Model")

        with st.expander("📦 Load an existing model", expanded=False):
            _picked = _model_explorer(DEFAULT_MODEL_DIR, key_prefix="off")
            if _picked is not None:
                st.session_state.off_model        = _picked
                st.session_state.off_model_path   = st.session_state.get("_off_loaded_path", "")
                st.session_state.off_test_results = None
                st.session_state.off_test_emb     = None
                st.success("Model loaded.")

        st.markdown("##### Or build a new model from recording(s)")
        enroll_paths = _enrollment_data_section()
        phase_sel    = st.selectbox("Phase filter", phase_opts,
                                    index=0, key="off_phase_sel",
                                    help="'(all phases)' is recommended when pooling "
                                         "multi-config sweeps (individual + pairs + combined).")
        model_name   = st.text_input("Model name", value="cf_001", key="off_model_name")

        if enroll_paths:
            st.caption(f"{len(enroll_paths)} recording(s) selected for pooling.")

        if st.button("Build Model", width='stretch',
                     disabled=not enroll_paths):
            X_parts, n_windows_total, failed = [], 0, []
            with st.spinner(f"Preprocessing {len(enroll_paths)} recording(s)…"):
                for p in enroll_paths:
                    df_e, fs_e = preprocess_csv(p, phase_filter=_resolve_phase(phase_sel))
                    if df_e is None:
                        failed.append(Path(p).name)
                        continue
                    try:
                        feat_e = extract_features(df_e, fs_e,
                                                  window_s=FEATURE_WIN_S,
                                                  hop_s=FEATURE_HOP_S)
                        X_e = feature_matrix(feat_e)
                    except Exception:
                        failed.append(Path(p).name)
                        continue
                    X_parts.append(X_e)
                    n_windows_total += X_e.shape[0]

            if failed:
                st.warning(f"Skipped {len(failed)} recording(s) that could not be "
                          f"processed: {', '.join(failed)}")

            if not X_parts:
                st.error("No usable recordings — check the phase filter matches "
                         "what was actually recorded.")
            else:
                X_pooled = np.vstack(X_parts)
                with st.spinner(f"Fitting manifold on {X_pooled.shape[0]} windows "
                               f"from {len(X_parts)} recording(s)…"):
                    try:
                        mdl = ManifoldModel()
                        mdl.fit_enrollment(X_pooled)
                        DEFAULT_MODEL_DIR.mkdir(exist_ok=True)
                        save_path = DEFAULT_MODEL_DIR / f"{model_name}.pkl"
                        mdl.save(save_path)
                        st.session_state.off_model        = mdl
                        st.session_state.off_model_path   = str(save_path)
                        st.session_state.off_test_results = None
                        st.session_state.off_test_emb     = None
                        st.success(f"Model saved: {save_path.name}  "
                                  f"({X_pooled.shape[0]} windows from {len(X_parts)} recordings)")
                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")

        off_mdl = st.session_state.off_model
        if off_mdl is not None:
            st.text(off_mdl.summary())
            st.plotly_chart(
                manifold_overlay_figure(off_mdl, label="enrollment"),
                width='stretch', config={"displayModeBar": False},
            )

    # ── Right column: Test ────────────────────────────────────────────────────
    with col_test:
        st.markdown("#### 2. Test a Session")

        off_mdl = st.session_state.off_model
        if off_mdl is None:
            st.info("Build a model first (left panel).", icon="ℹ️")
        else:
            st.caption(f"Model: {Path(st.session_state.off_model_path).name}")

            test_path      = _sweep_section("test")
            test_phase_sel = st.selectbox("Phase filter (test)", phase_opts,
                                          index=0, key="off_test_phase_sel",
                                          help="Use the same excitation coverage as "
                                               "enrollment for a fair comparison.")
            test_label     = st.text_input("Session label", value="test session",
                                           key="off_test_label")

            if st.button("Run Authentication Test", width='stretch',
                         disabled=not test_path):
                with st.spinner("Preprocessing…"):
                    df_t, fs_t = preprocess_csv(test_path.strip(),
                                                phase_filter=_resolve_phase(test_phase_sel))
                if df_t is None:
                    st.error("Cannot load test file.")
                else:
                    with st.spinner("Running pipeline…"):
                        try:
                            feat_t = extract_features(df_t, fs_t,
                                                      window_s=FEATURE_WIN_S,
                                                      hop_s=FEATURE_HOP_S)
                            X_t    = feature_matrix(feat_t)
                            dists, auth_flags = off_mdl.authenticate(X_t)
                            embs_t = off_mdl.transform(X_t)
                            win_times = (feat_t["window_start_s"].values
                                         if "window_start_s" in feat_t.columns
                                         else np.arange(len(dists)) * FEATURE_HOP_S)
                            st.session_state.off_test_results = {
                                "dists":    dists,
                                "auth":     auth_flags,
                                "win_times":win_times,
                                "pass_pct": float(auth_flags.mean()) * 100,
                                "label":    test_label,
                            }
                            st.session_state.off_test_emb = embs_t
                        except Exception as exc:
                            st.error(f"Pipeline error: {exc}")

            res = st.session_state.off_test_results
            if res is not None:
                pass_pct = res["pass_pct"]
                verdict  = "SAME DRONE" if pass_pct >= 80 else "IMPOSTOR / MODIFIED"
                r1, r2, r3 = st.columns(3)
                r1.metric("Pass rate", f"{pass_pct:.1f}%")
                r2.metric("Mean distance", f"{res['dists'].mean():.3f}",
                          delta=f"threshold: {off_mdl.threshold_:.3f}",
                          delta_color="off")
                r3.metric("Verdict", verdict)
                st.plotly_chart(
                    auth_timeseries_figure(res["dists"], res["auth"],
                                           off_mdl.threshold_, res["win_times"]),
                    width='stretch', config={"displayModeBar": False},
                )
                st.plotly_chart(
                    manifold_overlay_figure(
                        off_mdl,
                        test_emb  = st.session_state.off_test_emb,
                        test_auth = list(res["auth"]),
                        label     = res["label"],
                    ),
                    width='stretch', config={"displayModeBar": False},
                )
            else:
                st.info("Load a test CSV and click Run Authentication Test.", icon="ℹ️")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Motor Models
# ══════════════════════════════════════════════════════════════════════════════

with tab_models:

    st.markdown("### Motor Models — pick which calibration is active")
    st.caption(
        "The active model here is what pipeline.motor_model.pwm_to_rpm() "
        "(and everything built on it — Spectrogram's predicted lines, "
        "Motor Calibration's proposed peaks, the comb-filter preview) uses "
        "everywhere else in the dashboard. It's one shared choice, not "
        "per-tab — switch it here, then go look at the Spectrogram tab to "
        "see how a candidate model's predictions actually land on real "
        "data before committing to it."
    )

    active = get_active_model()
    st.success(f"**Active: {active.name}** — {active.description or '(no description)'}",
              icon="⚙️")

    st.markdown("##### Saved models")
    models = list_saved_motor_models()
    if not models:
        st.caption("No saved models yet.")
    else:
        for m in models:
            is_active = (m.name == active.name)
            with st.container(border=True):
                mc1, mc2 = st.columns([5, 1])
                with mc1:
                    st.markdown(f"**{m.name}**" + ("  🟢 *active*" if is_active else ""))
                    st.caption(m.description or "(no description)")
                    gains = "  ".join(f"M{k}={v:.3f}" for k, v in sorted(m.motor_gain.items()))
                    st.caption(
                        f"RPM = {m.rpm_slope:.5f}·(PWM·vbat) + {m.rpm_offset:.0f}   "
                        f"nominal vbat={m.nominal_vbat:.2f} V   gains: {gains}"
                    )
                    if m.source:
                        st.caption(f"Source: {m.source}"
                                  + (f"  ·  {m.created_iso}" if m.created_iso else ""))
                with mc2:
                    if st.button("Activate", key=f"model_activate_{m.name}",
                                disabled=is_active, width='stretch'):
                        set_active_motor_model(m)
                        st.rerun()
                    if st.button("🗑 Delete", key=f"model_delete_{m.name}",
                                disabled=is_active, width='stretch',
                                help="Delete this model (disabled while active)"):
                        delete_motor_model(m.name)
                        st.rerun()

    st.divider()
    with st.expander("➕ Create a new model manually"):
        st.caption(
            "Prefilled from the currently active model — tweak what you "
            "need and save under a new name. To create one FROM the "
            "Motor Calibration tab's vibration-based refit instead, use "
            "the '💾 Save as new model' button there once you've run a Refit."
        )
        nc1, nc2 = st.columns(2)
        new_name = nc1.text_input("Name (used as the filename)", key="newmodel_name")
        new_desc = nc2.text_input("Description", key="newmodel_desc")

        pc1, pc2, pc3 = st.columns(3)
        new_slope = pc1.number_input("RPM slope", value=active.rpm_slope,
                                     format="%.5f", key="newmodel_slope")
        new_offset = pc2.number_input("RPM offset", value=active.rpm_offset,
                                      step=1.0, key="newmodel_offset")
        new_vbat = pc3.number_input("Nominal vbat", value=active.nominal_vbat,
                                    step=0.1, key="newmodel_vbat")

        gc1, gc2, gc3, gc4 = st.columns(4)
        g1 = gc1.number_input("M1 gain", value=active.motor_gain.get(1, 1.0),
                              step=0.001, format="%.3f", key="newmodel_g1")
        g2 = gc2.number_input("M2 gain", value=active.motor_gain.get(2, 1.0),
                              step=0.001, format="%.3f", key="newmodel_g2")
        g3 = gc3.number_input("M3 gain", value=active.motor_gain.get(3, 1.0),
                              step=0.001, format="%.3f", key="newmodel_g3")
        g4 = gc4.number_input("M4 gain", value=active.motor_gain.get(4, 1.0),
                              step=0.001, format="%.3f", key="newmodel_g4")

        new_source = st.text_input("Source (optional note)", key="newmodel_source")

        existing_names = {m.name for m in models}
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", new_name.strip())
        if st.button("💾 Save model", disabled=not new_name.strip(),
                     key="newmodel_save_btn"):
            if safe_name in existing_names:
                st.error(f"A model named '{safe_name}' already exists — "
                        "pick a different name.")
            else:
                save_motor_model(MotorModel(
                    name=safe_name, description=new_desc.strip(),
                    rpm_slope=float(new_slope), rpm_offset=float(new_offset),
                    nominal_vbat=float(new_vbat),
                    motor_gain={1: float(g1), 2: float(g2), 3: float(g3), 4: float(g4)},
                    source=new_source.strip(),
                    created_iso=datetime.now().strftime("%Y-%m-%d"),
                ))
                st.success(f"Saved '{safe_name}'.")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Spectrogram (visual filter QA)
# ══════════════════════════════════════════════════════════════════════════════

with tab_spectro:

    st.markdown("### Spectrogram — visual filter QA")
    st.caption(
        "Load a recording — a 1 kHz SD-deck conversion from `usd_convert.py` "
        "works best (100 Hz telemetry is above Nyquist for the motor "
        "fundamental, see CLAUDE.md) — and compare raw vs filtered "
        "spectrograms side by side. Dashed lines are each motor's PREDICTED "
        "fundamental frequency (`pipeline.motor_model`, per-motor gain "
        "included): on the raw panel they should sit right on a real energy "
        "ridge — that validates the motor model's prediction against what "
        "was actually recorded; on the filtered panel that same ridge should "
        "be gone — that validates the comb filter notched the right "
        "frequency, per motor."
    )

    result = _recording_picker("spec")
    if result is not None:
        raw_df, source_label = result
        if raw_df.empty:
            st.error("File loaded but contains no rows.")
        else:
            st.session_state.spec_raw_df   = raw_df
            st.session_state.spec_last_csv = source_label
            st.session_state.spec_phases   = sorted(
                p for p in raw_df["phase"].dropna().unique().tolist()
            )
            # The multiselect below keeps its own state under this key once
            # rendered once — just changing `default=` on a later rerun does
            # NOT reselect it (a Streamlit widget with a key ignores
            # `default` whenever that key already has a stored value).
            # Overwriting the key directly here is what actually resets the
            # selection to "everything in this file" on every fresh load —
            # without it, phases new to this file (e.g. the all_hold_* run)
            # could stay silently unselected if an earlier load had picked a
            # narrower set. (found 2026-09-02: the all-four run was missing
            # from the spectrogram tab.)
            st.session_state.spec_phase_sel = st.session_state.spec_phases
            st.session_state.spec_df = None   # force rebuild on new load
            st.success(f"Loaded {len(raw_df):,} samples, "
                      f"{len(st.session_state.spec_phases)} phase(s).")

    spec_raw_df = st.session_state.get("spec_raw_df")
    if spec_raw_df is None:
        st.info("Load a recording to continue.", icon="ℹ️")
    else:
        # No default= here: the Load-recording handler above seeds
        # st.session_state.spec_phase_sel directly, and Streamlit warns (and
        # will eventually error) if a keyed widget gets both a default and a
        # pre-set session-state value for the same key.
        sel_phases = st.multiselect(
            "Phases to include", st.session_state.spec_phases,
            key="spec_phase_sel",
        )

        sp1, sp2, sp3 = st.columns(3)
        spec_channel = sp1.selectbox(
            "Channel", CH_OPTIONS, index=0, key="spec_channel",
        )
        spec_window_s = sp2.slider(
            "Spectrogram window (s)", 0.25, 2.0, 1.0, step=0.25,
            key="spec_window_s",
            help="FFT window length per time-slice — shorter gives finer "
                 "time resolution, longer gives finer frequency resolution.",
        )
        spec_show_harm = sp3.checkbox(
            "Show 2×f0 (blade-pass)", value=False, key="spec_show_harm",
        )

        if st.button("Build spectrogram", width='stretch',
                     disabled=not sel_phases, key="spec_build_btn"):
            with st.spinner("Filtering…"):
                df_f = spec_raw_df[spec_raw_df["phase"].isin(sel_phases)].reset_index(drop=True)
                if df_f.empty:
                    st.error("No samples in the selected phase(s).")
                else:
                    try:
                        fs_d = infer_fs(df_f)
                        df_f = highpass(df_f, fs_d)
                        if df_f[MOTOR_COLS].max().max() > MIN_MOTOR_PWM:
                            df_f = harmonic_deconvolution(df_f, fs_d)
                        st.session_state.spec_df = df_f
                        st.session_state.spec_fs = fs_d
                    except Exception as exc:
                        st.error(f"Preprocessing error: {exc}")

        spec_df = st.session_state.get("spec_df")
        spec_fs = st.session_state.get("spec_fs")
        if spec_df is not None and spec_fs is not None:
            dur = float(spec_df["timestamp_s"].iloc[-1] - spec_df["timestamp_s"].iloc[0])
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("Samples", f"{len(spec_df):,}")
            ic2.metric("Sampling rate", f"{spec_fs:.0f} Hz")
            ic3.metric("Duration", f"{dur:.1f} s")
            if "vbat_v" in spec_df.columns:
                ic4.metric("vbat range",
                          f"{spec_df['vbat_v'].min():.2f}–{spec_df['vbat_v'].max():.2f} V")

            st.markdown("##### A — Spectrogram (time-frequency overview)")
            st.caption(
                "Use this to see how the spectrum EVOLVES over the whole "
                "recording: a motor line sweeps diagonally as PWM ramps, "
                "while a genuine chassis resonance stays at a fixed height "
                "regardless of PWM — the classic Campbell-diagram check, at "
                "a glance, instead of comparing separate segment plots by "
                "hand. It also catches transient filter failures (a brief "
                "smear in the filtered panel where a fast PWM change "
                "outran the comb filter's block-wise re-estimate) that a "
                "single static spectrum below would miss. The white line "
                "marks where the snapshot below is taken from."
            )
            t_center = st.slider(
                "Snapshot instant (s)", 0.0, max(dur, 0.01),
                value=min(st.session_state.get("spec_t_center", dur / 2), dur),
                step=max(dur / 200, 0.01), key="spec_t_center",
            )
            st.plotly_chart(
                spectrogram_figure(spec_df, spec_fs, channel=spec_channel,
                                   window_s=spec_window_s,
                                   show_harmonics=spec_show_harm,
                                   mark_t=t_center),
                width='stretch', config={"displayModeBar": True},
            )

            st.divider()
            st.markdown("##### B — Spectrum snapshot + comb-filter preview")
            st.caption(
                "The spectrum at ONE instant (full frequency resolution, no "
                "time-axis trade-off), with a shaded band at every "
                "frequency the comb filter notches right now — one colour "
                "per active motor, width = that notch's real -3dB "
                "bandwidth (f/Q). This is the precise check: does the real "
                "peak actually fall inside the shaded band, or is the "
                "model's prediction off by more than the notch is wide? "
                "Pick an instant inside a steady PWM hold (e.g. an "
                "`all_hold_53000` phase) — the band positions use one mean "
                "PWM/vbat for the whole snapshot window."
            )
            sn1, sn2, sn3 = st.columns(3)
            snap_window_s = sn1.slider("Snapshot window (s)", 0.5, 5.0, 2.0,
                                       step=0.5, key="spec_snap_window_s",
                                       help="Longer = finer frequency resolution.")
            snap_q = sn2.slider("Notch Q", 5.0, 60.0, 30.0, step=1.0,
                                key="spec_snap_q",
                                help="Matches preprocessing.harmonic_deconvolution's "
                                     "Q parameter — higher Q = narrower notch band.")
            snap_harm = sn3.number_input("Harmonics shown", 1, 8, 5,
                                         key="spec_snap_harmonics")

            st.plotly_chart(
                spectrum_snapshot_figure(spec_df, spec_fs, channel=spec_channel,
                                         t_center=t_center, window_s=snap_window_s,
                                         n_harmonics=int(snap_harm), Q=snap_q),
                width='stretch', config={"displayModeBar": True},
            )
        else:
            st.info("Select phase(s) and click **Build spectrogram**.", icon="ℹ️")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Motor Calibration (human-in-the-loop)
# ══════════════════════════════════════════════════════════════════════════════

with tab_calib:

    st.markdown("### Motor Calibration — human-in-the-loop vibration ground truth")
    st.caption(
        "Laser-tachometer readings are hand-aimed and can vary point-to-point "
        "on the blade, which adds measurement noise the per-motor model "
        "inherits. This tool uses the IMU spectrum itself as ground truth "
        "instead — for each single-motor hold segment (record with "
        "`test_usd_motors.py --skip-all-hold`, then `usd_convert.py`), it "
        "proposes the spectral peak closest to the model's current "
        "prediction; you confirm it or pick/type the correct one. Confirmed "
        "(motor, PWM, vbat, frequency) points are saved incrementally and "
        "can be refit into an updated per-motor model at any time."
    )

    # ── A. Record a new single-motor sweep ───────────────────────────────────
    with st.expander("🎬 Record a new single-motor sweep", expanded=False):
        st.caption(
            f"Drives ONE motor through a {len(CALIB_DEFAULT_PWMS)}-point PWM "
            "ladder (8k-48k — capped below ~50k, where single-motor tilt "
            "presses the chassis into the rig's support wires and swamps "
            "the tone with contact noise instead of a clean spectral "
            "line), SD logging on for just that motor — one recording per "
            "motor (its own usd.logging start/stop, its own timeline file), "
            "not one long four-motor session. Shorter recordings mean a "
            "dropped connection or a dead battery partway through only "
            "costs that one motor's time, not the ones already safely on "
            "the card. Same protocol as `test_usd_motors.py --skip-all-hold "
            "--motor-order N ...`, just driven from here."
        )

        live_worker = st.session_state.get("worker")
        drone_ok = (live_worker is not None and live_worker.is_alive()
                   and live_worker.connected.is_set() and live_worker.cf is not None)
        if not drone_ok:
            st.warning("Connect to the drone in the sidebar first.", icon="⚠️")
        else:
            st.success("Drone connected — ready to record.", icon="🟢")

        recorded = st.session_state.setdefault("calib_recorded_motors", [])
        st.caption("Recorded this session: " + "  ".join(
            f"M{m} {'✅' if m in recorded else '⬜'}" for m in (1, 2, 3, 4)
        ))

        rec_motor = st.radio("Motor to record", [1, 2, 3, 4],
                             horizontal=True, key="calib_rec_motor",
                             format_func=lambda m: f"M{m}")

        rc1, rc2 = st.columns(2)
        pwms_str = rc1.text_input(
            "PWM ladder", value=" ".join(str(p) for p in CALIB_DEFAULT_PWMS),
            key="calib_rec_pwms", help="Space-separated PWM values.",
        )
        hold_s = rc2.number_input(
            "Hold per step (s)", min_value=3.0, max_value=30.0,
            value=CALIB_HOLD_S, step=1.0, key="calib_rec_hold",
        )
        try:
            rec_pwms = [int(v) for v in pwms_str.split()]
        except ValueError:
            rec_pwms = []
            st.error("PWM ladder must be space-separated integers.")

        if rec_pwms:
            est_s = sweep_duration_s(rec_pwms, hold_s, skip_all_hold=True, n_motors=1)
            st.caption(f"M{rec_motor} × {len(rec_pwms)} steps → "
                      f"~{est_s:.0f}s (~{est_s / 60:.1f} min)")

        if st.button(f"▶ Start SD Recording — M{rec_motor}", width='stretch',
                     disabled=(not drone_ok or not rec_pwms), key="calib_rec_btn"):
            err, tl_path = _run_calibration_sweep_blocking(
                live_worker, rec_pwms, hold_s, [rec_motor], DEFAULT_USD_DIR,
            )
            if err:
                st.error(f"Recording failed: {err}")
            else:
                st.success(f"M{rec_motor} recording complete. "
                          f"Timeline saved → `{tl_path}`")
                st.session_state["calib_last_timeline"] = str(tl_path)
                if rec_motor not in recorded:
                    recorded.append(rec_motor)
                st.markdown(
                    "**Next:** power off the drone, pull the microSD card, "
                    "copy the newest `logNN` file into `data/usd/`, then "
                    "pick it up in **Pick a recording** below — no separate "
                    "conversion step needed, loading it does that. "
                    "Re-insert the card before recording the next motor."
                )

    # ── B. Pick a recording to review ────────────────────────────────────────
    st.markdown("##### Pick a recording")

    def _finish_calib_load(raw_df: pd.DataFrame, source_label: str) -> None:
        segs = detect_hold_segments(raw_df)
        if not segs:
            st.error("No m{N}_hold_{pwm} single-motor segments found — "
                     "was this recorded with test_usd_motors.py?")
            return
        st.session_state.calib_raw_df   = raw_df
        st.session_state.calib_last_csv = source_label
        st.session_state.calib_fs       = infer_fs(raw_df)
        st.session_state.calib_segments = segs
        st.session_state.calib_idx      = 0
        if not st.session_state.calib_out_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.calib_out_path = str(
                DEFAULT_OUTPUT_DIR / "calibration" /
                f"vibration_calibration_{ts}.csv"
            )
        st.success(f"Loaded {len(raw_df):,} samples, "
                  f"{len(segs)} single-motor hold segment(s).")

    result = _recording_picker("calib")
    if result is not None:
        _finish_calib_load(*result)

    calib_raw_df = st.session_state.get("calib_raw_df")
    if calib_raw_df is None:
        st.info("Load a single-motor SD recording to continue.", icon="ℹ️")
    else:
        segs = st.session_state.calib_segments
        idx  = st.session_state.calib_idx
        cal_fs = st.session_state.calib_fs

        if st.session_state.calib_out_path:
            st.caption(f"Confirmed points save to "
                      f"`{Path(st.session_state.calib_out_path).name}` as you go.")

        if idx >= len(segs):
            st.success(f"All {len(segs)} segments reviewed.")
        else:
            s = segs[idx]
            seg_df = calib_raw_df[calib_raw_df["phase"] == s["phase"]]
            vbat = float(seg_df["vbat_v"].mean()) if "vbat_v" in seg_df.columns else None
            f0_pred = pwm_to_hz(s["pwm"], vbat, motor_id=s["motor"])
            ranked = rank_candidates(seg_df, cal_fs, f0_pred)

            st.progress(idx / len(segs), text=f"Segment {idx + 1}/{len(segs)}")
            vbat_txt = f"{vbat:.2f} V" if vbat is not None else "N/A (nominal used)"
            st.markdown(
                f"#### M{s['motor']} @ {s['pwm']} PWM  (vbat {vbat_txt}, "
                f"current model predicts {f0_pred:.1f} Hz, "
                f"gain={MOTOR_GAIN.get(s['motor'], 1.0):.3f})"
            )

            channel = st.selectbox(
                "Channel", CH_OPTIONS, index=0, key="calib_channel",
            )

            options = [f"#{i + 1}: {f:.1f} Hz  (prominence {p:.0f}dB, score {sc:.1f})"
                      for i, (f, p, sc) in enumerate(ranked)]
            options.append("✏️ Type exact value…")
            if not ranked:
                st.warning(f"No candidate peak found within ±{15:.0f}% of the "
                          "prediction — type the value manually or skip.")
            choice = st.radio("Which peak is the real motor line?", options,
                              index=0, key=f"calib_choice_{idx}")

            manual_freq = None
            chosen_idx  = None
            if choice == options[-1]:
                manual_freq = st.number_input(
                    "Frequency (Hz)", min_value=0.0, max_value=float(cal_fs / 2),
                    value=float(round(f0_pred, 1)), step=0.5,
                    key=f"calib_manual_{idx}",
                )
                confirmed_f0 = manual_freq
                method = "manual_value"
            else:
                chosen_idx = options.index(choice)
                confirmed_f0 = ranked[chosen_idx][0]
                method = "auto_top1" if chosen_idx == 0 else "manual_pick"

            st.plotly_chart(
                calibration_candidates_figure(seg_df, cal_fs, f0_pred, ranked,
                                              chosen_idx, channel=channel,
                                              manual_freq=manual_freq),
                width='stretch', config={"displayModeBar": True},
            )

            bc1, bc2, bc3 = st.columns(3)
            if bc1.button("⬅ Back", disabled=(idx == 0), key=f"calib_back_{idx}"):
                st.session_state.calib_idx -= 1
                st.rerun()
            if bc2.button("⏭ Skip (bad segment)", key=f"calib_skip_{idx}"):
                st.session_state.calib_idx += 1
                st.rerun()
            if bc3.button("✅ Confirm & Next", width='stretch', key=f"calib_confirm_{idx}"):
                row = {
                    "timestamp_iso":    datetime.now().isoformat(timespec="seconds"),
                    "source_csv":       Path(st.session_state.calib_last_csv).name,
                    "phase":            s["phase"],
                    "motor":            s["motor"],
                    "pwm":              s["pwm"],
                    "vbat_v":           round(vbat, 3) if vbat is not None else "",
                    "f0_confirmed_hz":  round(confirmed_f0, 2),
                    "rpm_confirmed":    round(confirmed_f0 * 60, 1),
                    "method":           method,
                }
                append_calibration_row(st.session_state.calib_out_path, row)
                st.session_state.calib_confirmed.append(row)
                st.session_state.calib_idx += 1
                st.rerun()

        # ── accumulated points + refit ──────────────────────────────────────
        st.divider()
        st.markdown("##### Confirmed points")

        if not st.session_state.calib_confirmed:
            prev_csvs = [f for f in list_saved_csvs(DEFAULT_OUTPUT_DIR / "calibration")
                        if f["name"].startswith("vibration_calibration_")]
            if prev_csvs:
                with st.expander("📂 Load points confirmed in an earlier session"):
                    st.caption(
                        "Confirmed points are saved to disk as you go (see path "
                        "above), but only held in memory for Refit/Save-as-model "
                        "within the session that confirmed them — reload one here "
                        "to pick that session back up without re-reviewing."
                    )
                    for f in prev_csvs:
                        pc1, pc2, pc3 = st.columns([3, 2, 1])
                        pc1.text(f["name"])
                        pc2.caption(f'{f["size_kb"]:.0f} KB · {f["mtime"]:%Y-%m-%d %H:%M}')
                        if pc3.button("Load", key=f"calib_load_prev_{f['name']}"):
                            try:
                                loaded = pd.read_csv(f["path"]).to_dict("records")
                                st.session_state.calib_confirmed = loaded
                                st.session_state.calib_out_path  = str(f["path"])
                                st.success(f"Loaded {len(loaded)} point(s) from "
                                          f"{f['name']} — further confirmations "
                                          "append to this same file.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Cannot load: {exc}")

        if st.session_state.calib_confirmed:
            conf_df = pd.DataFrame(st.session_state.calib_confirmed)
            st.dataframe(conf_df, width='stretch', height=200)

            if st.button("📐 Refit per-motor model from confirmed points",
                         key="calib_refit_btn"):
                results = {}
                for m in sorted(conf_df["motor"].unique()):
                    g = conf_df[(conf_df["motor"] == m) & (conf_df["vbat_v"] != "")]
                    if len(g) < 3:
                        results[int(m)] = {"n": len(g)}
                        continue
                    pwm_a  = g["pwm"].values.astype(float)
                    vbat_a = g["vbat_v"].values.astype(float)
                    rpm_a  = g["rpm_confirmed"].values.astype(float)
                    x = pwm_a * vbat_a

                    a, b = np.polyfit(x, rpm_a, 1)
                    rmse2 = float(np.sqrt(np.mean((rpm_a - (a * x + b)) ** 2)))

                    # gain-only vs the CURRENTLY ACTIVE model's bare base
                    # prediction (motor_id=None) — this is what makes the
                    # resulting k directly usable as a MOTOR_GAIN entry for
                    # a new model built on top of the same base.
                    base_pred = np.array([pwm_to_hz(p, v) * 60.0
                                          for p, v in zip(pwm_a, vbat_a)])
                    k = float(np.sum(base_pred * rpm_a) / np.sum(base_pred ** 2))
                    rmse_k = float(np.sqrt(np.mean((rpm_a - k * base_pred) ** 2)))

                    results[int(m)] = {"n": len(g), "a": a, "b": b, "rmse2": rmse2,
                                       "k": k, "rmse_k": rmse_k}
                st.session_state["calib_refit_results"] = results

            # Persisted (not just shown inline on the button's own render) so
            # the "Save as new model" section below can use it on later
            # reruns too, e.g. after the operator reviews the numbers first.
            refit = st.session_state.get("calib_refit_results")
            if refit:
                for m, r in sorted(refit.items()):
                    st.markdown(f"**M{m}** — {r['n']} confirmed point(s) with vbat")
                    if "k" not in r:
                        st.caption("Need at least 3 points (with vbat) for a fit.")
                        continue
                    st.text(f"  2-param fit:  RPM = {r['a']:.4f}*(PWM*vbat) + {r['b']:.0f}   "
                           f"RMSE={r['rmse2']:.0f} RPM")
                    st.text(f"  gain-only:    k={r['k']:.4f} (vs active model's base)   "
                           f"RMSE={r['rmse_k']:.0f} RPM")

                fittable = {m: r for m, r in refit.items() if "k" in r}
                if len(fittable) == 4:
                    with st.expander("💾 Save these gains as a new model"):
                        base = get_active_model()
                        st.caption(
                            f"Keeps the active model's base fit (slope="
                            f"{base.rpm_slope:.5f}, offset={base.rpm_offset:.0f}, "
                            f"nominal vbat={base.nominal_vbat:.2f}) and replaces "
                            "just the per-motor gains with the ones just computed."
                        )
                        sv_name = st.text_input("Name", key="calib_savemodel_name")
                        sv_desc = st.text_input(
                            "Description", key="calib_savemodel_desc",
                            value="Vibration-based per-motor gain (human-in-the-loop "
                                 "calibration tool).",
                        )
                        if st.button("Save", disabled=not sv_name.strip(),
                                     key="calib_savemodel_btn"):
                            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", sv_name.strip())
                            existing = {mm.name for mm in list_saved_motor_models()}
                            if safe_name in existing:
                                st.error(f"A model named '{safe_name}' already "
                                        "exists — pick a different name.")
                            else:
                                save_motor_model(MotorModel(
                                    name=safe_name, description=sv_desc.strip(),
                                    rpm_slope=base.rpm_slope, rpm_offset=base.rpm_offset,
                                    nominal_vbat=base.nominal_vbat,
                                    motor_gain={m: r["k"] for m, r in fittable.items()},
                                    source=f"Motor Calibration tab, {len(conf_df)} "
                                          "confirmed points, "
                                          f"{datetime.now():%Y-%m-%d}",
                                    created_iso=datetime.now().strftime("%Y-%m-%d"),
                                ))
                                st.success(f"Saved '{safe_name}' — activate it in "
                                          "the Motor Models tab.")
                                st.rerun()
                elif fittable:
                    st.caption(f"{len(fittable)}/4 motors have enough points to "
                              "fit — need all four to save as a model.")

            dl_col, clear_col = st.columns(2)
            dl_col.download_button(
                "⬇ Download confirmed CSV", data=conf_df.to_csv(index=False),
                file_name=(Path(st.session_state.calib_out_path).name
                          if st.session_state.calib_out_path else "calibration.csv"),
                key="calib_dl_btn",
            )
            if clear_col.button("🗑 Clear confirmed points (this session's list only — "
                                "the CSV on disk is untouched)", key="calib_clear_btn"):
                st.session_state.calib_confirmed = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Resonances
# ══════════════════════════════════════════════════════════════════════════════

with tab_resonances:

    st.markdown("### Chassis Resonances — the actual fingerprint signal")
    st.caption(
        "A structural resonance is a frequency the chassis 'prefers' to "
        "vibrate at — like a bell that always rings the same note no "
        "matter how you strike it. A motor line doesn't share that "
        "property: it moves when PWM changes. A frequency that keeps "
        "reappearing across many different PWM levels, motors, and "
        "recording sessions — after each segment's own motor line(s) have "
        "been removed — is evidence of a fixed structural mode, not a "
        "motor artifact. This is the actual identity signal the whole "
        "project is built around: it should be unique to this physical "
        "chassis, and should shift if the chassis is modified."
    )

    with st.expander("How the algorithm works", expanded=False):
        st.markdown(
            "**1. Remove every motor's own line, per segment.** For each "
            "recording segment (one motor spinning, or all four together), "
            "the active motor model (see the Motor Models tab) predicts "
            "where each spinning motor's tone should sit; that frequency "
            "and its harmonics are excluded from the segment's spectrum "
            "before anything else. For an all-four segment this means "
            "excluding **4** separate predicted lines, attributed via a "
            "small automatic matching between the 4 predictions and the "
            "detected peaks — not a human deciding which peak belongs to "
            "which motor (with 4 similarly-strong overlapping lines, a "
            "single spectrum doesn't contain enough information for that "
            "judgement call, automated or not — see the Motor Calibration "
            "tab).\n\n"
            "**2. Pool what's left.** Every peak that survives motor-line "
            "exclusion, in every segment, goes into one candidate pool.\n\n"
            "**3. Vote across segments.** A frequency is flagged as a "
            "resonance candidate if it reappears (within ±3 Hz) in a large "
            "enough fraction of ALL segments — different PWM levels, "
            "different motors, ideally different recording sessions. A "
            "motor line can't do this by construction (it moves with "
            "PWM); a genuine structural mode does, because it's a "
            "property of the physical chassis, not of any one motor or "
            "PWM setting.\n\n"
            "Known candidates from earlier sessions (103.4, 125.9, 162.9, "
            "333.7 Hz) are checked directly; anything else clearing the "
            "same bar is reported as newly discovered."
        )

    st.markdown("##### Select recording(s) to analyze")
    st.caption(
        "Pick more than one independent session (different days / battery "
        "levels) for the strongest evidence — a candidate confirmed across "
        "sessions is much better evidence than one confirmed only within a "
        "single file's own segments."
    )

    usd_1khz_files = [f for f in list_saved_csvs(DEFAULT_USD_DIR)
                      if f["name"].endswith("_1khz.csv")]
    if not usd_1khz_files:
        st.info(f"No converted recordings in `{DEFAULT_USD_DIR}` yet.", icon="ℹ️")
    else:
        sel_key = "reso_selected"
        st.session_state.setdefault(sel_key, [])
        selected = set(st.session_state[sel_key])
        for f in usd_1khz_files:
            checked = st.checkbox(
                f'{f["name"]}  ·  {f["size_kb"]:.0f} KB',
                value=(str(f["path"]) in selected),
                key=f"reso_pick_{f['name']}",
            )
            if checked:
                selected.add(str(f["path"]))
            else:
                selected.discard(str(f["path"]))
        st.session_state[sel_key] = sorted(selected)

        if st.button("🔍 Run analysis", disabled=not st.session_state[sel_key],
                     width='stretch', key="reso_run_btn"):
            with st.spinner(f"Analyzing {len(st.session_state[sel_key])} recording(s)…"):
                all_vote_segs: "list[dict]" = []
                per_file_info: "list[dict]" = []
                fs_list: "list[float]" = []
                for path_str in st.session_state[sel_key]:
                    try:
                        df_r = pd.read_csv(path_str, dtype={"phase": str})
                        dt_r = np.diff(df_r["timestamp_s"].values)
                        dt_r = dt_r[(dt_r > 0) & (dt_r < 0.1)]
                        fs_r = 1.0 / np.median(dt_r)
                        fs_list.append(fs_r)
                        single_segs, all_segs = collect_segments(df_r)
                        a_rows = attribute_single_drive(single_segs, fs_r)
                        motor_lines = single_drive_motor_lines(a_rows)
                        all_line_attribution = attribute_all_drive(all_segs, fs_r)
                        vs = build_vote_segs(single_segs, all_segs,
                                             motor_lines, all_line_attribution)
                        all_vote_segs.extend(vs)
                        per_file_info.append({
                            "name": Path(path_str).name,
                            "n_segs": len(vs),
                        })
                    except Exception as exc:
                        st.warning(f"Skipped {Path(path_str).name}: {exc}")

                if not all_vote_segs:
                    st.error("No usable m<id>_hold_<pwm> / all_hold_<pwm> segments "
                             "found in the selected recording(s).")
                else:
                    # All our SD-deck recordings use the same nominal 1 kHz
                    # config; per-file measured fs varies by under ~1.5% —
                    # far smaller than the ±3 Hz tolerance used for matching,
                    # so one representative fs for the combined pool is fine.
                    fs_combined = float(np.mean(fs_list))
                    result = find_resonance_candidates(all_vote_segs, fs_combined)
                    st.session_state["reso_result"]    = result
                    st.session_state["reso_file_info"] = per_file_info

    result = st.session_state.get("reso_result")
    if result is not None:
        st.divider()
        st.markdown("##### Results")

        info = st.session_state.get("reso_file_info", [])
        st.caption(
            f"{result['n_segs']} segments across {len(info)} recording(s): "
            + ", ".join(f"{i['name']} ({i['n_segs']} segs)" for i in info)
        )

        st.plotly_chart(
            resonance_bar_figure(result["known"], result["other"], result["n_segs"]),
            width='stretch', config={"displayModeBar": True},
        )

        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**Known candidates**")
            known_df = pd.DataFrame(result["known"])
            if not known_df.empty:
                known_df["confirmed"] = (known_df["hits"].astype(str) + "/"
                                         + known_df["n_segs"].astype(str))
                st.dataframe(
                    known_df[["freq", "confirmed", "reproduced"]]
                    .rename(columns={"freq": "frequency (Hz)"}),
                    width='stretch', hide_index=True,
                )
        with rc2:
            st.markdown("**Newly discovered**")
            other_df = pd.DataFrame(result["other"])
            if not other_df.empty:
                other_df["confirmed"] = (other_df["hits"].astype(str) + "/"
                                         + other_df["n_segs"].astype(str))
                st.dataframe(
                    other_df[["freq", "confirmed"]]
                    .rename(columns={"freq": "frequency (Hz)"}),
                    width='stretch', hide_index=True,
                )
            else:
                st.caption("(nothing beyond the known candidates)")
    else:
        st.info("Select recording(s) and click **Run analysis**.", icon="ℹ️")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — New Rig Session (micro-recording protocol, drone A/B)
# ══════════════════════════════════════════════════════════════════════════════
# One standard, fixed protocol — no per-session tuning (PWM, timings, which
# motors: all fixed in enrollment/micro_sweep.py, decided 2026-09-19). Each
# of the 15 stages (11 sweep + 4 stepped-hold, added 2026-09-20) is its own
# short SD recording; a lost connection costs only the stage in progress.
# The queue can be run in order or jumped around manually (added 2026-09-20,
# to resume a session with only new/missing stages). Raw SD files are
# matched back to what was recorded automatically
# (enrollment/session_manifest.py) — the operator
# only ever uses the Upload button, never hand-picks which timeline goes
# with which logNN file.

# Altered-drone sessions recorded before the GUI was translated (2026-10-29)
# store the description of the change in Italian, in their session.json;
# this maps it to the English choice, so the frozen widgets still show it.
LEGACY_CONDITION_TEXT = {
    "eliche nuove, stesso modello": "new propellers, same model",
    "eliche di un altro drone": "propellers from another drone",
    "eliche scambiate di posizione fra i motori": "propellers swapped between motors",
    "eliche di un modello diverso": "propellers of a different model",
    "elica danneggiata / tagliata": "damaged / trimmed propeller",
    "stesse eliche smontate e rimontate": "same propellers removed and remounted",
    "centro, sopra la scheda": "centre, above the board",
    "centro, sotto la scheda": "centre, below the board",
    "braccio M1": "arm M1", "braccio M2": "arm M2", "braccio M3": "arm M3", "braccio M4": "arm M4",
    "fuori centro (descrivi nelle note)": "off centre (describe in the notes)",
    "nastro biadesivo": "double-sided tape",
}

with tab_rig2:

    st.markdown("### New Rig Session — standard protocol")
    st.caption(
        f"Fixed protocol, fixed everything: {RIG2_MAX_PWM:,} PWM ceiling for "
        f"every stage, standard ramp/hold timings, {COOLDOWN_S:.0f}s cooldown "
        "between stages. One way to record, shared by both drones, so "
        "sessions stay comparable — nothing to tune here."
    )

    def _next_session_number(kind: str, drone: str) -> int:
        d = drone_dirs(rig_dir_for(kind), drone)["raw"].parent
        if not d.exists():
            return 1
        nums = [int(m.group(1)) for f in d.glob("session_*.json")
                if (m := re.match(r"session_(\d+)\.json", f.name))]
        return (max(nums) + 1) if nums else 1

    # Automatic mode state (see "Automatic recording" below). While it is
    # active the drone/kind/session selectors are frozen — the run is bound
    # to the session it was started on.
    auto = st.session_state.setdefault("rig2_auto", {"active": False})
    auto_on = bool(auto.get("active"))

    KIND_LABELS = {
        "Enrollment (builds the identity)": "enroll",
        "Test (validation, held-out data)": "test",
        "Motor swap A↔B (experiment)": "swap",
        "Board: IMU at rest (board fingerprint)": "board",
        "Boot: key-generation protocol": "boot",
        "Boot with changed propellers (altered drone)": "props",
        "Boot with added payload (altered drone)": "payload",
    }
    SWAP_STAGE_LABELS = {
        "pre": "1 · Before the swap (original motors)",
        "swap": "2 · Motors swapped",
        "post": "3 · After (motors back in place)",
    }

    def _next_swap_session_number(kind: str = "swap") -> int:
        d = rig_dir_for(kind)
        nums = [int(m.group(1)) for f in d.glob("s*_*") if f.is_dir()
                and (m := re.match(r"s(\d+)_", f.name))] if d.exists() else []
        return (max(nums) + 1) if nums else 1

    rc0, rc1, rc2 = st.columns([2, 1, 1])
    kind_label = rc0.radio(
        "Session type", list(KIND_LABELS),
        key="rig2_kind", disabled=auto_on,
        help="Test, motor-swap, board, propeller and payload sessions go to separate folder "
             "trees (data/fixed_rig_test/, _swap/, _board/, _props/, _payload/), so they can "
             "never end up by mistake in the data used to build the identity.",
    )
    kind = KIND_LABELS[kind_label]
    swap_motors = swap_stage = None

    if kind in BOOT_LIKE_KINDS:
        # Board / boot / altered-drone sessions: one session = one BOOT of the drone (the
        # firmware measures the board fingerprint at boot). Numbered globally
        # per kind, one folder per session; an existing number reopens it.
        st.session_state.setdefault(f"rig2_session_{kind}", _next_swap_session_number(kind))
        session_n = rc2.number_input(
            "Session", min_value=1, step=1, key=f"rig2_session_{kind}", disabled=auto_on,
            help="One numbering for all sessions of this type (1, 2, 3 ... in recording "
                 "order). One session = one boot of the drone.",
        )
        # The drone is locked only once the session has a recording: a folder
        # with an attempt-less session.json (left by an older version that
        # saved on opening) does not lock anything.
        existing = find_swap_session(session_n, rig_dir_for(kind))
        m_existing = (SessionManifest.load(existing / "session.json")
                      if existing is not None and (existing / "session.json").exists() else None)
        if m_existing is not None and m_existing.all_attempts():
            drone = m_existing.drone
            rc1.markdown(f"Drone **{drone}**")
            sess_dir = existing
        else:
            drone = rc1.selectbox("Drone", ["A", "B", "C", "D"], key=f"rig2_{kind}_drone", disabled=auto_on)
            sess_dir = rig_dir_for(kind) / f"s{session_n:02d}_drone{drone}"
        dirs  = swap_session_dirs(sess_dir)
        mpath = sess_dir / "session.json"
        what = ("A single phase: 30 s with the motors off." if kind == "board" else
                f"A single recording (about {micro_duration_s('boot-seq'):.0f} s): "
                f"{BOOT_REST_S:.0f} s with the motors off, then each motor alone at 58k PWM for "
                "a few seconds. It is the protocol the drone would run at every power-up to "
                "regenerate its key.")
        if kind in CONDITION_KINDS:
            what += (" **The drone was modified after enrollment**: describe below what you "
                     "changed before starting. The normal boot sessions of this drone are the "
                     "reference: record a few on the same day, before the change.")
        title = {"board": "🔬 **Board session**", "boot": "🔑 **Boot session**",
                 "props": "🌀 **Boot with changed propellers**",
                 "payload": "📦 **Boot with payload**"}[kind]
        st.info(
            f"{title} — drone **{drone}**. **Power-cycle the drone before every session** "
            "and keep it still in the rig while it boots (the firmware measures self-test, bias "
            f"and scale right at power-up), then connect it. {what} All data go to `{sess_dir.as_posix()}/`."
        )
    elif kind != "swap":
        drone = rc1.selectbox("Drone", ["A", "B", "C", "D"], key="rig2_drone", disabled=auto_on)
        default_session_key = f"rig2_session_{kind}_{drone}"
        st.session_state.setdefault(default_session_key, _next_session_number(kind, drone))
        session_n = rc2.number_input(
            "Session", min_value=1, step=1, key=default_session_key, disabled=auto_on,
            help="Numbered per drone and per type: test session 1 of A is independent "
                 "of enrollment session 1 of A and of test session 1 of B.",
        )
        if kind == "test":
            st.info("**TEST** session: data saved in `data/fixed_rig_test/`, never to be used "
                    "to build the identity, only to validate it.", icon="🧪")
        rig_dir = rig_dir_for(kind)
        dirs  = drone_dirs(rig_dir, drone)
        mpath = manifest_path(rig_dir, drone, session_n)
    else:
        # Swap sessions: numbered globally (1, 2, 3 … in recording order), one
        # folder per session. A session number that already exists reopens
        # that session with the configuration it was created with.
        st.session_state.setdefault("rig2_session_swap", _next_swap_session_number())
        session_n = rc2.number_input(
            "Session", min_value=1, step=1, key="rig2_session_swap", disabled=auto_on,
            help="One numbering for the whole experiment (1, 2, 3 … in recording order). "
                 "An existing number reopens that session.",
        )
        existing = find_swap_session(session_n)
        if existing is not None:
            m_old = SessionManifest.load(existing / "session.json")
            drone, swap_motors, swap_stage = m_old.drone, m_old.motors, m_old.swap_stage
            rc1.markdown(f"**{SWAP_STAGE_LABELS[swap_stage]}**  \nframe **{drone}**, motors **{swap_motors}**")
        else:
            swap_stage = rc1.selectbox("Stage of the experiment", list(SWAP_STAGE_LABELS),
                                       format_func=SWAP_STAGE_LABELS.get,
                                       key="rig2_swap_stage", disabled=auto_on)
            drone = rc1.selectbox("Frame (the connected drone)", ["A", "B"],
                                  key="rig2_swap_frame", disabled=auto_on)
            swap_motors = ({"A": "B", "B": "A"}[drone] if swap_stage == "swap" else drone)
        sess_dir = swap_session_dir(session_n, swap_stage, drone, swap_motors)
        dirs  = swap_session_dirs(sess_dir)
        mpath = sess_dir / "session.json"
        st.info(
            f"🔧 **Motor swap** — connect the drone with **frame {drone}**; motors mounted: "
            f"**{swap_motors}** ({'its own' if swap_motors == drone else 'from the other drone'}). "
            f"Reduced protocol: only the 4 single-motor stages. All data of this session "
            f"(raw, timelines, converted, analysis) go to `{sess_dir.as_posix()}/`."
        )

    manifest = SessionManifest.load_or_new(mpath, drone, session_n, kind,
                                           motors=swap_motors, swap_stage=swap_stage)

    # ── altered-drone sessions: what was changed (required before recording,
    # frozen once the first attempt exists, so the data and its description
    # can never disagree) ──────────────────────────────────────────────────
    if kind in CONDITION_KINDS:
        st.markdown("#### Change to the drone")
        frozen = bool(manifest.all_attempts())
        cond = dict(manifest.condition or {})
        ck = f"rig2_cond_{kind}_{session_n}"
        if kind == "props":
            PROP_CHANGES = ["new propellers, same model", "propellers from another drone",
                            "propellers swapped between motors", "propellers of a different model",
                            "damaged / trimmed propeller", "same propellers removed and remounted"]
            cur = LEGACY_CONDITION_TEXT.get(cond.get("change"), cond.get("change"))
            cc1, cc2 = st.columns(2)
            change = cc1.selectbox("What changed", PROP_CHANGES, disabled=frozen, key=f"{ck}_change",
                                   index=PROP_CHANGES.index(cur) if cur in PROP_CHANGES else 0)
            mots = cc2.multiselect("On which motors", [1, 2, 3, 4], default=cond.get("motors", [1, 2, 3, 4]),
                                   disabled=frozen, key=f"{ck}_motors")
            new = dict(change=change, motors=sorted(mots))
        else:
            POSITIONS = ["centre, above the board", "centre, below the board", "arm M1", "arm M2",
                         "arm M3", "arm M4", "off centre (describe in the notes)"]
            cur = LEGACY_CONDITION_TEXT.get(cond.get("position"), cond.get("position"))
            cc1, cc2, cc3 = st.columns(3)
            mass = cc1.number_input("Payload mass [g]", min_value=0.0, step=0.5, disabled=frozen,
                                    value=float(cond.get("mass_g", 5.0)), key=f"{ck}_mass",
                                    help="The Crazyflie weighs about 27 g; its maximum payload is ~15 g.")
            pos = cc2.selectbox("Position", POSITIONS, disabled=frozen, key=f"{ck}_pos",
                                index=POSITIONS.index(cur) if cur in POSITIONS else 0)
            mount = cc3.text_input("Mounting", value=LEGACY_CONDITION_TEXT.get(cond.get("mount"), cond.get("mount", "double-sided tape")),
                                   disabled=frozen, key=f"{ck}_mount")
            new = dict(mass_g=mass, position=pos, mount=mount)
        new["notes"] = st.text_input("Notes (optional)", value=cond.get("notes", ""),
                                     disabled=frozen, key=f"{ck}_notes")
        if frozen:
            st.caption("Description locked: the session already has a recording.")
        else:
            # Kept in memory only: written to session.json together with the
            # first attempt, so opening the page never creates the session.
            manifest.condition = new
            st.caption("Editable until the first recording; it is saved when you start.")

    live_worker = st.session_state.get("worker")
    drone_ok = (live_worker is not None
                and live_worker.is_alive()
                and live_worker.connected.is_set()
                and live_worker.motors is not None)
    if not drone_ok:
        st.warning("Connect to the drone in the sidebar first.", icon="⚠️")
    else:
        st.success("Drone connected — ready to record.", icon="🟢")

    # ── blocking runner for one micro-recording ─────────────────────────────

    def _run_micro_blocking(worker, purpose: str, cooldown: bool = True,
                            vbat_rest: "float | None" = None) -> "str | None":
        if worker is None or worker.cf is None or worker.motors is None:
            return "Drone not connected"
        cf = worker.cf

        def get_param(name: str) -> "str | None":
            try:
                return cf.param.get_value(name, timeout=3)
            except Exception:
                return None

        # Safety net: a previous attempt may have died mid-recording without
        # ever reaching its own usd.logging=0 (link was already gone by
        # then) — force it off BEFORE starting anything new, every time,
        # rather than trying to special-case "this is a retry after a drop".
        try:
            if str(get_param("usd.logging")) in ("1", "1.0"):
                cf.param.set_value("usd.logging", "0")
                time.sleep(0.3)
        except Exception:
            pass

        if purpose not in purposes_for(kind):
            return f"{purpose!r} is not part of sessions of type {kind!r}."
        if str(get_param("deck.bcUSD")) not in ("1", "1.0"):
            return "SD deck not detected."
        if str(get_param("usd.canLog")) not in ("1", "1.0"):
            return ("SD deck present but not ready (card FAT32? config.txt? "
                    "card inserted before power-on?).")

        # Battery gate: only start with a well-charged battery at rest (weak
        # stages in the four-drone study partly coincide with low starts).
        # A failed read does not block — a telemetry hiccup is not a reason
        # to refuse a stage.
        if vbat_rest is None:
            vbat_rest = read_vbat(cf)
        if vbat_rest is not None and vbat_rest < VBAT_MIN_REST_V:
            return (f"Battery at rest {vbat_rest:.2f} V, below the {VBAT_MIN_REST_V:.2f} V "
                    "threshold: let it recharge a few minutes and try again.")

        # Board sessions: read what the firmware measured at THIS boot, once per
        # session, before the rest stage. Same values as the previous board
        # session of this drone = the drone was not power-cycled in between.
        if kind in CONDITION_KINDS and not manifest.condition:
            return "Describe the change made to the drone first (section “Change to the drone”)."
        if kind in BOOT_LIKE_KINDS and not manifest.all_attempts():
            # Same session number left as an empty folder for another drone
            # (only a session.json without attempts): remove it, otherwise two
            # folders would share the number.
            for d in rig_dir_for(kind).glob(f"s{session_n:02d}_*"):
                if d == Path(mpath).parent or not (d / "session.json").exists():
                    continue
                if (not SessionManifest.load(d / "session.json").all_attempts()
                        and [p.name for p in d.iterdir()] == ["session.json"]):
                    (d / "session.json").unlink()
                    d.rmdir()
        if kind in BOOT_LIKE_KINDS and not manifest.board_params:
            bp = read_board_params(cf)
            if bp is not None:
                for f in sorted(rig_dir_for(kind).glob("s*_*/session.json")):
                    if f.parent == Path(mpath).parent:
                        continue
                    other = SessionManifest.load(f)
                    op = other.board_params or {}
                    if other.drone == drone and op.get("firmware") and all(
                            op.get(k) == bp[k] for k in ("gBiasX", "gBiasY", "gBiasZ", "accScale")):
                        return (f"Boot parameters identical to session {other.session}: the drone "
                                "was not power-cycled. Switch it off and on, then try again.")
            manifest.board_params = dict(
                firmware=bp is not None, **(bp or {}),
                board_temp_c=read_log_float(cf, "baro.temp"),
                vbat_rest=vbat_rest,
                read_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            manifest.save(mpath)

        dirs["timelines"].mkdir(parents=True, exist_ok=True)
        stem = micro_stem(drone, session_n, purpose, kind=kind,
                          motors=manifest.motors, stage=manifest.swap_stage)
        tl_path = unique_path(dirs["timelines"] / f"{stem}.csv")
        attempt = manifest.start_attempt(purpose, RIG2_MAX_PWM, tl_path)
        attempt.vbat_rest = vbat_rest
        manifest.save(mpath)   # on record before running — even a connect
                                # failure right after this is now tracked.

        tl = Timeline(tl_path)
        if vbat_rest is not None:
            tl.log("vbat_rest", f"{vbat_rest:.3f}")
        phase_recorder = UsdPhaseRecorder(tl)
        progress_ph = st.empty()
        motor_ph    = st.empty()

        def _progress_cb(phase, elapsed_s, total_s, remaining_s, motor_pwm):
            frac = 0.0 if total_s <= 0 else min(1.0, elapsed_s / total_s)
            progress_ph.progress(
                frac, text=f"{purpose} — {phase} — {remaining_s:0.0f}s remaining "
                          f"({elapsed_s:0.0f}s / {total_s:0.0f}s)",
            )
            cols = motor_ph.columns(4)
            for i, mid in enumerate((1, 2, 3, 4)):
                pwm = motor_pwm.get(mid, 0)
                rpm = pwm_to_hz(pwm, motor_id=mid) * 60.0
                cols[i].metric(f"M{mid}", f"{pwm} PWM",
                               f"{rpm:,.0f} RPM" if pwm else "off", delta_color="off")
            tl.log("pwm", max(motor_pwm.values()))

        error = None
        try:
            cf.param.set_value("usd.logging", "1")
            tl.log("usd_start")
            run_micro(purpose, worker.motors, phase_recorder, progress_cb=_progress_cb)
            # A dropped link doesn't make the motor/param calls above raise —
            # cflib just stops sending — so check explicitly, otherwise a
            # recording cut short by a disconnect would be marked completed.
            if not (worker.is_alive() and worker.connected.is_set()):
                raise RuntimeError(worker.error or "Connection lost during the recording")
            cf.param.set_value("usd.logging", "0")
            tl.log("usd_stop")
            attempt.status = "completed"
        except Exception as exc:
            error = str(exc)
            attempt.status = "interrupted"
            try:
                worker.motors.stop_all()
            except Exception:
                pass
            try:
                cf.param.set_value("usd.logging", "0")
                tl.log("usd_stop")
            except Exception:
                pass  # link may already be dead — next attempt's safety net handles it
        except BaseException:
            # Streamlit's rerun/stop (a click anywhere on the page mid-recording)
            # is a BaseException: without this the motors would keep spinning
            # and the SD would keep logging after the script run is torn down.
            attempt.status = "interrupted"
            try:
                worker.motors.stop_all()
            except Exception:
                pass
            try:
                cf.param.set_value("usd.logging", "0")
                tl.log("usd_stop")
            except Exception:
                pass
            raise
        finally:
            # save BEFORE touching st.* again: on a Streamlit rerun/stop the
            # next st call can raise once more and skip everything after it
            # (how drone C s01 sweep-rear_34 was left stuck as "recording").
            tl.close()
            manifest.save(mpath)
            try:
                progress_ph.empty()
                motor_ph.empty()
            except Exception:
                pass

        if error is None and cooldown:
            with st.spinner(f"Cooling down ({COOLDOWN_S:.0f}s)…"):
                time.sleep(COOLDOWN_S)

        return error

    # ── queue display + run/retry controls ───────────────────────────────

    # Outcome of the last manual run. Each run button calls st.rerun() right
    # after the stage, which wiped an st.error()/st.success() shown before it —
    # refusals (battery gate, not power-cycled, no SD deck) were never seen.
    # The message is kept in session_state and shown on the next run.
    def _flash(level: str, text: str) -> None:
        st.session_state["rig2_flash"] = (level, text)

    _fl = st.session_state.pop("rig2_flash", None)
    if _fl:
        (st.error if _fl[0] == "error" else st.success)(_fl[1])

    st.markdown("#### 1. Boot and motors-off phase (board)" if kind == "board" else
                "#### 1. Boot protocol (key)" if kind in ("boot", "props", "payload") else
                "#### 1. Reduced protocol queue (motor swap)" if kind == "swap"
                else "#### 1. Standard protocol queue")
    todo_first = None
    PURPOSES = purposes_for(kind)   # 15 standard; +unseen-pwm in test; 4 single-motor in swap
    for purpose in PURPOSES:
        rec = next((r for r in manifest.recordings if r.purpose == purpose), None)
        c1, c2, c3 = st.columns([2, 3, 1])
        c1.text(purpose)
        action = None
        if rec is None or rec.needs_recording:
            c2.caption("to do" if rec is None else "to do (previous recording discarded)")
            if todo_first is None:
                todo_first = purpose
        else:
            accepted = rec.accepted_attempt
            if accepted is not None:
                c2.success(f"OK — {Path(accepted.converted_path).name}")
            else:
                latest = rec.attempts[-1]
                if latest.status == "lost":
                    c2.error("data lost — record again")
                    action = "retry"
                elif latest.raw_log_path is None:
                    c2.info(f"recorded ({latest.status}), waiting for upload")
                    action = "discard"
                else:
                    c2.warning("uploaded but not valid — try again")
                    action = "retry"
        if action == "discard":
            with c3.popover("🗑", disabled=auto_on, help="Discard this recording and put the stage back in the queue"):
                st.markdown(
                    f"Discard the recording of **{purpose}** "
                    f"({latest.status}, {latest.created_iso.replace('T', ' ')})?\n\n"
                    "The stage goes back to **to do** and you record it again with the buttons "
                    "below. Nothing is deleted from disk. **Do not delete its `logNN` from the "
                    "SD card**: upload all files as usual — the discarded one is recognised by "
                    "its position, stored in `raw/discarded/` and not converted, so the later "
                    "files stay matched to the right stages."
                )
                if st.button("Yes, discard", type="primary",
                             key=f"rig2_discard_{kind}_{drone}_{session_n}_{latest.attempt_id}"):
                    manifest.discard_attempt(latest.attempt_id)
                    manifest.save(mpath)
                    st.rerun()
        if action == "retry" and c3.button("↻", key=f"rig2_retry_{kind}_{drone}_{session_n}_{purpose}",
                                           disabled=not drone_ok or auto_on,
                                           help="Record this stage again"):
            err = _run_micro_blocking(live_worker, purpose)
            if err:
                _flash("error", f"{purpose} failed: {err}")
            else:
                _flash("success", f"{purpose} completed.")
            st.rerun()

    bc1, bc2 = st.columns(2)
    if bc1.button("▶ Run next in queue", width='stretch',
                  disabled=not (drone_ok and todo_first) or auto_on):
        err = _run_micro_blocking(live_worker, todo_first)
        if err:
            _flash("error", f"{todo_first} failed: {err}")
        else:
            _flash("success", f"{todo_first} completed.")
        st.rerun()
    if bc2.button("🤖 Automatic recording", width='stretch',
                  disabled=not (drone_ok and todo_first) or auto_on,
                  help=f"Runs this stage and every later one still to do on its own, with a "
                       f"{AUTO_PAUSE_S:.0f}s battery pause between stages "
                       f"({AUTO_PAUSE_AFTER_ALL4_S:.0f}s from sweep-all4 on). It stops at the end "
                       "of the queue or at the first error/disconnection, with a sound."):
        st.session_state["rig2_auto"] = {
            "active": True, "kind": kind, "drone": drone, "session": int(session_n),
            "resume_at": time.time(), "current": None, "done": [],
        }
        st.session_state.pop("rig2_auto_msg", None)
        st.rerun()

    # ── automatic mode: one stage per script run ──────────────────────────
    # Each run: (pause until resume_at) → record todo_first → set the next
    # pause → st.rerun(). Keeping one stage per run means the page (queue,
    # statuses) refreshes between stages, and any stop condition is simply
    # "don't rerun". Only the Stop button should be touched while it runs:
    # any other click interrupts the script (motors are stopped safely by
    # _run_micro_blocking) and is treated as an error stop.

    def _auto_stop(ok: bool, text: str) -> None:
        st.session_state["rig2_auto"] = {"active": False}
        st.session_state["rig2_auto_msg"] = ("success" if ok else "error", text)
        alert(ok)
        st.rerun()

    msg = st.session_state.get("rig2_auto_msg")
    if msg and not auto_on:
        (st.success if msg[0] == "success" else st.error)(msg[1], icon="🔔")

    if auto_on:
        st.info(
            f"🤖 **Automatic recording running** — drone {auto['drone']}, "
            f"{'test' if auto['kind'] == 'test' else 'enrollment'} session {auto['session']}. "
            "Do not touch the page (except the Stop button): any other click interrupts "
            "the stage in progress."
        )
        if st.button("⏹ Stop automatic recording", type="primary", width='stretch'):
            interrupted = auto.get("current")
            st.session_state["rig2_auto"] = {"active": False}
            st.session_state["rig2_auto_msg"] = (
                "error",
                "Automatic recording stopped by the user"
                + (f" during {interrupted} (motors stopped; the stage is marked "
                   "interrupted — record it again from the manual selector after the upload)."
                   if interrupted else "."),
            )
            st.rerun()

        if (auto.get("kind"), auto.get("drone"), auto.get("session")) != (kind, drone, int(session_n)):
            _auto_stop(False, "Automatic recording stopped: the selected session changed.")
        if auto.get("current"):
            _auto_stop(False, f"Automatic recording stopped: {auto['current']} interrupted "
                              "(page reloaded or a click during the recording). "
                              "Motors stopped, stage marked interrupted.")
        if todo_first is None:
            done = auto.get("done", [])
            _auto_stop(True, f"Automatic recording completed ({len(done)} stages recorded). "
                             "Now upload the logNN files from the SD card below.")

        def _connected() -> bool:
            w = st.session_state.get("worker")
            return (w is not None and w.is_alive() and w.connected.is_set()
                    and w.motors is not None)

        if not _connected():
            _auto_stop(False, "Automatic recording stopped: drone disconnected "
                              f"before {todo_first}.")

        wait_ph = st.empty()
        while (left := auto["resume_at"] - time.time()) > 0:
            if not _connected():
                wait_ph.empty()
                _auto_stop(False, "Automatic recording stopped: drone disconnected "
                                  f"during the pause before {todo_first}.")
            last = auto.get("done", [])[-1:] or ["—"]
            wait_ph.info(f"⏳ Battery pause after {last[0]}: next stage **{todo_first}** "
                         f"in {left:0.0f}s")
            time.sleep(min(1.0, left))
        wait_ph.empty()

        # Battery gate: after the fixed pause, keep waiting while the resting
        # voltage is below threshold (the drone charges over USB meanwhile),
        # re-checking every VBAT_RECHECK_S, up to VBAT_WAIT_MAX_S extra.
        gate_t0 = time.time()
        while True:
            if not _connected():
                wait_ph.empty()
                _auto_stop(False, "Automatic recording stopped: drone disconnected "
                                  f"while waiting for the battery before {todo_first}.")
            vbat_now = read_vbat(live_worker.cf)
            if vbat_now is None or vbat_now >= VBAT_MIN_REST_V:
                break
            waited = time.time() - gate_t0
            if waited > VBAT_WAIT_MAX_S:
                wait_ph.empty()
                _auto_stop(False, f"Automatic recording stopped: the battery stays at "
                                  f"{vbat_now:.2f} V (threshold {VBAT_MIN_REST_V:.2f} V) after "
                                  f"{VBAT_WAIT_MAX_S / 60:.0f} minutes of waiting before {todo_first}. "
                                  "Check the USB cable / the battery.")
            for _ in range(int(VBAT_RECHECK_S)):
                wait_ph.warning(f"🔋 Battery at rest {vbat_now:.2f} V, below {VBAT_MIN_REST_V:.2f} V: "
                                f"waiting for it to recharge before **{todo_first}** "
                                f"(waiting for {time.time() - gate_t0:0.0f}s, max {VBAT_WAIT_MAX_S / 60:.0f} min)")
                time.sleep(1.0)
        wait_ph.empty()

        auto["current"] = todo_first
        err = _run_micro_blocking(live_worker, todo_first, cooldown=False, vbat_rest=vbat_now)
        auto["current"] = None
        if err:
            _auto_stop(False, f"Automatic recording stopped: {todo_first} failed — {err}")
        auto.setdefault("done", []).append(todo_first)
        auto["resume_at"] = time.time() + auto_pause_after(todo_first, PURPOSES)
        st.rerun()

    def _purpose_status_tag(purpose: str) -> str:
        rec = next((r for r in manifest.recordings if r.purpose == purpose), None)
        if rec is None or rec.needs_recording:
            return "to do"
        if rec.accepted_attempt is not None:
            return "OK"
        latest = rec.attempts[-1]
        if latest.status == "lost":
            return "data lost, record again"
        return "try again" if latest.raw_log_path else "waiting for upload"

    with st.expander("Or choose the stage to run by hand "
                     "(resume a session from a given point, redo a stage already done, "
                     "jump around)"):
        manual_purpose = st.selectbox(
            "Stage", PURPOSES,
            index=PURPOSES.index(todo_first) if todo_first else 0,
            format_func=lambda p: f"{p}  —  {_purpose_status_tag(p)}",
            key=f"rig2_manual_purpose_{kind}_{drone}_{session_n}",
        )
        if st.button("▶ Run the selected stage", width='stretch',
                     disabled=not drone_ok or auto_on,
                     key=f"rig2_manual_run_{kind}_{drone}_{session_n}"):
            err = _run_micro_blocking(live_worker, manual_purpose)
            if err:
                _flash("error", f"{manual_purpose} failed: {err}")
            else:
                _flash("success", f"{manual_purpose} completed.")
            st.rerun()

    if todo_first is None and all(
        (r := next((r for r in manifest.recordings if r.purpose == p), None))
        and r.accepted_attempt for p in PURPOSES
    ):
        st.success(f"Session complete — all {len(PURPOSES)} stages have a valid recording.")

    # ── batch upload + auto-match ─────────────────────────────────────────

    st.markdown("#### 2. Upload the SD recordings (batch)")
    st.caption(
        "Upload all the `logNN` files copied from the card at once — they are "
        "matched to the recorded attempts automatically, in order. Do not rename "
        "the files before uploading (the original name is needed to sort them)."
    )
    ups = st.file_uploader(
        "logNN files", accept_multiple_files=True,
        key=f"rig2_upload_{kind}_{drone}_{session_n}",
    )
    if ups and st.button("Upload and match", key=f"rig2_match_btn_{kind}_{drone}_{session_n}",
                         disabled=auto_on):
        results, leftover = ingest_upload_batch(
            manifest, mpath, [(up.name, up.getvalue()) for up in ups],
            dirs, convert_usd_log,
        )
        n_ok = sum(1 for _, ok, _ in results if ok)
        n_fail = len(results) - n_ok
        if n_ok:
            st.success(f"{n_ok} file(s) converted and matched.")
        if n_fail:
            st.error(f"{n_fail} file(s) failed to convert (marked not valid, raw file kept).")
        if leftover:
            st.warning(
                f"{len(leftover)} file(s) more than the recorded attempts (probably left "
                "by an interrupted and repeated attempt): saved in `raw/unmatched/`, not "
                "matched — check them by hand."
            )
        with st.expander("Conversion details"):
            for name, ok, msg in results:
                st.text(f"{name}: {'OK' if ok else 'FAILED'} — {msg}")
        st.rerun()

    still_pending = manifest.pending_upload_attempts()
    if still_pending:
        st.caption(f"{len(still_pending)} recording(s) still waiting for their SD file.")

    # ── board sessions: what the firmware measured at this boot ──────────
    if kind in BOOT_LIKE_KINDS:
        st.markdown("#### 3. Board parameters read at this boot")
        bp = manifest.board_params
        if not bp:
            st.caption("They are read when the motors-off phase starts.")
        elif not bp.get("firmware"):
            st.warning(
                "This drone does not have the firmware with the `uavibe` parameters (self-test, "
                "bias, scale): only the noise with the motors off was recorded. Flash the updated "
                "firmware to get the full board fingerprint.", icon="⚠️")
        else:
            px = lambda k: bp.get(k)
            rows = [
                ("Accelerometer self-test, + (x, y, z) [LSB, 24 g]", f"{px('stPosX'):.0f}, {px('stPosY'):.0f}, {px('stPosZ'):.0f}"),
                ("Accelerometer self-test, − (x, y, z) [LSB, 24 g]", f"{px('stNegX'):.0f}, {px('stNegY'):.0f}, {px('stNegZ'):.0f}"),
                ("Self-test passed", "yes" if px("stResult") == 0 else f"no ({px('stResult'):.0f})"),
                ("Gyroscope bias at boot (x, y, z) [LSB]", f"{px('gBiasX'):.2f}, {px('gBiasY'):.2f}, {px('gBiasZ'):.2f}"
                 + ("" if px("gBiasOk") else "  — not computed yet")),
                ("Accelerometer scale at boot", f"{px('accScale'):.5f}" + ("" if px("accScaleOk") else "  — not computed yet")),
                ("Board temperature [°C]", "—" if bp.get("board_temp_c") is None else f"{bp['board_temp_c']:.2f}"),
                ("Battery at rest [V]", "—" if bp.get("vbat_rest") is None else f"{bp['vbat_rest']:.2f}"),
            ]
            st.table(pd.DataFrame(rows, columns=["Quantity", "Value"]))

    # ── altered-drone sessions: compare with the drone's nominal boots ────
    if kind in CONDITION_KINDS:
        st.markdown("#### 4. Comparison with the normal boots of the drone")
        verdict_md = dirs["analysis"] / "verdict.md"
        n_ok = sum(1 for r in manifest.recordings if r.accepted_attempt is not None)
        st.caption(
            "Counts how many key symbols change against the normal boots of the same drone "
            "(motors and board separately), compares them with the normal spread and says "
            "whether the key would be regenerated. It changes neither the station nor the "
            f"drone's helper data. Result saved in `{dirs['analysis'].as_posix()}/`.")
        if st.button("🔬 Analyse this session", disabled=auto_on or n_ok == 0,
                     key=f"rig2_cond_analyse_{kind}_{session_n}"):
            from analysis.conditions import run as run_conditions
            try:
                with st.spinner("Analysing…"):
                    run_conditions(kind, session_n, verbose=False)
            except Exception as exc:
                _flash("error", f"Analysis failed: {exc}")
            st.rerun()
        if verdict_md.exists():
            st.markdown(verdict_md.read_text(encoding="utf-8"))

    # ── swap sessions: per-session verdict (frame vs motors) ──────────────
    if kind == "swap":
        st.markdown("#### 3. Analysis of the session")
        verdict_md = dirs["analysis"] / "verdict.md"
        n_ok = sum(1 for r in manifest.recordings if r.accepted_attempt is not None)
        st.caption(
            "Compares every stage with the identities of A and B (enrollment sessions and, "
            "when recorded on the same day, the *before the swap* sessions) and says, "
            "separately for the frame signature and for the motor signature, whether the "
            f"session looks like A or B. The result is saved in `{dirs['analysis'].as_posix()}/`."
        )
        if st.button("🔬 Analyse this session", disabled=auto_on or n_ok == 0,
                     key=f"rig2_swap_analyse_{session_n}"):
            from analysis.swap import run as run_swap_analysis
            with st.spinner("Analysing (about a minute the first time)…"):
                run_swap_analysis(session_n, verbose=False)
            st.rerun()
        if verdict_md.exists():
            st.markdown(verdict_md.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — Identity check (2026-10-09): key regeneration demo
# ─────────────────────────────────────────────────────────────────────────────
# Runs, on the PC, what the drone would run on board at power-up: fuzzy
# extractor on the boot-session measurements (no robustness tag since
# 2026-10-29: the station checks the helper's fingerprint) -> SPAKE2 with the
# (simulated) ground station -> one AES-GCM frame. Station database and the
# drone's helper data live in data/keystore/ (keyproto/boot.py).

with tab_identity:
    from keyproto import boot as kboot

    st.markdown("### Identity check — key generation at power-up")
    st.caption(
        "Simulates on the PC what the drone would do at every power-up: it measures board and "
        "motors (a **Boot** session), rebuilds the secret with the fuzzy extractor, runs the "
        "SPAKE2 exchange with the station and opens an AES-GCM encrypted channel. The station "
        "keeps the fingerprint of each drone's helper data and refuses a helper it does not "
        "know. Station data and the helper data of every drone are in `data/keystore/`."
    )
    st.info(
        "**This tab does not command the drone.** It works on sessions already recorded. Steps: "
        "(1) record sessions from the *New Rig Session* tab → type **Boot** and upload the SD logs; "
        "(2) here, section 1: enroll the drone using a few boot sessions as reference; "
        "(3) here, section 2: pick another boot session and press *Boot and generate the key*.",
        icon="ℹ️")

    msg = st.session_state.pop("kid_msg", None)
    if msg:
        (st.success if msg[0] == "success" else st.error)(msg[1])

    try:
        sysp = kboot.system_params()
        st.caption(f"System parameters: {len(sysp['features'])} symbols "
                   f"({len(kboot.MOTOR_FEATS)} motors + {len(kboot.BOARD_FEATS)} board), "
                   f"corrects up to t = {sysp['t']} wrong symbols.")
    except Exception as exc:
        st.error(f"System parameters cannot be computed: {exc}")
        sysp = None

    def _boot_sessions(drone=None, altered=False):
        """Boot sessions with a converted SD log; altered=True adds the
        propeller-change / payload ones (never offered for enrollment)."""
        out = []
        roots = [kboot.BOOT_DIR] + ([rig_dir_for(k) if rig_dir_for(k).is_absolute()
                                     else kboot.ROOT / rig_dir_for(k) for k in CONDITION_KINDS]
                                    if altered else [])
        for f in sorted(p for r in roots for p in r.glob("s*_*/session.json")):
            m = json.loads(f.read_text(encoding="utf-8"))
            if drone and m["drone"] != drone:
                continue
            ok = any(a.get("converted_path") for r in m["recordings"] for a in r["attempts"])
            out.append((f.parent, m["drone"], m["session"], ok))
        return out

    if sysp:
        st.markdown("#### 1. Enrollment of the drone")
        ec1, ec2 = st.columns([1, 3])
        e_drone = ec1.selectbox("Drone", ["A", "B", "C", "D"], key="kid_enroll_drone")
        sess = [s for s in _boot_sessions(e_drone) if s[3]]
        src = ec2.radio(
            "Reference", ["Recorded boot sessions", "Data of the earlier studies (first day)"],
            key="kid_enroll_src", horizontal=True,
            help="With the historical data the tab works even before any boot session is "
                 "recorded; the measurements come, however, from different sessions (motors "
                 "and board recorded separately).")
        chosen = []
        if src.startswith("Recorded"):
            chosen = st.multiselect(
                "Sessions to use as reference (their median)", [s[0].name for s in sess],
                default=[s[0].name for s in sess][:3], key=f"kid_enroll_sel_{e_drone}")
        if st.button("📝 Enroll the drone", key="kid_enroll_btn",
                     disabled=src.startswith("Recorded") and not chosen):
            try:
                if src.startswith("Recorded"):
                    X = [kboot.boot_features(kboot.BOOT_DIR / n)[0] for n in chosen]
                    ref = np.nanmedian(np.vstack(X), axis=0)
                    label = "boot:" + ",".join(chosen)
                else:
                    ref = kboot.historical_reference(e_drone)
                    label = "historical"
                kboot.enroll(e_drone, ref, label)
                st.session_state["kid_msg"] = ("success", f"Drone {e_drone} enrolled ({label}).")
            except Exception as exc:
                st.session_state["kid_msg"] = ("error", f"Enrollment failed: {exc}")
            st.rerun()

        st.markdown("#### 2. Boot of the drone")
        all_sess = [s for s in _boot_sessions(altered=True) if s[3]]
        if not all_sess:
            st.info("No **Boot** session with an uploaded SD log. Record them from the "
                    "*New Rig Session* tab → type *Boot*.")
        else:
            _alt = {"fixed_rig_props": " — propellers changed", "fixed_rig_payload": " — with payload"}
            labels = {f"{p.parent.name.replace('fixed_rig_', '')}/{p.name} (drone {d})"
                      f"{_alt.get(p.parent.name, '')}": (p, d) for p, d, _, _ in all_sess}
            pick = st.selectbox("Boot session", list(labels), key="kid_boot_pick_v2")
            claim = st.selectbox("The drone claims to be", ["A", "B", "C", "D"],
                                 index=["A", "B", "C", "D"].index(labels[pick][1]), key="kid_claim",
                                 help="Pick another drone to simulate an impostor.")
            if st.button("🔑 Boot and generate the key", key="kid_boot_btn"):
                try:
                    x, info = kboot.boot_features(labels[pick][0])
                    r = kboot.boot(claim, x)
                    st.session_state["kid_result"] = (pick, claim, info, r)
                except Exception as exc:
                    st.session_state["kid_msg"] = ("error", f"Boot failed: {exc}")
                st.rerun()

        res = st.session_state.get("kid_result")
        if res:
            pick, claim, info, r = res
            st.markdown(f"**Last boot:** `{pick}`, claimed as drone **{claim}** — "
                        f"board temperature {info.get('board_temp_c') or float('nan'):.1f} °C, "
                        f"missing quantities {info['n_missing']}"
                        + ("" if info.get("selftest_valid") else " (self-test not valid)"))
            if r["ok"]:
                st.success(
                    f"✅ Key rebuilt ({r['wrong_symbols']} symbols corrected out of {r['n_symbols']}, "
                    f"at most {r['t']}), SPAKE2 succeeded, encrypted channel open. "
                    f"Message decrypted by the station: “{r['decrypted']}”. "
                    "The drone's helper data were refreshed (drift).")
                _speed_alarm_warning(r)
                st.code(f"session key (first bytes): {r['session_key_prefix']}…\n"
                        f"ciphertext (first bytes):  {r['ciphertext_prefix']}…")
            else:
                st.error(f"❌ Rejected — stage: {r['stage']}; reason: {r['reason']}")
                _speed_alarm_warning(r)

        st.markdown("#### 3. Drones enrolled in the station")
        station = kboot._jload(kboot.KEYSTORE / "station.json", {})
        if station:
            st.caption("A drone in this table is enrolled: the station knows its password and "
                       "the drone holds its helper data.")
            st.table(pd.DataFrame([dict(drone=d, failed_attempts=v["failures"],
                                        enrolled_from=(kboot._jload(kboot.KEYSTORE / f"drone{d}_helper.json", {})
                                                       or {}).get("enrolled_from", "?"),
                                        speed_alarm=(", ".join(f"M{m}" for m in v["speed_alarm"]["motors"])
                                                          + f" ({v['speed_alarm']['at']})")
                                        if v.get("speed_alarm") else "—")
                                   for d, v in sorted(station.items())]))
            if st.button("Unlock the station (reset the failed attempts)", key="kid_reset"):
                for v in station.values():
                    v["failures"] = 0
                kboot._jsave(kboot.KEYSTORE / "station.json", station)
                st.rerun()
        else:
            st.caption("No drone enrolled.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — On-board key (2026-10-14): real-time test with firmware FW2
# ─────────────────────────────────────────────────────────────────────────────
# The drone (firmware variant FW2, firmware/uavibe_key) runs the boot protocol
# itself, regenerates the password from its own measurements and runs CPace
# with this station over the app channel. Station logic and database:
# keyproto/onboard_station.py (data/keystore/onboard/). "Simulated drone" runs
# the same exchange against a Python twin of the firmware fed with the
# recorded boot sessions, to rehearse without hardware.

with tab_onboard:
    from keyproto import boot as kb_sys
    from keyproto import onboard as kob
    from keyproto import onboard_station as kos

    st.markdown("### On-board key — real-time authentication")
    st.caption(
        "With firmware **FW2 (uavibe-key)** the drone measures its motors and board by itself, "
        "rebuilds the password with the fuzzy extractor and runs CPace with this station over the "
        "radio; if it succeeds it sends an encrypted message. Firmware, changes and procedure: "
        "`firmware/FIRMWARE.md`.")
    st.info(
        "**Steps.** 1) Flash FW2 and connect the drone from the sidebar. "
        "2) **Enrollment**: power-cycle the drone, “Enrollment boot”; repeat for "
        f"{kos.N_ENROLL} boots, then “Create and send the helper data”. "
        "3) **Authentication**: power-cycle, “Boot and authenticate”. To try an impostor, "
        "pick another name in “The station believes it is talking to”.", icon="ℹ️")
    st.warning("During every boot the motors spin at 58k PWM, one at a time: the drone must be "
               "clamped to the rig.", icon="⚠️")

    onb_msg = st.session_state.pop("onb_msg", None)
    if onb_msg:
        (st.success if onb_msg[0] == "success" else st.error)(onb_msg[1])

    try:
        onb_sysp = kb_sys.system_params()
    except Exception as exc:
        st.error(f"System parameters not available: {exc}")
        onb_sysp = None

    onb_mode = st.radio("Link", ["Real drone (firmware FW2)", "Simulated drone (no hardware)"],
                        horizontal=True, key="onb_mode")
    onb_sim = onb_mode.startswith("Simulated")
    onb_store = kos.STORE / "sim" if onb_sim else kos.STORE

    def _onb_log(rec: dict) -> None:
        onb_store.mkdir(parents=True, exist_ok=True)
        with open(onb_store / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(at=datetime.now().isoformat(timespec="seconds"), **rec), default=str) + "\n")

    @st.cache_data(show_spinner="Computing the on-board measurements from the recorded boots…")
    def _recorded_onboard_features() -> dict:
        out = {}
        for sd in sorted(kb_sys.BOOT_DIR.glob("s*_drone*")):
            try:
                inp = kob.boot_inputs(sd)
                out.setdefault(inp["drone"], []).append(kob.onboard_features(inp, kb_sys.system_params()).tolist())
            except Exception:
                pass
        return out

    onb_link = None
    if onb_sysp is not None and onb_sim:
        rec_f = _recorded_onboard_features()
        if not rec_f:
            st.info("No boot recorded in data/fixed_rig_boot/: the simulated drone has no measurements.")
        else:
            sc1, sc2 = st.columns(2)
            sim_drone = sc1.selectbox("Simulated drone", sorted(rec_f), key="onb_sim_drone",
                                      help="The Python twin of the firmware, with its helper data in memory.")
            meas_from = sc2.selectbox("The next boot uses the measurements of drone", sorted(rec_f),
                                      index=sorted(rec_f).index(sim_drone), key="onb_sim_meas",
                                      help="Another drone simulates an impostor posing as the simulated drone.")
            sims = st.session_state.setdefault("onb_sims", {})
            if sim_drone not in sims:
                sims[sim_drone] = kos.SimDrone(onb_sysp, [], kos.station_key(onb_store, create=True)[1])
            onb_link = sims[sim_drone]
            sim_idx = st.session_state.setdefault("onb_sim_idx", {})
            i_next = sim_idx.get(meas_from, 0) % len(rec_f[meas_from])
            onb_link.boots = [np.asarray(rec_f[meas_from][i_next])]
            st.caption(f"Next simulated boot: measurement {i_next + 1} of {len(rec_f[meas_from])} "
                       f"of drone {meas_from} (real boots of 2026-10-13, computed as on board).")
    elif onb_sysp is not None:
        w_onb = st.session_state.get("worker")
        if w_onb is not None and w_onb.is_alive() and w_onb.connected.is_set() and w_onb.cf is not None:
            onb_link = kos.CflibLink(w_onb.cf)
        else:
            st.warning("Connect the drone from the sidebar.", icon="⚠️")

    if onb_link is not None:
        station_onb = kos.Station(onb_link, onb_sysp, onb_store)
        if station_onb.key is None:
            st.error("The station key (Ed25519) that signs START, enrollment and erase is missing. "
                     "Run `python -m keyproto.onboard_station keygen` in auth_client, then rebuild and "
                     "reflash FW2 on every drone (the public key is compiled into the firmware).")

        def _advance_sim():
            if onb_sim:
                m_ = st.session_state["onb_sim_meas"]
                st.session_state["onb_sim_idx"][m_] = st.session_state["onb_sim_idx"].get(m_, 0) + 1

        def _battery_ok() -> bool:
            if onb_sim:
                return True
            v = read_vbat(onb_link.cf)
            if v is not None and v < VBAT_MIN_REST_V:
                st.session_state["onb_msg"] = ("error", f"Battery at rest {v:.2f} V, below {VBAT_MIN_REST_V:.2f} V: "
                                                        "let it recharge and try again.")
                return False
            return True

        def _run_blocking(expected: str) -> dict:
            bar = st.progress(0.0, text="Starting the protocol…")
            total = micro_duration_s("boot-seq") + 2.0
            t0 = time.monotonic()

            def prog(phase, motor, pct):
                el = time.monotonic() - t0
                bar.progress(min(1.0, el / total),
                             text=f"{phase}{f' — motor {motor}' if motor else ''} ({el:.0f} s)")

            return station_onb.run(expected, progress=prog)

        # ── 1. status ──
        st.markdown("#### 1. Drone status")
        hs = st.session_state.get("onb_hello")
        if st.button("🔎 Read status", key="onb_hello_btn"):
            hs = station_onb.hello()
            st.session_state["onb_hello"] = hs
            if hs is None:
                st.session_state["onb_msg"] = ("error", "No answer on the app channel: does the drone run firmware FW2?")
            st.rerun()
        if hs:
            st.markdown(f"Enrolled: **{'yes, as drone ' + hs['drone'] if hs['enrolled'] else 'no'}** · "
                        f"{hs['n']} symbols, corrects up to {hs['t']}"
                        + ("" if hs.get("signed_commands") else
                           " · ⚠️ no challenge in STATUS: firmware older than 2026-10-29, reflash it") + ".")

        # ── 2. enrollment ──
        st.markdown("#### 2. Enrollment")
        ec1, ec2 = st.columns([1, 3])
        e_default = st.session_state.get("onb_sim_drone", "A") if onb_sim else "A"
        e_name = ec1.selectbox("Drone name", ["A", "B", "C", "D"], key="onb_enroll_name",
                               index=["A", "B", "C", "D"].index(e_default))
        e_boots = station_onb.enroll_boots(e_name)
        ec2.markdown(f"Enrollment boots collected for **{e_name}**: **{len(e_boots)}** "
                     f"({kos.N_ENROLL} needed; the last {kos.N_ENROLL} are used).")
        b1, b2, b3 = st.columns(3)
        if b1.button("▶ Enrollment boot (motors!)", key="onb_enroll_run"):
            if _battery_ok():
                r = _run_blocking(e_name)
                _advance_sim()
                _onb_log(dict(sim=onb_sim, expected=e_name, **{k: v for k, v in r.items() if k != "features"}))
                if r.get("mode") == "enroll":
                    st.session_state["onb_msg"] = ("success", f"Enrollment boot {r['n_boots']} saved "
                                                              f"({r['n_missing']} missing quantities, {r['seconds']:.0f} s). "
                                                              "Power-cycle the drone before the next one.")
                elif r.get("ok") is False and r.get("stage") == "station":
                    st.session_state["onb_msg"] = ("error", r.get("reason", "?"))
                else:
                    st.session_state["onb_msg"] = ("error", "The drone is already enrolled (it answered with the key "
                                                            "exchange): to enroll it again use “Restart enrollment”.")
            st.rerun()
        if b2.button("📝 Create and send the helper data", key="onb_enroll_finish",
                     disabled=len(e_boots) < kos.N_ENROLL):
            r = station_onb.finish_enrollment(e_name)
            _onb_log(dict(sim=onb_sim, expected=e_name, mode="finish_enrollment", **r))
            st.session_state["onb_msg"] = (("success", f"Drone {e_name} enrolled: {r['helper_bytes']} bytes "
                                                       "of helper data in the drone's EEPROM, password in the station.")
                                           if r["ok"] else ("error", r["reason"]))
            st.session_state.pop("onb_hello", None)
            st.rerun()
        sure = b3.checkbox(f"I want to erase drone {e_name}", key=f"onb_forget_sure_{e_name}")
        if b3.button("🗑 Restart enrollment", key="onb_enroll_forget", disabled=not sure,
                     help="Removes the drone from the station and sets its enrollment boots aside; "
                          "it also erases the helper data on the CONNECTED drone (ERASE signed by the "
                          "station). Refused if the connected drone is enrolled under another name."):
            hs_now = None if onb_sim else station_onb.hello()
            if hs_now is not None and hs_now["enrolled"] and hs_now["drone"] != e_name:
                # 2026-11: "A" left selected while D was connected removed A from the
                # station and erased D's memory
                st.session_state["onb_msg"] = (
                    "error", f"The connected drone is enrolled as {hs_now['drone']}, not {e_name}: "
                             "nothing erased. Pick the right name.")
                st.rerun()
            st.session_state.pop(f"onb_forget_sure_{e_name}", None)
            station_onb.forget(e_name)
            if onb_sim:
                st.session_state["onb_sims"].pop(st.session_state["onb_sim_drone"], None)
            else:
                r_er = station_onb.erase()
                if not r_er["ok"]:
                    st.session_state["onb_msg"] = ("error", f"Erase on the drone failed: {r_er['reason']}")
            st.session_state.pop("onb_hello", None)
            st.rerun()

        # ── 3. authentication ──
        st.markdown("#### 3. Boot and authentication")
        onb_db = station_onb.db()
        onb_names = sorted(onb_db) or ["A", "B", "C", "D"]
        c_default = hs["drone"] if hs and hs.get("drone") in onb_names else onb_names[0]
        claim_onb = st.selectbox("The station believes it is talking to", onb_names,
                                 index=onb_names.index(c_default), key="onb_claim",
                                 help="A name other than the connected drone's simulates a drone posing "
                                      "as another one: it must fail.")
        onb_conds = {"nominal (as enrolled)": "nominal", "propellers changed": "propeller",
                     "tape on the blades": "tape", "payload added": "payload",
                     "memory cloned from another drone": "cloned_helper", "other": "other"}
        cc1, cc2 = st.columns([1, 2])
        cond_lbl = cc1.selectbox("Condition of the drone", list(onb_conds), key="onb_cond",
                                 help="Written to the log: `python -m analysis.onboard_log` estimates the "
                                      "false rejections on the nominal boots only.")
        cond_note = cc2.text_input("Note (which motor, what change…)", key="onb_cond_note")
        if st.button("🔑 Boot and authenticate (motors!)", key="onb_auth_run", disabled=claim_onb not in onb_db):
            if _battery_ok():
                r = _run_blocking(claim_onb)
                _advance_sim()
                _onb_log(dict(sim=onb_sim, expected=claim_onb, condition=onb_conds[cond_lbl],
                              condition_note=cond_note or None,
                              **{k: v for k, v in r.items() if k != "features"}))
                st.session_state["onb_last"] = (claim_onb, r)
            st.rerun()
        onb_last = st.session_state.get("onb_last")
        if onb_last:
            claim_l, r = onb_last
            if r.get("ok"):
                st.success(f"✅ Authenticated as **{claim_l}** in {r['seconds']:.0f} s: password regenerated "
                           f"on board, CPace succeeded, mutual confirmation. Encrypted message decrypted by "
                           f"the station: “{r['message']}”. Session key (first bytes): "
                           f"`{r['session_key_prefix']}…`")
                _speed_alarm_warning(r)
            elif r.get("mode") == "enroll":
                st.warning("The drone is not enrolled: it sent its measurements (enrollment boot saved).")
            else:
                st.error(f"❌ Rejected ({r.get('stage')}): {r.get('reason')}"
                         + (f" — wrong symbols {r['wrong']}" if r.get("wrong") not in (None, -1) else ""))
                _speed_alarm_warning(r)

        # ── 4. station ──
        st.markdown("#### 4. Station")
        if onb_db:
            st.table(pd.DataFrame([dict(drone=d, failed_attempts=v["failures"], enrolled=v.get("enrolled_at"),
                                        last_access=v.get("last_ok", "—"),
                                        helper_fingerprint=(v["helper_fp"][:12] + "…") if v.get("helper_fp")
                                        else "to register",
                                        speed_alarm=(", ".join(f"M{m}" for m in v["speed_alarm"]["motors"])
                                                          + f" ({v['speed_alarm']['at']})")
                                        if v.get("speed_alarm") else "—")
                                   for d, v in sorted(onb_db.items())]))
            rc1, rc2 = st.columns([1, 3])
            r_d = rc1.selectbox("Reset the failed attempts of", sorted(onb_db), key="onb_reset_d")
            if rc1.button("Reset", key="onb_reset_btn"):
                station_onb.reset_failures(r_d)
                st.rerun()
            rc2.caption("Since 2026-10-29 the helper data carry no robustness tag (it was an offline "
                        "oracle): the station keeps their **fingerprint** and the drone binds it into the "
                        "CPace exchange. A drone enrolled before (v2 helper) must be migrated once: reflash "
                        "the firmware, power the drone on and press the button below (in the lab only: the "
                        "station trusts what it reads). The password does not change, no new enrollment.")
            if rc2.button("🔗 Register the helper fingerprint of the connected drone", key="onb_fp_btn"):
                r = station_onb.register_helper_fp()
                _onb_log(dict(sim=onb_sim, mode="register_helper_fp", **r))
                st.session_state["onb_msg"] = (("success", f"Helper fingerprint of {r['drone']} registered"
                                                           + (" (v2 helper with a tag: the drone already "
                                                              "converted it to v3)" if r["legacy"] else "") + ".")
                                               if r["ok"] else ("error", r["reason"]))
                st.rerun()
        else:
            st.caption("No drone enrolled in this station.")
        onb_logp = onb_store / "log.jsonl"
        if onb_logp.exists():
            with st.expander("Log of the attempts"):
                onb_rows = [json.loads(l) for l in onb_logp.read_text(encoding="utf-8").splitlines() if l.strip()]
                st.dataframe(pd.DataFrame(onb_rows[::-1]), width="stretch")

        # ── 5. attack: cloned memory ──
        st.markdown("#### 5. Security test: cloned memory")
        st.caption("The attacker reads the memory of an enrolled drone (its helper data) and copies it into "
                   "another drone, which then presents itself under that name. It must be rejected: the helper "
                   "data alone are not enough, the mechanics and the board of the real drone are needed. "
                   "Steps: 1) connect the drone to clone and copy its helper; 2) connect the other drone, save "
                   "a copy of its helper (to put it back later), erase it and write the cloned one; 3) in "
                   "section 3 pick the name of the cloned drone and boot; 4) erase and write the original "
                   "helper back. Since 2026-10-29 the station accepts only helper data whose fingerprint it "
                   "knows (current or previous): a copy saved more than one successful boot ago is refused "
                   "even on the right drone, so put the original back right away, or register the "
                   "fingerprint again (section 4).")
        atk_dir = onb_store / "attack"
        ac1, ac2 = st.columns(2)
        if ac1.button("📥 Copy the helper data of this drone", key="onb_atk_read"):
            blob = station_onb.read_helper()
            if blob is None:
                st.session_state["onb_msg"] = ("error", "The drone is not enrolled (or does not answer): nothing to copy.")
            else:
                who = chr(kob.unpack_helper(blob)["drone_id"])
                atk_dir.mkdir(parents=True, exist_ok=True)
                p = atk_dir / f"helper_{who}_{datetime.now():%Y%m%d_%H%M%S}.bin"
                p.write_bytes(blob)
                _onb_log(dict(sim=onb_sim, mode="attack_read", drone=who, bytes=len(blob), file=p.name))
                st.session_state["onb_msg"] = ("success", f"Copied the helper data of drone {who} "
                                                          f"({len(blob)} bytes) to {p.name}.")
            st.rerun()
        if ac2.button("🧹 Erase only the helper on the drone (signed ERASE)", key="onb_atk_erase",
                      help="Does not touch the station: needed to write a cloned helper or put the original "
                           "back. In the test we play the attacker who writes the memory, which is why the "
                           "erase goes through the station."):
            r_er = station_onb.erase()
            if not r_er["ok"]:
                st.session_state["onb_msg"] = ("error", f"Erase failed: {r_er['reason']}")
            _onb_log(dict(sim=onb_sim, mode="attack_erase"))
            st.session_state.pop("onb_hello", None)
            st.rerun()
        atk_files = sorted(atk_dir.glob("helper_*.bin"), reverse=True) if atk_dir.exists() else []
        if atk_files:
            pc1, pc2 = st.columns([3, 1])
            atk_pick = pc1.selectbox("Helper to write on the connected drone", [p.name for p in atk_files],
                                     key="onb_atk_pick")
            if pc2.button("📤 Write", key="onb_atk_put"):
                blob = (atk_dir / atk_pick).read_bytes()
                ok = station_onb.put_helper(blob)
                _onb_log(dict(sim=onb_sim, mode="attack_put", file=atk_pick, ok=ok))
                st.session_state["onb_msg"] = (("success", f"Wrote {atk_pick} on the drone: it now presents "
                                                           f"itself as drone {chr(kob.unpack_helper(blob)['drone_id'])}.")
                                               if ok else ("error", "The drone refused the helper: is it still "
                                                                    "enrolled? Erase it first (🧹)."))
                st.session_state.pop("onb_hello", None)
                st.rerun()

        if not onb_sim:
            onb_link.close()


# ─────────────────────────────────────────────────────────────────────────────
# Keep the sidebar's connection badge live
# ─────────────────────────────────────────────────────────────────────────────
# Unconditional — no checkbox — but only while a worker exists (nothing to
# watch once disconnected, no reason to keep waking the script up). This is
# safe from the GIL-contention bug that the old Live-tab auto-refresh could
# trigger (see _run_sweep_blocking's docstring above): that bug came from a
# SWEEP running in its own background thread while auto-refresh reran the
# main thread concurrently. Every recording/sweep call in this file now
# runs synchronously in the main script thread instead, so by the time this
# block is even reached the script has already finished whatever blocking
# call happened earlier in the SAME run — there's no other thread for this
# rerun to race.

if st.session_state.worker is not None:
    time.sleep(CONNECTION_POLL_S)
    st.rerun()
