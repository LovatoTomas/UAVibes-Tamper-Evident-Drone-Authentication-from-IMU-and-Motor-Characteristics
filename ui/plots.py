"""
UAVibes dashboard — Plotly figure builders.

Figures:

  signal_figure               — time-domain raw signal vs noise floor
  fft_figure                  — frequency-domain power spectrum of noise floor
  spectrogram_figure          — time-frequency view, raw vs filtered, with
                                each motor's predicted fundamental overlaid
  spectrum_snapshot_figure    — one-instant spectrum + comb-filter preview
  calibration_candidates_figure — human-in-the-loop motor-model calibration:
                                ranked candidate peaks near the prediction
  manifold_figure              — 2-D latent space with enrollment cluster and
                                live trajectory, plus the Mahalanobis
                                threshold ellipse
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import spectrogram as _sp_spectrogram, welch as _sp_welch

from pipeline.motor_model import pwm_to_hz, MIN_SPIN_PWM
from pipeline.preprocessing import MIN_MOTOR_HZ


# ── colour palette ────────────────────────────────────────────────────────────

_RAW_COLOUR   = "rgba(160,160,160,0.6)"
_NOISE_COLOUR = "#1f77b4"           # blue
_MOTOR_COLOUR = "rgba(255,127,14,0.7)"  # orange
_AUTH_GREEN   = "#2ca02c"
_AUTH_RED     = "#d62728"
_ENROLL_CLR   = "rgba(150,150,150,0.4)"
_TRAJ_CLR     = "Blues"             # plotly colorscale for trajectory age


# ── Panel 1: Raw vs Filtered Signal ──────────────────────────────────────────

def signal_figure(
    df: pd.DataFrame,
    channel: str = "acc.x",
    display_s: float = 5.0,
) -> go.Figure:
    """
    Time-domain comparison of raw signal vs noise floor.

    Two vertically stacked subplots share the X (time) axis:
      • Top    — raw IMU value (light grey)
      • Bottom — noise floor after highpass + harmonic decon (blue)

    Parameters
    ----------
    df        : preprocessed DataFrame with both raw and noise.* columns.
    channel   : base channel name, e.g. ``"acc.x"`` or ``"gyro.z"``.
    display_s : window width in seconds (most recent N seconds shown).
    """
    raw_col   = channel
    noise_col = "noise." + channel
    unit      = "g" if channel.startswith("acc") else "°/s"

    if raw_col not in df.columns or noise_col not in df.columns:
        return _empty_fig(f"Waiting for data ({channel})…")

    # Keep only last display_s seconds
    t_max = df["timestamp_s"].iloc[-1]
    mask  = df["timestamp_s"] >= (t_max - display_s)
    sub   = df[mask]
    t     = sub["timestamp_s"].values - sub["timestamp_s"].iloc[0]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.08,
        subplot_titles=["Raw signal", "Noise floor (after filtering)"],
    )

    fig.add_trace(
        go.Scatter(x=t, y=sub[raw_col].values, name="raw",
                   line=dict(color=_RAW_COLOUR, width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=sub[noise_col].values, name="noise floor",
                   line=dict(color=_NOISE_COLOUR, width=1.5)),
        row=2, col=1,
    )

    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_yaxes(title_text=unit, row=1, col=1)
    fig.update_yaxes(title_text=unit, row=2, col=1)
    fig.update_layout(
        height=380,
        margin=dict(l=50, r=20, t=40, b=30),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Panel 2: Spectral / Feature View ─────────────────────────────────────────

def fft_figure(
    df: pd.DataFrame,
    fs: float,
    channel: str = "acc.x",
    motor_hz: "float | None" = None,
    window_s: float = 1.0,
) -> go.Figure:
    """
    Power spectrum (dB) of the noise floor for the last ``window_s`` seconds.

    Motor harmonic frequencies are marked with vertical dashed lines when
    ``motor_hz`` (the fundamental blade-pass frequency) is provided.

    Parameters
    ----------
    df       : preprocessed DataFrame with noise.* columns.
    fs       : sampling rate in Hz.
    channel  : base channel name (prefix "noise." is prepended automatically).
    motor_hz : fundamental motor frequency [Hz]; harmonics drawn at k × motor_hz.
    window_s : length of the FFT window in seconds.
    """
    noise_col = "noise." + channel

    if noise_col not in df.columns or len(df) < 8:
        return _empty_fig("Waiting for data…")

    # Take last window_s seconds
    n_win = min(len(df), int(window_s * fs))
    x     = df[noise_col].values[-n_win:].astype(float)
    win   = np.hanning(len(x))
    X     = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)

    # Convert to dB relative to max
    X_db  = 10 * np.log10(np.maximum(X, 1e-20))
    X_db -= X_db.max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=X_db,
        mode="lines",
        name=noise_col,
        line=dict(color=_NOISE_COLOUR, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(31,119,180,0.15)",
    ))

    # Motor harmonic markers
    if motor_hz and motor_hz > 0:
        nyq = fs / 2.0
        k   = 1
        while k * motor_hz < nyq:
            fig.add_vline(
                x=k * motor_hz,
                line=dict(color=_MOTOR_COLOUR, width=1, dash="dot"),
                annotation_text=f"{k}f₀" if k <= 3 else "",
                annotation_font_size=10,
            )
            k += 1

    unit = "g" if channel.startswith("acc") else "°/s"
    fig.update_layout(
        xaxis_title="frequency (Hz)",
        yaxis_title="power (dB, rel. max)",
        height=200,
        margin=dict(l=50, r=20, t=10, b=40),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(range=[0, fs / 2])
    fig.update_yaxes(range=[-80, 5])
    return fig


# ── Panel: Spectrogram (raw vs filtered) — visual filter QA ────────────────────

_MOTOR_LINE_COLORS = {1: "#e41a1c", 2: "#ff7f00", 3: "#4daf4a", 4: "#984ea3"}
_MOTOR_TRACK_SAMPLE_PTS = 300   # decimation before predicting f0(t) — PWM
                                # changes slowly enough that this is plenty;
                                # avoids calling pwm_to_hz per raw sample.
_MIN_MOTOR_HZ_TRACK = 10.0      # mirrors preprocessing.MIN_MOTOR_HZ — below
                                # this the overlay line breaks (motor "off")
                                # instead of drawing a misleading flat line.


def _alias_into_nyquist(freq: np.ndarray, fs: float) -> np.ndarray:
    """
    Fold a frequency (or array of them) into [0, fs/2] the way real sampling
    aliasing does. At flight-range PWM (45k-55k) the blade-pass 2nd harmonic
    (2xf0, roughly 590-810 Hz) exceeds the SD deck's 500 Hz Nyquist and
    aliases DOWN into ~190-410 Hz — the same band resonance candidates have
    been searched in. Without this fold, an overlay line for 2xf0 would sit
    off-chart (invisible) while the real aliased energy shows up in-band
    anyway, silently misleading the "does the model predict the right
    frequency" check this plot exists for. preprocessing.harmonic_
    deconvolution does NOT currently notch this aliased line (it only
    notches harmonics below Nyquist at their true frequency) — a peak in
    that band on a high-PWM recording may be this alias, not a chassis mode.
    """
    nyq = fs / 2.0
    f = np.mod(freq, fs)
    return np.where(f > nyq, fs - f, f)


def _motor_f0_track(df: pd.DataFrame, motor_id: int, t_query: np.ndarray) -> np.ndarray:
    """
    Predicted fundamental frequency for one motor (pipeline.motor_model,
    per-motor MOTOR_GAIN included), resampled onto ``t_query`` — normally a
    spectrogram's time-bin centres. Uses real per-sample vbat when the
    DataFrame has a ``vbat_v`` column (SD-deck conversions do), else the
    model's nominal voltage. NaN wherever the motor is below
    ``_MIN_MOTOR_HZ_TRACK`` so the overlay line breaks rather than drawing a
    flat line at zero.
    """
    col = f"motor.m{motor_id}"
    if col not in df.columns:
        return np.full_like(t_query, np.nan, dtype=float)

    n = len(df)
    n_sample = min(_MOTOR_TRACK_SAMPLE_PTS, n)
    idx = np.linspace(0, n - 1, n_sample).astype(int)
    t0 = df["timestamp_s"].values[0]
    t_s    = df["timestamp_s"].values[idx] - t0
    pwm_s  = df[col].values[idx].astype(float)
    vbat_s = (df["vbat_v"].values[idx].astype(float) if "vbat_v" in df.columns
              else [None] * n_sample)

    f0_s = np.array([pwm_to_hz(p, v, motor_id=motor_id) for p, v in zip(pwm_s, vbat_s)])
    f0_q = np.interp(t_query, t_s, f0_s)
    return np.where(f0_q >= _MIN_MOTOR_HZ_TRACK, f0_q, np.nan)


def spectrogram_figure(
    df: pd.DataFrame,
    fs: float,
    channel: str = "acc.x",
    window_s: float = 1.0,
    show_harmonics: bool = False,
    mark_t: "float | None" = None,
) -> go.Figure:
    """
    Two stacked spectrograms (dB power, SHARED colour reference) on a common
    time axis: raw signal on top, filtered noise floor (post highpass + comb
    filter) on the bottom. Sharing the dB reference (both normalised to the
    raw panel's max) is deliberate — it's what lets a notch's effect actually
    read as "darker" on the filtered panel instead of each panel silently
    re-normalising itself back to full contrast.

    Each active motor's PREDICTED fundamental frequency (pipeline.motor_model,
    including its per-motor gain) is overlaid as a dashed line on both panels:
    on the raw panel it should sit on top of a real energy ridge (validates
    the motor model's prediction against what was actually recorded); on the
    filtered panel that ridge should be gone (validates the comb filter
    notched the right frequency, per-motor — see
    preprocessing.harmonic_deconvolution).

    show_harmonics also overlays each motor's 2nd harmonic (blade-pass, 2xf0)
    as a fainter dotted line — the next line the comb filter removes.
    """
    raw_col   = channel
    noise_col = "noise." + channel
    has_noise = noise_col in df.columns

    if raw_col not in df.columns or len(df) < 32:
        return _empty_fig(f"Waiting for data ({channel})…")

    nperseg  = int(np.clip(window_s * fs, 32, len(df)))
    noverlap = nperseg // 2

    x_raw = df[raw_col].values.astype(float)
    f, t_spec, Sxx_raw = _sp_spectrogram(
        x_raw, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann"
    )
    Sxx_raw_db = 10 * np.log10(np.maximum(Sxx_raw, 1e-20))
    ref = Sxx_raw_db.max()   # shared reference for both panels — see docstring
    Sxx_raw_db = Sxx_raw_db - ref

    n_rows = 2 if has_noise else 1
    titles = ["Raw signal"] + (["Filtered (noise floor)"] if has_noise else [])
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5] if has_noise else [1.0],
        vertical_spacing=0.12,
        subplot_titles=titles,
    )

    fig.add_trace(go.Heatmap(
        x=t_spec, y=f, z=Sxx_raw_db,
        colorscale="Viridis", zmin=-80, zmax=0,
        colorbar=dict(title="dB", len=0.9 if not has_noise else 0.42,
                       y=0.5 if not has_noise else 0.79),
        hovertemplate="t=%{x:.2f}s  f=%{y:.1f}Hz  %{z:.1f}dB<extra></extra>",
    ), row=1, col=1)

    if has_noise:
        x_noise = df[noise_col].values.astype(float)
        _, _, Sxx_noise = _sp_spectrogram(
            x_noise, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann"
        )
        Sxx_noise_db = 10 * np.log10(np.maximum(Sxx_noise, 1e-20)) - ref
        fig.add_trace(go.Heatmap(
            x=t_spec, y=f, z=Sxx_noise_db,
            colorscale="Viridis", zmin=-80, zmax=0,
            colorbar=dict(title="dB", len=0.42, y=0.21),
            hovertemplate="t=%{x:.2f}s  f=%{y:.1f}Hz  %{z:.1f}dB<extra></extra>",
        ), row=2, col=1)

    active_motors = [
        m for m in (1, 2, 3, 4)
        if f"motor.m{m}" in df.columns and df[f"motor.m{m}"].max() > MIN_SPIN_PWM
    ]
    for m in active_motors:
        colour = _MOTOR_LINE_COLORS[m]
        f0_true = _motor_f0_track(df, m, t_spec)   # true predicted frequency, unfolded
        f0_plot = _alias_into_nyquist(f0_true, fs)
        aliased = np.nanmax(np.abs(f0_plot - f0_true)) > 0.5 if len(f0_true) else False
        for row in range(1, n_rows + 1):
            fig.add_trace(go.Scatter(
                x=t_spec, y=f0_plot, mode="lines",
                name=f"M{m} f0 (predicted)" + (" [aliased]" if aliased else ""),
                line=dict(color=colour, width=1.5, dash="dash"),
                showlegend=(row == 1),
                legendgroup=f"m{m}",
                hoverinfo="skip",
            ), row=row, col=1)
            if show_harmonics:
                harm2_plot = _alias_into_nyquist(f0_true * 2, fs)
                fig.add_trace(go.Scatter(
                    x=t_spec, y=harm2_plot, mode="lines",
                    name=f"M{m} 2×f0 (blade-pass, aliased if folded)",
                    line=dict(color=colour, width=1, dash="dot"),
                    opacity=0.6,
                    showlegend=(row == 1),
                    legendgroup=f"m{m}2",
                    hoverinfo="skip",
                ), row=row, col=1)

    if mark_t is not None:
        for row in range(1, n_rows + 1):
            fig.add_vline(
                x=mark_t, row=row, col=1,
                line=dict(color="white", width=1.5, dash="solid"),
                opacity=0.9,
            )

    fig.update_yaxes(title_text="Hz", range=[0, fs / 2], row=1, col=1)
    if has_noise:
        fig.update_yaxes(title_text="Hz", range=[0, fs / 2], row=2, col=1)
        fig.update_xaxes(title_text="time (s)", row=2, col=1)
    else:
        fig.update_xaxes(title_text="time (s)", row=1, col=1)

    fig.update_layout(
        height=600 if has_noise else 340,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Panel: Spectrum snapshot + comb-filter preview at one instant ──────────────

def spectrum_snapshot_figure(
    df: pd.DataFrame,
    fs: float,
    channel: str = "acc.x",
    t_center: float = 0.0,
    window_s: float = 2.0,
    n_harmonics: int = 5,
    Q: float = 30.0,
) -> go.Figure:
    """
    Power spectrum (dB) of a single window_s-second slice centred on
    t_center — raw vs filtered, overlaid with a shaded band at every
    frequency the comb filter notches for THAT instant, one colour per
    active motor.

    Where spectrogram_figure trades frequency resolution for a time axis
    (so you can watch a line move or a resonance persist across a whole
    recording), this trades the time axis away for maximum frequency
    resolution at one moment — the right tool for "does the real peak
    actually sit inside the notch, or is the model off by more than the
    notch is wide". Each shaded band mirrors preprocessing._notch_cascade
    exactly: centred at k*f0 for k=1..n_harmonics (stopping at the same
    Nyquist cutoff the real filter uses), width = (k*f0)/Q — the notch's
    real -3dB bandwidth, e.g. ~3.3% of centre frequency at the default Q=30.
    f0 itself is predicted from this window's own mean PWM and vbat via
    pipeline.motor_model, per-motor gain included.

    This uses ONE mean PWM per motor for the whole window, unlike the real
    filter which re-estimates every block_n=256 samples — accurate as long
    as the window sits inside a single steady PWM hold (e.g. an
    "all_hold_53000" phase), which is the intended use.
    """
    raw_col   = channel
    noise_col = "noise." + channel
    has_noise = noise_col in df.columns

    if raw_col not in df.columns or len(df) < 16:
        return _empty_fig(f"Waiting for data ({channel})…")

    t0    = df["timestamp_s"].values[0]
    t_rel = df["timestamp_s"].values - t0
    mask  = (t_rel >= t_center - window_s / 2) & (t_rel < t_center + window_s / 2)
    sub   = df[mask]
    if len(sub) < 16:
        return _empty_fig("Not enough samples in this window.")

    def _spectrum(col: str) -> "tuple[np.ndarray, np.ndarray]":
        x     = sub[col].values.astype(float)
        win   = np.hanning(len(x))
        X     = np.abs(np.fft.rfft(x * win)) ** 2
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
        return freqs, 10 * np.log10(np.maximum(X, 1e-20))

    freqs, raw_db = _spectrum(raw_col)
    ref = raw_db.max()
    raw_db = raw_db - ref

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=raw_db, mode="lines", name="raw",
        line=dict(color=_RAW_COLOUR, width=1.3),
    ))
    if has_noise:
        _, noise_db = _spectrum(noise_col)
        noise_db = noise_db - ref
        fig.add_trace(go.Scatter(
            x=freqs, y=noise_db, mode="lines", name="filtered",
            line=dict(color=_NOISE_COLOUR, width=1.5),
        ))

    # Per-motor notch bands — mirrors preprocessing._notch_cascade's own
    # loop and break condition exactly, so the shading is literally what
    # the real filter would do for this window's mean PWM/vbat.
    nyq_cut   = fs / 2.0 - 1.0
    mean_vbat = float(sub["vbat_v"].mean()) if "vbat_v" in sub.columns else None
    for m in (1, 2, 3, 4):
        col = f"motor.m{m}"
        if col not in sub.columns:
            continue
        pwms   = sub[col].values.astype(float)
        active = pwms[pwms > MIN_SPIN_PWM]
        if active.size == 0:
            continue
        mean_pwm = active.mean()
        f0 = pwm_to_hz(mean_pwm, mean_vbat, motor_id=m)
        if f0 < MIN_MOTOR_HZ:
            continue

        colour = _MOTOR_LINE_COLORS[m]
        for k in range(1, n_harmonics + 1):
            fk = k * f0
            if fk >= nyq_cut:
                break
            bw = fk / Q
            fig.add_vrect(
                x0=fk - bw / 2, x1=fk + bw / 2,
                fillcolor=colour, opacity=0.15, line_width=0,
                annotation_text=(f"M{m}" if k == 1 else ""),
                annotation_position="top left",
                annotation_font_size=9, annotation_font_color=colour,
            )
            fig.add_vline(
                x=fk, line=dict(color=colour, width=1, dash="dot"), opacity=0.6,
            )

    fig.update_layout(
        xaxis_title="frequency (Hz)",
        yaxis_title="power (dB, rel. max)",
        height=600,
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(range=[0, fs / 2])
    fig.update_yaxes(range=[-140, 0])
    return fig


# ── Panel: human-in-the-loop motor-model calibration ────────────────────────────

def calibration_candidates_figure(
    seg: pd.DataFrame,
    fs: float,
    f0_pred: float,
    ranked: "list[tuple[float, float, float]]",
    chosen_idx: "int | None",
    channel: str = "acc.x",
    manual_freq: "float | None" = None,
) -> go.Figure:
    """
    Welch spectrum of one hold segment, zoomed around the model's prediction,
    with every ranked candidate peak (from usd_motor_analysis.rank_candidates)
    marked and numbered by rank. The currently CHOSEN candidate (chosen_idx
    into ``ranked``, or None when a manual value is being used instead) is
    highlighted solid green; the rest are thin grey dashed. The model's
    prediction itself is a dashed orange line, separate from any candidate,
    so a big gap between it and the chosen peak is visible at a glance —
    that gap is exactly the correction this calibration pass is collecting
    ground truth for. manual_freq draws a solid blue line for an operator-
    typed frequency (not one of the auto-detected candidates), so a manual
    override can be visually checked before confirming it.
    """
    if channel not in seg.columns or len(seg) < 32:
        return _empty_fig(f"Not enough data ({channel})")

    x = seg[channel].values.astype(float)
    x = x - x.mean()
    nper = min(2048, len(x))
    freqs, psd = _sp_welch(x, fs=fs, nperseg=nper)
    psd_db = 10 * np.log10(np.maximum(psd, 1e-20))
    ref = psd_db.max()
    psd_db = psd_db - ref

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=psd_db, mode="lines", name=channel,
        line=dict(color=_NOISE_COLOUR, width=1.5),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.12)",
    ))

    fig.add_vline(
        x=f0_pred, line=dict(color=_MOTOR_COLOUR, width=1.5, dash="dash"),
        annotation_text="model prediction", annotation_position="top",
        annotation_font_size=10,
    )

    for i, (f, prom, score) in enumerate(ranked):
        is_chosen = (i == chosen_idx)
        fig.add_vline(
            x=f,
            line=dict(color=("#2ca02c" if is_chosen else "#999999"),
                      width=(2.5 if is_chosen else 1), dash="solid" if is_chosen else "dot"),
            annotation_text=f"#{i+1}" + ("  CHOSEN" if is_chosen else ""),
            annotation_position="top",
            annotation_font_size=(12 if is_chosen else 9),
            annotation_font_color=("#2ca02c" if is_chosen else "#999999"),
        )

    if manual_freq is not None:
        fig.add_vline(
            x=manual_freq, line=dict(color="#1f77b4", width=2.5, dash="solid"),
            annotation_text="manual", annotation_position="bottom",
            annotation_font_size=11, annotation_font_color="#1f77b4",
        )

    window = max(f0_pred * 0.25, 15.0, abs((manual_freq or f0_pred) - f0_pred) * 1.3)
    fig.update_layout(
        xaxis_title="frequency (Hz)",
        yaxis_title="power (dB, rel. max)",
        height=420,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(range=[max(0, f0_pred - window), f0_pred + window])
    fig.update_yaxes(range=[-100, 5])
    return fig


# ── Panel: structural resonance candidates (cross-segment vote) ────────────────

def resonance_bar_figure(
    known: "list[dict]", other: "list[dict]", n_segs: int,
) -> go.Figure:
    """
    One bar per candidate frequency, height = how many segments (across
    however many recordings were selected) confirmed a peak there after
    each segment's own motor line(s) were excluded — this IS the evidence:
    a motor line moves with PWM so it can't rack up hits across many
    different PWM/motor conditions the way a fixed structural resonance
    does (see usd_motor_analysis.find_resonance_candidates).

    Pre-registered candidates (``known``, from KNOWN_CANDIDATES) are green
    if they cleared the reproduction threshold this run and red if not;
    newly-discovered ones (``other``) are blue. The dashed line marks that
    threshold (max(2, n_segs//4)) so "why is this one green" is visible at
    a glance, not just asserted.
    """
    if not known and not other:
        return _empty_fig("No candidates — run the analysis first.")

    freqs  = [k["freq"] for k in known] + [o["freq"] for o in other]
    hits   = [k["hits"] for k in known] + [o["hits"] for o in other]
    colors = (["#2ca02c" if k["reproduced"] else "#d62728" for k in known]
             + ["#1f77b4"] * len(other))
    kind   = (["known candidate"] * len(known) + ["newly discovered"] * len(other))
    text   = [f"{h}/{n_segs}" for h in hits]

    fig = go.Figure(go.Bar(
        x=freqs, y=hits, marker_color=colors, text=text, textposition="outside",
        customdata=kind,
        hovertemplate="%{x:.1f} Hz — %{y} segments (%{customdata})<extra></extra>",
    ))

    threshold = max(2, n_segs // 4)
    fig.add_hline(
        y=threshold, line=dict(color="#888", width=1.5, dash="dash"),
        annotation_text=f"reproduction threshold ({threshold}/{n_segs})",
        annotation_position="top right", annotation_font_size=11,
    )

    fig.update_layout(
        xaxis_title="frequency (Hz)",
        yaxis_title=f"segments confirming (of {n_segs})",
        height=420,
        margin=dict(l=50, r=20, t=30, b=40),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(range=[0, max(hits + [threshold]) * 1.25])
    return fig


# ── Feature label helpers ─────────────────────────────────────────────────────

_FEAT_META: "dict[str, tuple[str, str, str]]" = {
    # feat_suffix: (display_name, unit_acc, unit_gyro)
    "var":      ("Variance",          "g²",   "(°/s)²"),
    "skew":     ("Skewness",          "—",    "—"),
    "kurt":     ("Kurtosis (excess)", "—",    "—"),
    "centroid": ("Spectral Centroid", "Hz",   "Hz"),
    "rolloff":  ("Spectral Roll-off", "Hz",   "Hz"),
    "flatness": ("Spectral Flatness", "0–1",  "0–1"),
}

_FEAT_COLORS = {
    "Variance":          "#4472C4",
    "Skewness":          "#ED7D31",
    "Kurtosis (excess)": "#A9D18E",
    "Spectral Centroid": "#9B59B6",
    "Spectral Roll-off": "#E74C3C",
    "Spectral Flatness": "#1ABC9C",
    "Coherence":         "#F39C12",
    "LFCC":              "#95A5A6",
}


def _feat_label(key: str) -> "tuple[str, str]":
    """
    Return (readable_label, colour) for a feature key.

    Examples
    --------
    "acc.x.var"            → "acc.x  Variance  [g²]",          blue
    "gyro.z.centroid"      → "gyro.z  Spectral Centroid  [Hz]", purple
    "acc.x.lfcc03"         → "acc.x  LFCC-03  [log-energy]",   grey
    "coh.acc.x__gyro.x"   → "Coherence  acc.x – gyro.x  [0–1]", amber
    """
    if key.startswith("coh."):
        pair  = key[4:].replace("__", " – ")
        return f"Coherence  {pair}  [0–1]", _FEAT_COLORS["Coherence"]

    parts = key.split(".")
    if len(parts) >= 3:
        sensor = parts[0]          # "acc" | "gyro"
        axis   = parts[1]          # "x" | "y" | "z"
        feat   = ".".join(parts[2:])
        ch     = f"{sensor}.{axis}"
        is_acc = sensor == "acc"

        if feat.startswith("lfcc"):
            n = feat[4:]
            return f"{ch}  LFCC-{n}  [log-energy]", _FEAT_COLORS["LFCC"]

        if feat in _FEAT_META:
            name, u_acc, u_gyro = _FEAT_META[feat]
            unit = u_acc if is_acc else u_gyro
            return f"{ch}  {name}  [{unit}]", _FEAT_COLORS.get(name, "#888")

    return key, "#888"


def feature_bar_figure(
    feature_dict: dict,
    show_lfcc: bool = False,
    max_features: "int | None" = None,
) -> go.Figure:
    """
    Horizontal bar chart with readable labels, units, and category colours.

    Statistical features  — Variance [g² / (°/s)²], Skewness, Kurtosis
    Spectral features     — Centroid [Hz], Roll-off [Hz], Flatness [0–1]
    Coherence             — 0–1 mean squared coherence between axis pairs
    LFCC (opt.)           — log-energy cepstral coefficients (hidden by default)

    Each bar is coloured by feature type for quick visual grouping.
    max_features limits the number of bars shown (None = no limit).
    """
    if not feature_dict:
        return _empty_fig("No features yet.")

    skip = {"window_start_s", "phase"}
    rows = []
    for k, v in feature_dict.items():
        if k in skip or not isinstance(v, (int, float)):
            continue
        if not show_lfcc and "lfcc" in k:
            continue
        label, colour = _feat_label(k)
        rows.append((label, float(v), colour))

    if max_features is not None:
        rows = rows[:max_features]

    if not rows:
        return _empty_fig("No features to display.")

    labels, values, colors = zip(*rows)

    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(labels),
        orientation="h",
        marker_color=list(colors),
        hovertemplate="%{y}: %{x:.5f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(300, len(rows) * 22),
        margin=dict(l=10, r=30, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="value",
        yaxis=dict(tickfont_size=10, automargin=True),
    )
    return fig


# ── Panel 3: Manifold & Authentication State ──────────────────────────────────

def manifold_figure(
    model,
    traj_emb:   "np.ndarray | None" = None,
    traj_auth:  "list[bool] | None" = None,
    current_emb: "np.ndarray | None" = None,
    current_auth: "bool | None"      = None,
) -> go.Figure:
    """
    2-D scatter plot of the latent space.

    Layers (back to front):
      1. Threshold ellipse  — Mahalanobis distance = model.threshold_
      2. Enrollment cluster — grey dots from model.embedding_
      3. Live trajectory    — recent embeddings coloured by recency (blue fade)
      4. Current point      — large marker, green (auth) or red (rejected)

    Parameters
    ----------
    model        : fitted ManifoldModel; if None shows a placeholder.
    traj_emb     : (N, 2) array of recent live embeddings.
    traj_auth    : list of bool, one per row in traj_emb.
    current_emb  : (2,) array for the most recent embedding.
    current_auth : bool — True = authenticated.
    """
    if model is None or model.embedding_ is None:
        return _empty_fig("Load a model to see the manifold.")

    emb_e = model.embedding_  # (n_enroll, 2)
    fig   = go.Figure()

    # ── 1. Threshold ellipse ─────────────────────────────────────────────────
    try:
        ex, ey = _mahalanobis_ellipse(
            model.baseline_mean_, model.baseline_cov_, model.threshold_
        )
        fig.add_trace(go.Scatter(
            x=np.append(ex, ex[0]), y=np.append(ey, ey[0]),
            mode="lines",
            name="auth boundary",
            line=dict(color="rgba(44,160,44,0.6)", width=1.5, dash="dash"),
            fill="toself",
            fillcolor="rgba(44,160,44,0.05)",
            hoverinfo="skip",
        ))
    except Exception:
        pass

    # ── 2. Enrollment cluster ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=emb_e[:, 0], y=emb_e[:, 1],
        mode="markers",
        name="enrollment",
        marker=dict(color=_ENROLL_CLR, size=4, symbol="circle"),
        hovertemplate="enrollment<extra></extra>",
    ))

    # ── 3. Live trajectory ───────────────────────────────────────────────────
    if traj_emb is not None and len(traj_emb) > 0:
        t = np.asarray(traj_emb)
        n = len(t)
        auth_arr = list(traj_auth) if traj_auth else [True] * n
        dot_colors = [_AUTH_GREEN if a else _AUTH_RED for a in auth_arr]
        # Faded trajectory line
        fig.add_trace(go.Scatter(
            x=t[:, 0], y=t[:, 1],
            mode="lines",
            name="trajectory",
            line=dict(color="rgba(100,100,200,0.4)", width=1),
            hoverinfo="skip",
        ))
        # Trajectory dots coloured by auth state
        fig.add_trace(go.Scatter(
            x=t[:, 0], y=t[:, 1],
            mode="markers",
            name="live",
            marker=dict(color=dot_colors, size=5, opacity=0.7),
            hovertemplate="live point<extra></extra>",
        ))

    # ── 4. Current point ─────────────────────────────────────────────────────
    if current_emb is not None:
        cx, cy   = float(current_emb[0]), float(current_emb[1])
        c_colour = _AUTH_GREEN if current_auth else _AUTH_RED
        c_symbol = "circle" if current_auth else "x"
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers",
            name="current",
            marker=dict(color=c_colour, size=14, symbol=c_symbol,
                        line=dict(color="white", width=2)),
            hovertemplate=f"Mahalanobis: current<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="latent dim 1",
        yaxis_title="latent dim 2",
        height=400,
        margin=dict(l=50, r=20, t=10, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Panel 4: Offline authentication time-series ───────────────────────────────

def auth_timeseries_figure(
    distances:  "np.ndarray",
    auth_flags: "np.ndarray",
    threshold:  float,
    window_times: "np.ndarray | None" = None,
) -> go.Figure:
    """
    Mahalanobis distance over time for an offline authentication session.

    Each window is a dot coloured green (authenticated) or red (rejected).
    A horizontal dashed line marks the threshold.

    Parameters
    ----------
    distances    : (N,) Mahalanobis distances, one per feature window.
    auth_flags   : (N,) bool array — True = authenticated.
    threshold    : enrollment threshold value.
    window_times : (N,) window start times in seconds (optional; uses index if None).
    """
    n    = len(distances)
    x    = window_times if window_times is not None else np.arange(n)
    clrs = [_AUTH_GREEN if a else _AUTH_RED for a in auth_flags]

    fig = go.Figure()

    # Distance trace
    fig.add_trace(go.Scatter(
        x=x, y=distances,
        mode="lines+markers",
        name="Mahalanobis dist",
        line=dict(color="rgba(100,100,200,0.5)", width=1),
        marker=dict(color=clrs, size=6),
        hovertemplate="t=%{x:.1f}s  d=%{y:.3f}<extra></extra>",
    ))

    # Threshold line
    fig.add_hline(
        y=threshold,
        line=dict(color=_AUTH_GREEN, width=1.5, dash="dash"),
        annotation_text=f"threshold ({threshold:.3f})",
        annotation_position="top right",
        annotation_font_size=11,
    )

    pass_pct = float(auth_flags.mean()) * 100
    fig.update_layout(
        title=dict(
            text=f"Authentication result — {pass_pct:.1f}% windows PASS",
            font=dict(size=13),
            x=0.0,
        ),
        xaxis_title="window start (s)",
        yaxis_title="Mahalanobis distance",
        height=280,
        margin=dict(l=50, r=20, t=50, b=40),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def manifold_overlay_figure(
    model,
    test_emb:  "np.ndarray | None" = None,
    test_auth: "list[bool] | None" = None,
    label: str = "test session",
) -> go.Figure:
    """
    Manifold figure with enrollment cluster + offline test session overlay.

    Parameters
    ----------
    model     : fitted ManifoldModel with enrollment embedding_.
    test_emb  : (N, 2) UMAP embeddings of the test session.
    test_auth : bool list, one per row in test_emb.
    label     : legend label for test points.
    """
    if model is None or model.embedding_ is None:
        return _empty_fig("Build a model first.")

    emb_e = model.embedding_
    fig   = go.Figure()

    # Threshold ellipse
    try:
        ex, ey = _mahalanobis_ellipse(
            model.baseline_mean_, model.baseline_cov_, model.threshold_
        )
        fig.add_trace(go.Scatter(
            x=np.append(ex, ex[0]), y=np.append(ey, ey[0]),
            mode="lines",
            name="auth boundary",
            line=dict(color="rgba(44,160,44,0.6)", width=1.5, dash="dash"),
            fill="toself",
            fillcolor="rgba(44,160,44,0.05)",
            hoverinfo="skip",
        ))
    except Exception:
        pass

    # Enrollment cluster
    fig.add_trace(go.Scatter(
        x=emb_e[:, 0], y=emb_e[:, 1],
        mode="markers",
        name="enrollment",
        marker=dict(color=_ENROLL_CLR, size=5, symbol="circle"),
        hovertemplate="enrollment<extra></extra>",
    ))

    # Test session overlay
    if test_emb is not None and len(test_emb) > 0:
        t      = np.asarray(test_emb)
        auths  = list(test_auth) if test_auth else [True] * len(t)
        clrs   = [_AUTH_GREEN if a else _AUTH_RED for a in auths]
        fig.add_trace(go.Scatter(
            x=t[:, 0], y=t[:, 1],
            mode="markers",
            name=label,
            marker=dict(color=clrs, size=7, symbol="diamond",
                        line=dict(color="white", width=0.8)),
            hovertemplate=f"{label}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="latent dim 1",
        yaxis_title="latent dim 2",
        height=380,
        margin=dict(l=50, r=20, t=10, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── helpers ───────────────────────────────────────────────────────────────────

def _mahalanobis_ellipse(
    mean: np.ndarray,
    cov:  np.ndarray,
    threshold: float,
    n_pts: int = 120,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Parametric ellipse at Mahalanobis distance = threshold.

    Returns (x_coords, y_coords) arrays for plotting.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    a       = threshold * np.sqrt(eigvals[0])
    b       = threshold * np.sqrt(eigvals[1])
    theta   = np.linspace(0, 2 * np.pi, n_pts)
    pts     = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
    rotated = pts @ eigvecs.T
    return mean[0] + rotated[:, 0], mean[1] + rotated[:, 1]


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#888"),
    )
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
