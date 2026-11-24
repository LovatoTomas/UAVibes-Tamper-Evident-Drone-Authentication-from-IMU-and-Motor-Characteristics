"""
UAVibes — Session manifest for the micro-recording protocol (2026-09, new
fixed-base rig, drone A/B comparison).
=============================================================================

One excitation stage (a single motor, a pair, the combined sweep) is now
its own short, independent SD recording instead of one continuous multi-
phase session — a dropped connection during "M2 hold" now only costs that
one micro-recording, not the whole ~5 minute protocol.

That only pays off if the raw SD files uploaded afterwards can be matched
back to what was recorded WITHOUT the user re-typing which-is-which by
hand. This module is the record of what was attempted: every micro-
recording START is logged here — including ones later interrupted by a
lost connection — as a MicroAttempt with its own PC-side timeline. A batch
of uploaded logNN files is then matched, oldest-attempt to oldest-file, in
order; if the counts don't line up (extra files from an interrupted-then-
retried attempt, say) that's surfaced to the user rather than guessed at
silently.

One JSON manifest per (drone, session), e.g.
  data/fixed_rig/droneA/session_03.json
"""

import re
import json
import dataclasses
from datetime import datetime
from pathlib import Path

ATTEMPT_STATUSES = ("recording", "completed", "interrupted", "matched", "invalid", "lost",
                    "discarded")
# "lost": the attempt happened (its timeline exists) but its SD data no longer
# exists anywhere — never matched against future uploads, must be re-recorded.
# "discarded": thrown away by the operator from the dashboard (2026-09-27)
# before upload — the stage counts as "to do" again. Its SD file (if logging
# ever started) is still on the card and still consumes its slot in the
# upload FIFO, so later files keep matching the right attempts; it is stored
# under raw/discarded/ and never converted.

# ── folder / filename conventions ───────────────────────────────────────────

DEFAULT_RIG_DIR = Path("data") / "fixed_rig"
# Held-out TEST sessions (validating an identity built from the enrollment
# sessions above) live in a SIBLING tree, not under fixed_rig/: a recursive
# glob over data/fixed_rig/ while building an identity must never be able to
# pick test data up by accident. Same droneA|droneB/{raw,timelines,converted}
# layout, sessions numbered independently per drone and per kind.
DEFAULT_TEST_RIG_DIR = Path("data") / "fixed_rig_test"

# Motor-SWAP experiment (2026-09-30): A's and B's motors exchanged, to tell
# whether the identity follows the chassis or the motors. Its own sibling
# tree, and — unlike enroll/test, which are laid out per drone — ONE FOLDER
# PER SESSION holding everything that session produced:
#   data/fixed_rig_swap/s03_swap_frameA_motorsB/
#       session.json  raw/  timelines/  converted/  analysis/
# The folder name carries the configuration (stage, frame = the physical
# Crazyflie with its IMU/SD deck, motors = whose motors are mounted), so a
# folder is self-describing even when copied elsewhere.
DEFAULT_SWAP_RIG_DIR = Path("data") / "fixed_rig_swap"
SWAP_STAGES = ("pre", "swap", "post")   # before the swap / swapped / motors put back

# BOARD sessions (2026-10-03): the IMU at rest, motors off — one session = one
# boot of the drone; one folder per session, like the swap experiment:
#   data/fixed_rig_board/s07_droneC/{session.json, raw/, timelines/, converted/, analysis/}
DEFAULT_BOARD_RIG_DIR = Path("data") / "fixed_rig_board"

# BOOT sessions (2026-10-09): the key-generation boot protocol (rest + each
# motor at 58k for a few seconds), one folder per session like board sessions.
DEFAULT_BOOT_RIG_DIR = Path("data") / "fixed_rig_boot"

# CONDITION sessions (2026-10-14): the same boot protocol run on a drone that
# was deliberately altered after enrollment — propellers changed ("props") or
# a payload added ("payload"). Separate trees, like test data, so a glob over
# fixed_rig_boot/ (enrollment, analyses) can never pick an altered boot up.
# What was changed is stored in SessionManifest.condition.
DEFAULT_PROPS_RIG_DIR = Path("data") / "fixed_rig_props"
DEFAULT_PAYLOAD_RIG_DIR = Path("data") / "fixed_rig_payload"

KINDS = ("enroll", "test", "swap", "board", "boot", "props", "payload")
_KIND_TAG = {"enroll": "s", "test": "t", "swap": "w", "board": "b", "boot": "a",   # a = "avvio" (boot)
             "props": "e", "payload": "p"}                                         # e = "eliche" (propellers)
# one folder per session, one session = one boot, board params read at boot
BOOT_LIKE_KINDS = ("board", "boot", "props", "payload")
CONDITION_KINDS = ("props", "payload")


def rig_dir_for(kind: str) -> Path:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return {"enroll": DEFAULT_RIG_DIR, "test": DEFAULT_TEST_RIG_DIR,
            "swap": DEFAULT_SWAP_RIG_DIR, "board": DEFAULT_BOARD_RIG_DIR,
            "boot": DEFAULT_BOOT_RIG_DIR, "props": DEFAULT_PROPS_RIG_DIR,
            "payload": DEFAULT_PAYLOAD_RIG_DIR}[kind]


def board_session_dir(session: int, drone: str, rig_dir: "Path | str | None" = None) -> Path:
    """Folder of one board session (an existing one for this number is reused)."""
    rig_dir = Path(rig_dir or DEFAULT_BOARD_RIG_DIR)
    existing = find_swap_session(session, rig_dir)
    return existing if existing is not None else rig_dir / f"s{session:02d}_drone{drone}"


def validate_swap_config(stage: str, frame: str, motors: str) -> None:
    """pre/post: the frame carries its own motors; swap: the other drone's."""
    if stage not in SWAP_STAGES:
        raise ValueError(f"stage must be one of {SWAP_STAGES}, got {stage!r}")
    if frame not in ("A", "B") or motors not in ("A", "B"):
        raise ValueError("the swap experiment uses drones A and B only")
    if (stage == "swap") != (frame != motors):
        raise ValueError(f"stage {stage!r} with frame {frame} and motors {motors} is inconsistent")


def swap_session_dir(session: int, stage: str, frame: str, motors: str,
                     rig_dir: "Path | str | None" = None) -> Path:
    """Folder of one swap session. An existing folder for this session number
    is returned as-is (its configuration is checked by load_or_new)."""
    rig_dir = Path(rig_dir or DEFAULT_SWAP_RIG_DIR)
    existing = find_swap_session(session, rig_dir)
    if existing is not None:
        return existing
    validate_swap_config(stage, frame, motors)
    return rig_dir / f"s{session:02d}_{stage}_frame{frame}_motors{motors}"


def find_swap_session(session: int, rig_dir: "Path | str | None" = None) -> "Path | None":
    hits = sorted(Path(rig_dir or DEFAULT_SWAP_RIG_DIR).glob(f"s{session:02d}_*"))
    return hits[0] if hits else None


def swap_session_dirs(session_dir: "Path | str") -> "dict[str, Path]":
    """Same keys as drone_dirs(), plus 'analysis' for per-session results."""
    root = Path(session_dir)
    return {k: root / k for k in ("raw", "timelines", "converted", "analysis")}


def drone_dirs(rig_dir: "Path | str", drone: str) -> "dict[str, Path]":
    """{'raw', 'timelines', 'converted'} -> Path, under <rig_dir>/drone<A|B>/."""
    root = Path(rig_dir) / f"drone{drone}"
    return {kind: root / kind for kind in ("raw", "timelines", "converted")}


def manifest_path(rig_dir: "Path | str", drone: str, session: int) -> Path:
    return Path(rig_dir) / f"drone{drone}" / f"session_{session:02d}.json"


def micro_stem(drone: str, session: int, purpose: str,
               ts: "datetime | None" = None, kind: str = "enroll",
               motors: "str | None" = None, stage: "str | None" = None) -> str:
    """Filename stem encoding date/time, drone, session, purpose — the
    naming scheme every renamed/generated file for this protocol uses.
    Test sessions use _t03_ instead of _s03_, so a file stays self-describing
    even if it is ever moved out of its folder. Swap sessions spell out the
    configuration: {ts}_frameA_motorsB_swap_w03_{purpose}."""
    ts = ts or datetime.now()
    if kind == "swap":
        return f"{ts:%Y%m%d_%H%M%S}_frame{drone}_motors{motors}_{stage}_w{session:02d}_{purpose}"
    return f"{ts:%Y%m%d_%H%M%S}_drone{drone}_{_KIND_TAG[kind]}{session:02d}_{purpose}"


# ── data model ────────────────────────────────────────────────────────────

@dataclasses.dataclass
class MicroAttempt:
    """One start-to-(stop|interruption) of a single micro-recording."""
    attempt_id: str                    # e.g. "20260919_143022" — unique per attempt
    timeline_path: str
    status: str = "recording"          # see ATTEMPT_STATUSES
    raw_log_path: "str | None" = None
    converted_path: "str | None" = None
    created_iso: str = ""
    sd_log_name: "str | None" = None   # original SD filename (logNN) — the
                                       # card's numbering restarts at 00 every
                                       # time it's emptied, so it identifies
                                       # nothing on its own; kept for audit only
    vbat_rest: "float | None" = None   # resting battery voltage read just before
                                       # the stage (battery gate, 2026-09-30)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MicroAttempt":
        return cls(**d)


@dataclasses.dataclass
class MicroRecording:
    """
    One planned excitation stage (e.g. "sweep-m2", "sweep-diag13",
    "sweep-all4") — may have several attempts if earlier ones were
    interrupted by a lost connection and retried.
    """
    purpose: str
    pwm: int
    attempts: "list[MicroAttempt]" = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return {"purpose": self.purpose, "pwm": self.pwm,
                "attempts": [a.to_dict() for a in self.attempts]}

    @classmethod
    def from_dict(cls, d: dict) -> "MicroRecording":
        return cls(purpose=d["purpose"], pwm=d["pwm"],
                    attempts=[MicroAttempt.from_dict(a) for a in d.get("attempts", [])])

    @property
    def needs_recording(self) -> bool:
        """No attempts yet, or the newest one was discarded by the operator
        — the stage is back in the "to do" queue."""
        return not self.attempts or (self.attempts[-1].status == "discarded"
                                     and self.accepted_attempt is None)

    @property
    def accepted_attempt(self) -> "MicroAttempt | None":
        """The attempt considered good for pooling (newest completed/matched
        one with a converted CSV), or None if this stage has no valid data
        yet — interrupted attempts are never returned here, but stay in
        .attempts for visibility (nothing is silently dropped)."""
        for a in reversed(self.attempts):
            if a.status in ("matched", "completed") and a.converted_path:
                return a
        return None


@dataclasses.dataclass
class SessionManifest:
    drone: str                          # "A" or "B"
    session: int
    created_iso: str
    recordings: "list[MicroRecording]" = dataclasses.field(default_factory=list)
    kind: str = "enroll"                # "enroll" | "test" | "swap" — see KINDS
    # swap sessions only: `drone` is the FRAME (the physical Crazyflie, its
    # IMU and SD deck); `motors` is whose motors are mounted on it
    motors: "str | None" = None
    swap_stage: "str | None" = None     # see SWAP_STAGES
    # board sessions only: values the firmware computed at THIS boot (param
    # group "uavibe": accel self-test, gyro bias, accel scale) + board
    # temperature and battery, read by the dashboard before the rest stage
    board_params: "dict | None" = None
    # props/payload sessions only: what was altered on the drone (propeller
    # type and positions, payload mass and position, free notes)
    condition: "dict | None" = None

    # ── persistence ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = {
            "drone": self.drone, "session": self.session, "kind": self.kind,
            "created_iso": self.created_iso,
            "recordings": [r.to_dict() for r in self.recordings],
        }
        if self.kind == "swap":
            d.update(frame=self.drone, motors=self.motors, swap_stage=self.swap_stage)
        if self.kind in BOOT_LIKE_KINDS:
            d["board_params"] = self.board_params
        if self.kind in CONDITION_KINDS:
            d["condition"] = self.condition
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SessionManifest":
        return cls(
            drone=d["drone"], session=d["session"], created_iso=d["created_iso"],
            kind=d.get("kind", "enroll"),
            motors=d.get("motors"), swap_stage=d.get("swap_stage"),
            board_params=d.get("board_params"), condition=d.get("condition"),
            recordings=[MicroRecording.from_dict(r) for r in d.get("recordings", [])],
        )

    def save(self, path: "Path | str") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: "Path | str") -> "SessionManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_or_new(cls, path: "Path | str", drone: str, session: int,
                    kind: str = "enroll", motors: "str | None" = None,
                    swap_stage: "str | None" = None) -> "SessionManifest":
        path = Path(path)
        if path.exists():
            m = cls.load(path)
            if m.kind != kind:
                raise ValueError(f"{path} is a {m.kind!r} manifest, not {kind!r} — "
                                 "enrollment and test sessions must never share a file")
            if kind in BOOT_LIKE_KINDS and m.drone != drone:
                raise ValueError(f"board session {session} belongs to drone {m.drone}, not {drone}")
            if kind == "swap" and (m.drone, m.motors, m.swap_stage) != (drone, motors, swap_stage):
                raise ValueError(
                    f"swap session {session} was created as {m.swap_stage}, frame {m.drone}, "
                    f"motors {m.motors} — not {swap_stage}, frame {drone}, motors {motors}")
            return m
        if kind == "swap":
            validate_swap_config(swap_stage, drone, motors)
        return cls(drone=drone, session=session, kind=kind, motors=motors, swap_stage=swap_stage,
                   created_iso=datetime.now().isoformat(timespec="seconds"))

    # ── attempt tracking ─────────────────────────────────────────────────

    def start_attempt(self, purpose: str, pwm: int, timeline_path: "Path | str") -> MicroAttempt:
        """
        Record a new attempt for `purpose` (creating the MicroRecording on
        its first attempt). Call this BEFORE running the micro-recording
        and save() right after, so even an attempt that never reaches
        usd_start (connection died mid-setup) is still on record.
        """
        rec = next((r for r in self.recordings if r.purpose == purpose), None)
        if rec is None:
            rec = MicroRecording(purpose=purpose, pwm=pwm)
            self.recordings.append(rec)
        now = datetime.now()
        attempt = MicroAttempt(
            # Microsecond precision (not just seconds): all_attempts() sorts
            # by attempt_id to recover creation order, and two attempts can
            # legitimately start within the same second (a connection that
            # dies before usd_start, immediately retried).
            attempt_id=now.strftime("%Y%m%d_%H%M%S_%f"),
            timeline_path=str(timeline_path),
            created_iso=now.isoformat(timespec="seconds"),
        )
        rec.attempts.append(attempt)
        return attempt

    def find_attempt(self, attempt_id: str) -> "MicroAttempt | None":
        for r in self.recordings:
            for a in r.attempts:
                if a.attempt_id == attempt_id:
                    return a
        return None

    def all_attempts(self) -> "list[MicroAttempt]":
        """Every attempt across every recording, oldest first."""
        out = [a for r in self.recordings for a in r.attempts]
        out.sort(key=lambda a: a.attempt_id)
        return out

    def pending_upload_attempts(self) -> "list[MicroAttempt]":
        """Attempts that should have produced an SD file not uploaded yet,
        oldest first — the FIFO queue a batch upload is matched against.
        Includes discarded attempts (their file is still on the card) but
        not ones that never reached usd_start (no file was ever created)."""
        return [a for a in self.all_attempts()
                if a.raw_log_path is None and a.status != "lost"
                and sd_file_expected(a)]

    def discard_attempt(self, attempt_id: str) -> MicroAttempt:
        """Mark a not-yet-uploaded attempt as discarded (see ATTEMPT_STATUSES).
        Nothing is deleted: the timeline stays, and the SD file is still
        expected at upload so the FIFO matching doesn't shift."""
        a = self.find_attempt(attempt_id)
        if a is None:
            raise KeyError(attempt_id)
        if a.raw_log_path is not None:
            raise ValueError("already uploaded — only attempts awaiting upload can be discarded")
        a.status = "discarded"
        return a

    def match_uploaded_batch(
        self, log_paths: "list[Path]",
    ) -> "tuple[list[tuple[MicroAttempt, Path]], list[Path]]":
        """
        Pair uploaded raw logNN files against pending attempts, oldest-
        attempt-to-oldest-file — log_paths must already be sorted by the SD
        card's own log id (list_sd_logs does this; the Crazyflie has no RTC
        so log id order is the only reliable ordering available).

        Returns (matched pairs, leftover files). Leftover is non-empty when
        there are more uploaded files than pending attempts — an
        interrupted-then-retried stage can leave an extra file the PC side
        never logged an attempt for — surfaced here rather than guessed.
        Pending attempts left unmatched (fewer files than expected) stay in
        pending_upload_attempts() for the caller to show as still missing.
        """
        pending = self.pending_upload_attempts()
        n = min(len(pending), len(log_paths))
        matched = list(zip(pending[:n], log_paths[:n]))
        leftover = list(log_paths[n:])
        return matched, leftover


# ── upload ingestion (never overwrites) ─────────────────────────────────────

def sd_file_expected(attempt: MicroAttempt) -> bool:
    """False only when the attempt's timeline exists and shows SD logging was
    never started (connection died between start_attempt and usd_start) —
    then no logNN file exists for it. A missing timeline: assume a file
    exists (conservative, the pre-2026-09-27 behaviour)."""
    p = Path(attempt.timeline_path)
    if not p.exists():
        return True
    try:
        return ",usd_start," in p.read_text(encoding="utf-8")
    except OSError:
        return True


def unique_path(path: "Path | str") -> Path:
    """`path` if it doesn't exist, else `stem_2.ext`, `stem_3.ext`... —
    nothing under data/fixed_rig/ may ever be overwritten."""
    path = Path(path)
    if not path.exists():
        return path
    n = 2
    while True:
        cand = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not cand.exists():
            return cand
        n += 1


def _sd_log_id(name: str) -> float:
    m = re.match(r"^log(\d+)$", name)
    return int(m.group(1)) if m else float("inf")


def ingest_upload_batch(
    manifest: SessionManifest,
    manifest_file: "Path | str",
    uploads: "list[tuple[str, bytes]]",
    dirs: "dict[str, Path]",
    convert_fn,
) -> "tuple[list[tuple[str, bool, str]], list[Path]]":
    """
    Match a batch of uploaded raw SD files to this session's pending attempts
    and store/convert them under names derived from the ATTEMPT, never from
    the uploaded filename.

    The card's own numbering (log00, log01, ...) restarts every time the card
    is emptied, so the same names recur in every session — using them as
    on-disk names is exactly how session 1's raw and converted files were
    overwritten by session 2's (2026-09-20). Files are instead named after
    the attempt's timeline stem ({date}_{time}_drone{X}_s{NN}_{purpose}):
    raw/<stem>.raw and converted/<stem>_1khz.csv. Every write goes through
    unique_path(), so even a name clash cannot destroy an existing file.

    uploads: [(original_filename, bytes)] in any order (sorted here by SD
    log id). convert_fn(raw_path, timeline_path) -> (DataFrame, stats), i.e.
    usd_convert.convert. Uploads with no pending attempt to match are kept
    under raw/unmatched/ (session- and time-prefixed), not dropped and not
    stored under their bare name.

    Returns (results, leftover_paths); results = [(original_name, ok, msg)].
    The manifest is saved after every file, so a crash mid-batch loses
    nothing already stored.
    """
    dirs["raw"].mkdir(parents=True, exist_ok=True)
    dirs["converted"].mkdir(parents=True, exist_ok=True)
    ordered = sorted(uploads, key=lambda u: _sd_log_id(u[0]))
    matched, leftover = manifest.match_uploaded_batch(ordered)

    results: "list[tuple[str, bool, str]]" = []
    for attempt, (orig_name, data) in matched:
        stem = Path(attempt.timeline_path).stem
        if attempt.status == "discarded":
            # consume its slot, keep the bytes for audit, never convert/pool
            disc_dir = dirs["raw"] / "discarded"
            disc_dir.mkdir(parents=True, exist_ok=True)
            raw_path = unique_path(disc_dir / f"{stem}.raw")
            raw_path.write_bytes(data)
            attempt.raw_log_path = str(raw_path)
            attempt.sd_log_name = orig_name
            results.append((orig_name, True, f"discarded recording -> raw/discarded/{raw_path.name} (not converted)"))
            manifest.save(manifest_file)
            continue
        raw_path = unique_path(dirs["raw"] / f"{stem}.raw")
        raw_path.write_bytes(data)
        attempt.raw_log_path = str(raw_path)
        attempt.sd_log_name = orig_name
        try:
            df, stats = convert_fn(raw_path, Path(attempt.timeline_path))
            out_csv = unique_path(dirs["converted"] / f"{stem}_1khz.csv")
            df.to_csv(out_csv, index=False)
            attempt.converted_path = str(out_csv)
            attempt.status = "matched"
            results.append((orig_name, True, f"{stats['n_samples']:,} campioni -> {out_csv.name}"))
        except Exception as exc:
            attempt.status = "invalid"
            results.append((orig_name, False, str(exc)))
        manifest.save(manifest_file)

    leftover_paths: "list[Path]" = []
    if leftover:
        unmatched_dir = dirs["raw"] / "unmatched"
        unmatched_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for orig_name, data in leftover:
            p = unique_path(unmatched_dir / f"{_KIND_TAG[manifest.kind]}{manifest.session:02d}_{stamp}_{orig_name}")
            p.write_bytes(data)
            leftover_paths.append(p)
    manifest.save(manifest_file)
    return results, leftover_paths
