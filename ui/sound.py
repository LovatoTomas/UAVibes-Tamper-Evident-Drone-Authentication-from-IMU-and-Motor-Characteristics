"""
Acoustic alerts played on the PC running the dashboard (the Streamlit server
runs on the same machine the operator sits at — the rig, the radio and the
laptop are all together). Used by the New Rig Session tab's automatic mode
to call the operator back when the run finishes or stops on an error.

Played in a daemon thread so the dashboard never waits for the sound.
winsound (Windows) when available, terminal bell otherwise.
"""

import threading
import time

try:
    import winsound
except ImportError:          # not Windows
    winsound = None

# (frequency Hz, duration ms); frequency 0 = silence
_SUCCESS = [(523, 150), (659, 150), (784, 150), (1047, 400), (0, 400)] * 2
_ERROR   = [(440, 450), (0, 150), (330, 450), (0, 150), (220, 800), (0, 600)] * 3


def _play(pattern) -> None:
    for freq, ms in pattern:
        if freq == 0:
            time.sleep(ms / 1000.0)
            continue
        try:
            if winsound is not None:
                winsound.Beep(freq, ms)
            else:
                print("\a", end="", flush=True)
                time.sleep(ms / 1000.0)
        except Exception:
            # no sound device / Beep unsupported — fall back to the system sound
            try:
                if winsound is not None:
                    winsound.MessageBeep(winsound.MB_ICONHAND)
            except Exception:
                pass
            time.sleep(ms / 1000.0)


def play(ok: bool) -> None:
    """Same sounds as alert(), but waits until they end (command-line
    scripts, which may exit right after)."""
    _play(_SUCCESS if ok else _ERROR)


def alert(ok: bool) -> None:
    """Rising chime if ok, repeated falling low tones if not."""
    threading.Thread(target=_play, args=(_SUCCESS if ok else _ERROR,),
                     daemon=True).start()
