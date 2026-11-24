"""
Why did the drone reset? Connects, asks the firmware to print its assert
snapshot (param system.assertInfo = 1) and shows the console for a few
seconds. Run it right after an unexpected reboot (LED M1 blinking in groups =
self-test failed after a watchdog reset or a failed assert), before
power-cycling: a power cycle clears the snapshot.

    python tools/fw_why_reset.py              # radio, channel 80, 2M
    python tools/fw_why_reset.py usb://0
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crazyflie.connection import open_sync, radio_uri   # applies the Windows libusb patch


def main() -> None:
    uri = sys.argv[1] if len(sys.argv) > 1 else radio_uri()
    lines = []
    with open_sync(uri) as scf:
        cf = scf.cf
        cf.console.receivedChar.add_callback(lambda s: (print(s, end="", flush=True), lines.append(s)))
        time.sleep(1.0)
        cf.param.set_value("system.assertInfo", "1")
        time.sleep(3.0)
        try:
            print(f"\nuvkey.enrolled = {cf.param.get_value('uvkey.enrolled')}")
        except Exception:
            print("\n(no uvkey parameters: the drone is not running FW2)")
    if not "".join(lines).strip():
        print("No console output received.")


if __name__ == "__main__":
    main()
