"""
Connection helpers for Crazyflie 2.1 — USB direct and CrazyRadio.

USB
  scan()              → list of (uri, name) via USB
  get_first_uri()     → URI of first USB device found
  open_sync(uri)      → SyncCrazyflie context manager (works for any URI)

CrazyRadio
  radio_uri()         → build a radio URI without scanning
  scan_radio()        → scan all channels for radio devices (slow, ~10 s)
  get_first_radio_uri() → URI of first radio device found, or None

Windows libusb1 patches (applied at import time)
  Both cflib.drivers.cfusb (USB direct) and cflib.drivers.crazyradio
  hard-code the libusb0 backend on Windows, but libusb-package ships
  libusb-1.0.dll (libusb1) — different ABI.  Both _find_devices()
  functions are monkey-patched to use libusb1 instead.
"""

import os
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# ── Windows libusb1 patch (must run before any cflib USB call) ────────────────

def _apply_windows_libusb_patch() -> None:
    if os.name != 'nt':
        return
    import libusb_package
    import usb.core
    import cflib.drivers.cfusb as _cfusb

    def _find():
        ret = []
        backend = libusb_package.get_libusb1_backend()
        devices = usb.core.find(
            idVendor=_cfusb.USB_VID,
            idProduct=_cfusb.USB_PID,
            find_all=True,
            backend=backend,
        )
        if devices:
            for d in devices:
                try:
                    if d.manufacturer == 'Bitcraze AB':
                        ret.append(d)
                except Exception:
                    ret.append(d)
        return ret

    _cfusb._find_devices = _find


_apply_windows_libusb_patch()


def _apply_windows_radio_libusb_patch() -> None:
    if os.name != 'nt':
        return
    import libusb_package
    import usb.core
    import cflib.drivers.crazyradio as _cr

    def _find(serial=None):
        ret = []
        backend = libusb_package.get_libusb1_backend()
        devices = usb.core.find(
            idVendor=_cr.CRADIO_VID,
            idProduct=_cr.CRADIO_PID,
            find_all=True,
            backend=backend,
        )
        if devices:
            for d in devices:
                if serial is not None and serial == d.serial_number:
                    return d
                ret.append(d)
        return ret

    _cr._find_devices = _find


_apply_windows_radio_libusb_patch()

# ── USB helpers ───────────────────────────────────────────────────────────────

def scan() -> list[tuple[str, str]]:
    """Init cflib drivers and return list of (uri, name) for USB devices."""
    cflib.crtp.init_drivers(enable_debug_driver=False)
    return cflib.crtp.scan_interfaces()


def get_first_uri() -> str | None:
    """Return URI of the first detected USB Crazyflie, or None."""
    devices = scan()
    return devices[0][0] if devices else None


def open_sync(uri: str) -> SyncCrazyflie:
    """Return a SyncCrazyflie context manager for the given URI (USB or radio)."""
    cflib.crtp.init_drivers(enable_debug_driver=False)
    return SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache'))


# ── CrazyRadio helpers ────────────────────────────────────────────────────────

_RADIO_DEFAULT_ADDRESS = 0xE7E7E7E7E7


def radio_uri(channel: int = 80, datarate: str = '2M',
              address: str = 'E7E7E7E7E7') -> str:
    """Build a CrazyRadio URI without scanning. Example: radio://0/80/2M/E7E7E7E7E7"""
    return f'radio://0/{channel}/{datarate}/{address}'


def scan_radio(address: int = _RADIO_DEFAULT_ADDRESS) -> list[tuple[str, str]]:
    """Scan all radio channels for Crazyflie devices. May take several seconds."""
    cflib.crtp.init_drivers(enable_debug_driver=False)
    all_devices = cflib.crtp.scan_interfaces(address)
    return [(u, n) for u, n in all_devices if u.startswith('radio://')]


def get_first_radio_uri(address: int = _RADIO_DEFAULT_ADDRESS) -> str | None:
    """Return the URI of the first Crazyflie found via CrazyRadio, or None."""
    devices = scan_radio(address)
    return devices[0][0] if devices else None
