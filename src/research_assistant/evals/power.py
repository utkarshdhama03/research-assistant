"""Power state and sleep prevention (Windows, via ctypes; no-ops elsewhere)."""

import ctypes
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@dataclass(frozen=True)
class PowerState:
    on_ac: bool | None  # None: unknown
    battery_percent: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def _kernel32() -> Any:
    return getattr(ctypes, "windll").kernel32  # noqa: B009 - windll exists only on Windows


def power_state() -> PowerState:
    if sys.platform != "win32":
        return PowerState(None, None)
    status = _SystemPowerStatus()
    if not _kernel32().GetSystemPowerStatus(ctypes.byref(status)):
        return PowerState(None, None)
    on_ac = {0: False, 1: True}.get(status.ACLineStatus)
    percent = None if status.BatteryLifePercent == 255 else int(status.BatteryLifePercent)
    return PowerState(on_ac, percent)


@contextmanager
def keep_awake() -> Iterator[None]:
    """Prevent system sleep while the block runs (only this process's request; no settings change)."""
    if sys.platform != "win32":
        yield
        return
    kernel32 = _kernel32()
    kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
