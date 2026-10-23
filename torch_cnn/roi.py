"""Region-of-interest markers for tools/opcount_roi.so.

begin()/end() issue getppid(MAGIC, 1|2); the plugin snapshots its counters on
these. getppid ignores its arguments, so the calls are harmless without the
plugin. They are no-ops when not running as riscv64 (the syscall number differs).
end() also copies /proc/self/maps to $ROI_MAPS so the plugin can attribute
addresses to shared libraries.
"""
import ctypes
import os
import platform

_MAGIC = 0x524F49
_SYS_GETPPID = 173  # riscv64
_ACTIVE = platform.machine() == "riscv64"
_libc = ctypes.CDLL(None) if _ACTIVE else None


def begin():
    if _ACTIVE:
        _libc.syscall(_SYS_GETPPID, _MAGIC, 1)


def end():
    if _ACTIVE:
        _libc.syscall(_SYS_GETPPID, _MAGIC, 2)
        if path := os.environ.get("ROI_MAPS"):
            with open("/proc/self/maps") as src, open(path, "w") as dst:
                dst.write(src.read())
