"""VRAM / RAM logging per pipeline stage."""
import logging
import subprocess

import psutil

log = logging.getLogger("ragscript")
VRAM_BUDGET_MB = 14500
_peak = 0


def vram():
    """(used_mb, total_mb) from nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=5)
        used, total = (int(x) for x in out.splitlines()[0].split(","))
        return used, total
    except Exception:
        return None


def log_stage(stage: str):
    global _peak
    ram = psutil.virtual_memory()
    v = vram()
    if v:
        _peak = max(_peak, v[0])
        msg = f"VRAM {v[0]}/{v[1]} MB (peak {_peak})"
        if v[0] > VRAM_BUDGET_MB:
            msg += f"  WARNING: over {VRAM_BUDGET_MB} MB budget"
    else:
        msg = "VRAM n/a"
    log.info("[%s] %s | RAM %.1f/%.1f GB", stage, msg, ram.used / 2**30, ram.total / 2**30)
