from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .domain import ProjectConfig


def available_memory_bytes() -> int | None:
    """Read free physical RAM without importing a monitoring dependency."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                    (name, ctypes.c_ulonglong)
                    for name in (
                        "total_physical",
                        "available_physical",
                        "total_page",
                        "available_page",
                        "total_virtual",
                        "available_virtual",
                        "available_extended",
                    )
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            probe = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
            probe.argtypes = [ctypes.POINTER(MemoryStatus)]
            probe.restype = wintypes.BOOL
            return int(status.available_physical) if probe(ctypes.byref(status)) else None
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError):
        return None


def estimate_storage(config: ProjectConfig) -> dict[str, Any]:
    config.validate(check_reference=False)
    average = (config.quality.min_duration + config.quality.max_duration) / 2
    count = max(config.target_count, math.ceil(config.target_seconds / average))
    pcm = math.ceil(count * average * 24000 * 2)
    attempts = min(config.max_attempts, max(count, count * (config.max_retries + 1)))
    working = attempts * average * 24000 * 2
    originals = attempts * average * 24000 * 4 if config.keep_original else 0
    return {
        "accepted_pcm_bytes": pcm,
        "estimated_total_bytes": math.ceil((working + originals + pcm) * 1.1),
        "estimated_average_duration": average,
        "explanation": "24 kHz mono PCM16 working audio and export; float32 originals if retained; "
        "retry attempts plus 10% metadata headroom. Model weights are excluded.",
    }


def _nvidia_smi_rows() -> list[dict] | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            shell=False,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    rows: list[dict] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or not parts[0]:
            continue
        rows.append(
            {
                "name": parts[0],
                "memory_total_mb": _to_int(parts[1]),
                "memory_free_mb": _to_int(parts[2]),
                "utilization_percent": _to_int(parts[3]),
            }
        )
    return rows or None


def _to_int(text: str) -> int | None:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def gpu_status() -> dict:
    """Report CUDA/GPU availability without raising; torch is imported lazily."""
    result: dict = {
        "cuda_available": False,
        "torch_installed": False,
        "torch_cuda_allocated_bytes": None,
        "torch_cuda_reserved_bytes": None,
        "torch_device_name": None,
        "torch_mem_get_info": None,
        "nvidia_smi": None,
        "reason": None,
    }
    try:
        from importlib.machinery import PathFinder

        spec = PathFinder.find_spec("torch")
    except Exception:
        spec = None
    if spec is None:
        result["reason"] = "torch is not installed"
    else:
        result["torch_installed"] = True
        try:
            import torch  # lazy: heavy import when torch present

            if torch.cuda.is_available():
                result["cuda_available"] = True
                result["torch_device_name"] = torch.cuda.get_device_name(0)
                result["torch_cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
                result["torch_cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
                free, total = torch.cuda.mem_get_info()
                result["torch_mem_get_info"] = {"free_bytes": int(free), "total_bytes": int(total)}
            else:
                result["reason"] = "torch reports no CUDA device"
        except Exception as exc:  # never surface from diagnostics
            result["cuda_available"] = False
            result["reason"] = f"torch CUDA probe failed: {type(exc).__name__}"
    result["nvidia_smi"] = _nvidia_smi_rows()
    if not result["cuda_available"] and result["reason"] is None:
        if result["nvidia_smi"]:
            result["reason"] = "GPU visible but torch CUDA unavailable"
        else:
            result["reason"] = "no NVIDIA GPU detected"
    return result


def _disk_free_bytes(root: Path | None) -> int | None:
    try:
        target = root if root is not None else Path.cwd()
        usage = shutil.disk_usage(target)
        return int(usage.free)
    except (OSError, ValueError):
        return None


def _dependency_versions() -> dict:
    try:
        from .export import dependency_versions  # type: ignore

        versions = dependency_versions()
        return dict(versions) if isinstance(versions, dict) else {}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: unable to load dependency versions"}


def diagnostics(root: Path | None = None) -> dict:
    """Collect environment diagnostics. May block; run from a worker thread."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "gpu": gpu_status(),
        "ffmpeg": shutil.which("ffmpeg"),
        "disk_free_bytes": _disk_free_bytes(root),
        "dependencies": _dependency_versions(),
    }
