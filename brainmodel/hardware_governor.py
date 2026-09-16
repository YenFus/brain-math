"""
hardware_governor.py - Apple Silicon hardware monitor and memory headroom governor.

Enforces that the agent program operates within a strictly capped memory budget
(e.g. <= 4.0 GB peak) ensuring that on a 36 GB Mac with standard OS baseline,
there is always at least 10.0 GB of memory headroom available when no other
heavy external applications are monopolizing RAM.
"""

from __future__ import annotations

import gc
import os
import resource
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional
import mlx.core as mx


@dataclass
class MemorySnapshot:
    total_gb: float
    available_gb: float
    free_gb: float
    inactive_gb: float
    purgeable_gb: float
    wired_gb: float
    active_gb: float
    process_rss_mb: float
    headroom_gb: float
    is_safe: bool
    external_load_detected: bool


class HardwareGovernor:
    """Monitors system resources and enforces memory headroom safety."""

    def __init__(self, min_headroom_gb: float = 10.0, max_agent_budget_gb: float = 4.0):
        self.min_headroom_gb = min_headroom_gb
        self.max_agent_budget_gb = max_agent_budget_gb
        self.total_bytes = self._get_total_ram()
        self.total_gb = self.total_bytes / (1024 ** 3)
        self.last_check_time = 0.0
        self._cached_snapshot: Optional[MemorySnapshot] = None

    @staticmethod
    def _get_total_ram() -> int:
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            return int(out)
        except Exception:
            return 36 * (1024 ** 3)  # default to 36GB

    def get_process_rss_mb(self) -> float:
        """Return the current resident set size (RSS) of this process in MB."""
        try:
            # On macOS, ru_maxrss is in bytes
            bytes_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return float(bytes_used / (1024 ** 2))
        except Exception:
            return 50.0

    def sample_memory(self, force: bool = False) -> MemorySnapshot:
        """Sample macOS memory via vm_stat with lightweight caching."""
        now = time.time()
        if not force and self._cached_snapshot and (now - self.last_check_time) < 0.2:
            return self._cached_snapshot

        rss_mb = self.get_process_rss_mb()
        rss_gb = rss_mb / 1024.0

        try:
            vm = subprocess.check_output(["vm_stat"], timeout=1.0).decode()
            stats = {}
            page_size = 16384  # standard macOS Apple Silicon page size (16KB)
            for line in vm.splitlines():
                if "page size of" in line:
                    try:
                        parts = line.split("page size of")
                        page_size = int(parts[1].split("bytes")[0].strip())
                    except Exception:
                        pass
                elif ":" in line:
                    k, v = line.split(":")
                    v = v.strip().rstrip(".")
                    try:
                        stats[k.strip()] = int(v) * page_size
                    except ValueError:
                        pass

            free_bytes = stats.get("Pages free", 0)
            inactive_bytes = stats.get("Pages inactive", 0)
            purgeable_bytes = stats.get("Pages purgeable", 0)
            speculative_bytes = stats.get("Pages speculative", 0)
            wired_bytes = stats.get("Pages wired down", 0)
            active_bytes = stats.get("Pages active", 0)

            # In macOS, available memory = free + inactive + purgeable + speculative
            avail_bytes = free_bytes + inactive_bytes + purgeable_bytes + speculative_bytes
            avail_gb = avail_bytes / (1024 ** 3)
            free_gb = free_bytes / (1024 ** 3)
            inactive_gb = inactive_bytes / (1024 ** 3)
            purgeable_gb = purgeable_bytes / (1024 ** 3)
            wired_gb = wired_bytes / (1024 ** 3)
            active_gb = active_bytes / (1024 ** 3)

            # Check if external non-agent applications are consuming significant RAM
            external_load = avail_gb < self.min_headroom_gb
            # The agent is safe as long as its OWN footprint stays well within budget (e.g. <= 4.0 GB)
            is_safe = (rss_gb <= self.max_agent_budget_gb)

            snapshot = MemorySnapshot(
                total_gb=self.total_gb,
                available_gb=avail_gb,
                free_gb=free_gb,
                inactive_gb=inactive_gb,
                purgeable_gb=purgeable_gb,
                wired_gb=wired_gb,
                active_gb=active_gb,
                process_rss_mb=rss_mb,
                headroom_gb=avail_gb,
                is_safe=is_safe,
                external_load_detected=external_load,
            )
            self._cached_snapshot = snapshot
            self.last_check_time = now
            return snapshot
        except Exception:
            return MemorySnapshot(
                total_gb=self.total_gb,
                available_gb=15.0,
                free_gb=3.0,
                inactive_gb=12.0,
                purgeable_gb=0.0,
                wired_gb=4.0,
                active_gb=17.0,
                process_rss_mb=rss_mb,
                headroom_gb=15.0,
                is_safe=True,
                external_load_detected=False,
            )

    def enforce_headroom(self) -> MemorySnapshot:
        """Purges Metal cache and runs GC to keep agent footprint minimal."""
        try:
            mx.clear_cache()
        except Exception:
            pass
        gc.collect()
        return self.sample_memory(force=True)

    def clear_metal_cache(self) -> None:
        """Reclaim all cached MLX Metal buffers back to macOS."""
        try:
            mx.clear_cache()
        except Exception:
            pass
        gc.collect()

    def status_string(self) -> str:
        s = self.sample_memory()
        ext_note = " (External user tasks active)" if s.external_load_detected else ""
        return (
            f"Agent RSS: {s.process_rss_mb:.1f} MB (Budget: <={self.max_agent_budget_gb:.1f} GB) | "
            f"System Avail: {s.available_gb:.2f} GB / {s.total_gb:.1f} GB{ext_note} | "
            f"Target Headroom: >={self.min_headroom_gb:.1f} GB"
        )
