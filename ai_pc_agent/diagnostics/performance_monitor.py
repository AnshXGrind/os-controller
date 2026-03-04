"""ai_pc_agent/diagnostics/performance_monitor.py

Track execution times, memory usage, and system health.
"""

from __future__ import annotations
import time
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import psutil

from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.perf")


@dataclass
class Measurement:
    label:      str
    elapsed_ms: float
    ts:         float = field(default_factory=time.time)


class PerformanceMonitor:
    """Lightweight performance monitoring with timing context manager."""

    def __init__(self, history_size: int = 200):
        self._measurements: deque[Measurement] = deque(maxlen=history_size)
        self._lock  = threading.Lock()
        self._alerts: list[tuple[str, float]] = []   # (label, threshold_ms)

    # ── Timing ────────────────────────────────────────────────────────────────

    @contextmanager
    def timer(self, label: str) -> Generator[None, None, None]:
        """Context manager: measures wall-clock time of a block."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            m = Measurement(label=label, elapsed_ms=elapsed)
            with self._lock:
                self._measurements.append(m)
            if elapsed > 1000:
                logger.debug("[PERF] %s: %.1f ms", label, elapsed)
            self._check_alerts(label, elapsed)

    def record(self, label: str, elapsed_ms: float):
        with self._lock:
            self._measurements.append(Measurement(label=label, elapsed_ms=elapsed_ms))

    # ── Alerts ────────────────────────────────────────────────────────────────

    def add_alert(self, label: str, threshold_ms: float):
        """Warn when *label* exceeds *threshold_ms*."""
        self._alerts.append((label, threshold_ms))

    def _check_alerts(self, label: str, elapsed: float):
        for al_label, threshold in self._alerts:
            if al_label in label and elapsed > threshold:
                logger.warning("[PERF ALERT] '%s' took %.1f ms (threshold %.0f ms)",
                               label, elapsed, threshold)

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self, label: str | None = None) -> dict:
        with self._lock:
            items = [m for m in self._measurements
                     if label is None or label in m.label]
        if not items:
            return {}
        times = [m.elapsed_ms for m in items]
        return {
            "count":  len(times),
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
        }

    def slowest(self, n: int = 5) -> list[Measurement]:
        with self._lock:
            return sorted(self._measurements, key=lambda m: m.elapsed_ms, reverse=True)[:n]

    # ── System health ─────────────────────────────────────────────────────────

    @staticmethod
    def system_health() -> dict:
        try:
            return {
                "cpu_pct":    psutil.cpu_percent(interval=0.2),
                "mem_pct":    psutil.virtual_memory().percent,
                "disk_pct":   psutil.disk_usage("/").percent,
                "cpu_count":  psutil.cpu_count(),
                "mem_total_mb": psutil.virtual_memory().total // (1024 * 1024),
                "mem_avail_mb": psutil.virtual_memory().available // (1024 * 1024),
            }
        except Exception as exc:
            logger.warning("system_health error: %s", exc)
            return {}

    def report(self) -> str:
        health = self.system_health()
        slow   = self.slowest(3)
        lines  = [
            "── Performance Report ──────────────────",
            f"  CPU:  {health.get('cpu_pct', '?')}%",
            f"  RAM:  {health.get('mem_pct', '?')}%  "
            f"({health.get('mem_avail_mb', '?')} MB free)",
            "  Slowest operations:",
        ]
        for m in slow:
            lines.append(f"    {m.label}: {m.elapsed_ms:.1f} ms")
        return "\n".join(lines)
