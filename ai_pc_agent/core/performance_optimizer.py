"""ai_pc_agent/core/performance_optimizer.py

Monitor and optimise the agent's own runtime performance.
"""

from __future__ import annotations
import functools
import time
from collections import defaultdict
from typing import Callable, Any

from ai_pc_agent.diagnostics.performance_monitor import PerformanceMonitor
from ai_pc_agent.utils.logger                    import get_logger

logger = get_logger("agent.optimizer")


class PerformanceOptimizer:
    """
    Provides:
    - result caching (memoization with TTL)
    - call-frequency counters
    - slow-call warnings
    - simple adaptive throttle
    """

    def __init__(self, monitor: PerformanceMonitor):
        self.monitor  = monitor
        self._cache:  dict[str, tuple[Any, float]] = {}
        self._counts: defaultdict[str, int]         = defaultdict(int)

    # ── TTL Cache ─────────────────────────────────────────────────────────────

    def cache(self, ttl_seconds: float = 5.0):
        """Decorator: caches function results for *ttl_seconds*."""
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                key = f"{fn.__qualname__}:{args}:{kwargs}"
                now = time.time()
                if key in self._cache:
                    result, ts = self._cache[key]
                    if now - ts < ttl_seconds:
                        return result
                result = fn(*args, **kwargs)
                self._cache[key] = (result, now)
                return result
            return wrapper
        return decorator

    def invalidate_cache(self, prefix: str = ""):
        if prefix:
            keys = [k for k in self._cache if k.startswith(prefix)]
        else:
            keys = list(self._cache.keys())
        for k in keys:
            del self._cache[k]
        logger.debug("Cache invalidated: %d entries removed", len(keys))

    # ── Slow-call guard ───────────────────────────────────────────────────────

    def timed(self, label: str = "", warn_ms: float = 2000):
        """Decorator: wraps a function with a performance timer."""
        def decorator(fn: Callable) -> Callable:
            lbl = label or fn.__qualname__
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self.monitor.timer(lbl):
                    result = fn(*args, **kwargs)
                stats = self.monitor.stats(lbl)
                if stats and stats.get("avg_ms", 0) > warn_ms:
                    logger.warning("[OPTIMIZER] '%s' avg=%.1f ms — consider caching.", lbl, stats["avg_ms"])
                return result
            return wrapper
        return decorator

    # ── Frequency counter ─────────────────────────────────────────────────────

    def count(self, label: str):
        self._counts[label] += 1
        return self._counts[label]

    def hot_paths(self, top: int = 5) -> list[tuple[str, int]]:
        return sorted(self._counts.items(), key=lambda x: x[1], reverse=True)[:top]

    # ── Adaptive throttle ─────────────────────────────────────────────────────

    def throttle(self, calls_per_second: float):
        """Decorator: rate-limit a function."""
        min_interval = 1.0 / calls_per_second
        last: list[float] = [0.0]
        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                wait = min_interval - (time.time() - last[0])
                if wait > 0:
                    time.sleep(wait)
                last[0] = time.time()
                return fn(*args, **kwargs)
            return wrapper
        return decorator

    def optimize_now(self) -> str:
        """Run a quick optimisation pass and return a summary."""
        hot  = self.hot_paths(3)
        slow = self.monitor.slowest(3)
        lines = ["── Optimiser Report ─────────────────────"]
        lines.append("  Hot paths (call count):")
        for name, cnt in hot:
            lines.append(f"    {name}: {cnt} calls")
        lines.append("  Slowest ops:")
        for m in slow:
            lines.append(f"    {m.label}: {m.elapsed_ms:.1f} ms")
        return "\n".join(lines)
