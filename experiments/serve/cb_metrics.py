"""Minimal Prometheus-format metrics for CBEngine + cb_api.

Handrolled to stay lean (no prometheus_client dep). Three metric types — Counter,
Gauge, Histogram — and a Registry that formats them into the text exposition
format Prometheus + most scrapers understand. Used by:

  - CBEngine: per-step latency + per-request TTFT/duration + counters for
    submit/done/cancel/reject + gauges for queue depth + slot utilization.
  - cb_api.GET /metrics: renders `engine.metrics.registry.format_prometheus()`
    as `text/plain; version=0.0.4`.

Thread-safety: every mutating op acquires a per-metric lock (Counter, Gauge,
Histogram all carry one). Gauges may be backed by a callable that snapshots
external state at scrape time — that read is unlocked and may see a slightly
stale view (standard Prometheus semantics).
"""
from __future__ import annotations

import threading
from typing import Callable, Optional


# Default histogram buckets — covers per-step latency (10ms..500ms typical) and
# end-to-end request duration (~10s for short replies, longer for big max_new).
DEFAULT_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class Counter:
    """Monotonically increasing integer."""
    __slots__ = ("name", "help", "_v", "_lock")

    def __init__(self, name: str, help_: str):
        self.name = name
        self.help = help_
        self._v = 0
        self._lock = threading.Lock()

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self._v += n

    def get(self) -> int:
        return self._v


class Gauge:
    """Snapshot value. If `fn` is set, the value is sampled at scrape time."""
    __slots__ = ("name", "help", "_fn", "_v", "_lock")

    def __init__(self, name: str, help_: str, fn: Optional[Callable[[], float]] = None):
        self.name = name
        self.help = help_
        self._fn = fn
        self._v: float = 0.0
        self._lock = threading.Lock()

    def set(self, v: float) -> None:
        with self._lock:
            self._v = v

    def get(self) -> float:
        if self._fn is not None:
            try:
                return float(self._fn())
            except Exception:
                return 0.0
        return self._v


class Histogram:
    """Cumulative-bucketed histogram. `buckets` are upper bounds (inclusive).
    `_counts[i]` = number of observations <= buckets[i] (already cumulative)."""
    __slots__ = ("name", "help", "buckets", "_counts", "_sum", "_count", "_lock")

    def __init__(self, name: str, help_: str, buckets=DEFAULT_BUCKETS):
        self.name = name
        self.help = help_
        self.buckets = buckets
        self._counts = [0] * len(buckets)
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def observe(self, v: float) -> None:
        with self._lock:
            self._sum += v
            self._count += 1
            # buckets are sorted; once v <= b[i], v <= every b[j>i] too.
            for i, b in enumerate(self.buckets):
                if v <= b:
                    self._counts[i] += 1

    def snapshot(self):
        with self._lock:
            return list(self._counts), self._sum, self._count


class Registry:
    """Holds metrics in insertion order; renders the Prometheus exposition
    format. Insertion order = exposition order, which is the standard."""

    def __init__(self):
        self._metrics: list = []

    def add(self, m):
        self._metrics.append(m)
        return m

    def counter(self, name: str, help_: str) -> Counter:
        return self.add(Counter(name, help_))

    def gauge(self, name: str, help_: str, fn=None) -> Gauge:
        return self.add(Gauge(name, help_, fn))

    def histogram(self, name: str, help_: str, buckets=DEFAULT_BUCKETS) -> Histogram:
        return self.add(Histogram(name, help_, buckets))

    def format_prometheus(self) -> str:
        out: list[str] = []
        for m in self._metrics:
            out.append(f"# HELP {m.name} {m.help}")
            if isinstance(m, Counter):
                out.append(f"# TYPE {m.name} counter")
                out.append(f"{m.name} {m.get()}")
            elif isinstance(m, Gauge):
                out.append(f"# TYPE {m.name} gauge")
                out.append(f"{m.name} {m.get()}")
            elif isinstance(m, Histogram):
                out.append(f"# TYPE {m.name} histogram")
                counts, s, c = m.snapshot()
                for b, ct in zip(m.buckets, counts):
                    out.append(f'{m.name}_bucket{{le="{b}"}} {ct}')
                out.append(f'{m.name}_bucket{{le="+Inf"}} {c}')
                out.append(f"{m.name}_sum {s}")
                out.append(f"{m.name}_count {c}")
        return "\n".join(out) + "\n"
