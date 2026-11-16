"""Conservative ETA estimates from comparable, completed invocations only."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import median


@dataclass
class TimingSamples:
    samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def observe(self, key: str, elapsed: float) -> None:
        if elapsed > 0:
            self.samples[key].append(elapsed)

    def estimate(
        self, pending: Sequence[tuple[str, float]], *, external_wait: bool = False
    ) -> dict[str, object]:
        if not pending:
            return {"available": True, "seconds_interval": [0.0, 0.0], "sample_count": 0}
        if external_wait:
            return {"available": False, "reason": "external_resource_wait_duration_unknown"}
        lower = upper = central = 0.0
        used = 0
        for key, active_elapsed in pending:
            samples = self.samples.get(key, [])
            if len(samples) < 2:
                return {
                    "available": False,
                    "reason": "insufficient_comparable_completed_samples",
                    "minimum_samples_per_comparison_group": 2,
                }
            if active_elapsed >= max(samples):
                return {
                    "available": False,
                    "reason": "active_duration_exceeds_comparable_completed_samples",
                    "sample_count": len(samples),
                }
            used += len(samples)
            lower += max(0.0, min(samples) - active_elapsed)
            upper += max(0.0, max(samples) - active_elapsed)
            central += max(0.0, median(samples) - active_elapsed)
        return {
            "available": True,
            "seconds_interval": [lower, upper],
            "median_serial_work_seconds": central,
            "sample_count": used,
            "semantics": "remaining_execution_work_not_wall_clock_completion",
        }


def waiting_record(wait: object | None) -> Mapping[str, object] | None:
    if wait is None:
        return None
    return {key: getattr(wait, key, None) for key in ("reason", "since", "detail")}


def wall_clock_bounds(work: Mapping[str, object], *, max_parallel: int) -> dict[str, object]:
    """Conservative empirical bounds, not a convergence/confidence guarantee."""
    if not work.get("available"):
        return dict(work)
    lower, upper = work["seconds_interval"]
    return {
        **work,
        "seconds_interval": [lower / max(1, max_parallel), upper],
        "semantics": "empirical_wall_clock_bounds_parallel_lower_serial_upper",
        "caveat": "completed comparable samples only; future slowdown may exceed these bounds",
    }
