"""Fixed A8 capture-bank timing with an audited cumulative step target."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CaptureSchedule:
    start_update: int
    end_update: int
    interval: int
    target_fraction_per_active_update: float
    total_updates: int

    @classmethod
    def from_config(cls, config: dict, total_updates: int,
                    max_step_fraction: float) -> "CaptureSchedule":
        if config.get("mode") != "fixed_active_updates_v1":
            raise ValueError("unsupported capture schedule mode")
        schedule = cls(
            start_update=int(config["start_update"]),
            end_update=int(config["end_update"]),
            interval=int(config["interval"]),
            target_fraction_per_active_update=float(
                config["target_fraction_per_active_update"]),
            total_updates=int(total_updates),
        )
        if not (1 <= schedule.start_update <= schedule.end_update <= total_updates):
            raise ValueError("capture schedule update range is outside training")
        if schedule.interval < 1:
            raise ValueError("capture schedule interval must be positive")
        if not (0 < schedule.target_fraction_per_active_update
                < max_step_fraction <= 0.5):
            raise ValueError("capture schedule target must be below per-rollout cap")
        if schedule.active_count != int(config.get("expected_active_updates", -1)):
            raise ValueError("capture schedule active count differs from frozen protocol")
        return schedule

    @property
    def active_count(self) -> int:
        return (self.end_update - self.start_update) // self.interval + 1

    def active_index(self, update: int) -> int:
        """Count active updates through ``update``; zero before onset."""
        if update < self.start_update:
            return 0
        capped = min(update, self.end_update)
        return (capped - self.start_update) // self.interval + 1

    def active(self, update: int) -> bool:
        return (self.start_update <= update <= self.end_update
                and (update - self.start_update) % self.interval == 0)

    def planned_cumulative_steps(self, update: int, rollout_steps: int) -> int:
        return math.floor(self.active_index(update) * rollout_steps
                          * self.target_fraction_per_active_update + 0.5)

    def budget(self, update: int, rollout_steps: int, completed_steps: int,
               max_step_fraction: float) -> int:
        if not self.active(update):
            return 0
        cap = int(rollout_steps * max_step_fraction)
        outstanding = self.planned_cumulative_steps(update, rollout_steps) - completed_steps
        return min(cap, max(0, outstanding))
