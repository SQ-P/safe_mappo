"""Stateful liveness and safety-recovery mode controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import SafetyMode


@dataclass(frozen=True)
class RecoveryConfigV1:
    recovery_minimum_steps: int = 2
    recovery_clear_steps: int = 2
    stall_clear_steps: int = 2
    release_margin_m: float = 0.25

    @classmethod
    def from_mapping(cls, payload: dict) -> "RecoveryConfigV1":
        return cls(
            recovery_minimum_steps=int(payload.get("recovery_minimum_steps", 2)),
            recovery_clear_steps=int(payload.get("recovery_clear_steps", 2)),
            stall_clear_steps=int(payload.get("stall_clear_steps", 2)),
            release_margin_m=float(payload.get("release_margin_m", 0.25)),
        )


class RecoveryControllerV1:
    def __init__(self, config: RecoveryConfigV1) -> None:
        self.config = config
        if min(
            config.recovery_minimum_steps,
            config.recovery_clear_steps,
            config.stall_clear_steps,
        ) < 1 or config.release_margin_m < 0.0:
            raise ValueError("recovery hysteresis must be positive")
        self.reset()

    def reset(self) -> None:
        self.mode = SafetyMode.NORMAL
        self.mode_steps = 0
        self.clear_steps = 0

    def update(
        self,
        *,
        any_stalled: bool,
        certified_candidate: bool,
        clearance_ok: bool | None = None,
    ) -> SafetyMode:
        self.mode_steps += 1
        clear = bool(
            certified_candidate and (clearance_ok is None or clearance_ok)
        )
        if not clear:
            if self.mode is not SafetyMode.SAFETY_RECOVERY:
                self.mode = SafetyMode.SAFETY_RECOVERY
                self.mode_steps = 0
            self.clear_steps = 0
            return self.mode
        if self.mode is SafetyMode.SAFETY_RECOVERY:
            self.clear_steps += 1
            if (
                self.mode_steps >= self.config.recovery_minimum_steps
                and self.clear_steps >= self.config.recovery_clear_steps
            ):
                self.mode = SafetyMode.STALL_ESCAPE if any_stalled else SafetyMode.NORMAL
                self.mode_steps = 0
                self.clear_steps = 0
            return self.mode
        if any_stalled:
            if self.mode is not SafetyMode.STALL_ESCAPE:
                self.mode = SafetyMode.STALL_ESCAPE
                self.mode_steps = 0
            self.clear_steps = 0
            return self.mode
        if self.mode is SafetyMode.STALL_ESCAPE:
            self.clear_steps += 1
            if self.clear_steps >= self.config.stall_clear_steps:
                self.mode = SafetyMode.NORMAL
                self.mode_steps = 0
                self.clear_steps = 0
        else:
            self.mode = SafetyMode.NORMAL
        return self.mode
