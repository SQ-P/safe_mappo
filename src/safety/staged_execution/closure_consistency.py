"""Observable-only incident closure semantics for the final STAR audit.

This module deliberately owns no control policy.  It decides whether an
already-open incident may close from real-step observations and a release
certificate produced by the frozen execution stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


CLOSURE_ACTIVE = "ACTIVE"
CLOSURE_PENDING = "CLEARANCE_PENDING"
CLOSURE_VERIFIED = "VERIFIED_CLEAR"
CLOSURE_UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ClosureObservationV1:
    track_available: bool
    same_hazard_active: bool
    release_certificate_passed: bool


@dataclass(frozen=True)
class ClosureDecisionV1:
    state: str
    close_allowed: bool
    pending_reappearance: bool
    clear_streak: int
    certificate_tail: tuple[bool, ...]


@dataclass
class ClosureConsistencyV1:
    """Five-real-step closure tracker with a three-step certificate tail."""

    clear_window_steps: int = 5
    certificate_tail_steps: int = 3
    state: str = CLOSURE_ACTIVE
    clear_streak: int = 0
    certificate_tail: list[bool] = field(default_factory=list)
    pending_reappearance_count: int = 0

    def __post_init__(self) -> None:
        if self.clear_window_steps <= 0:
            raise ValueError("clear_window_steps must be positive")
        if not 0 < self.certificate_tail_steps <= self.clear_window_steps:
            raise ValueError("certificate tail must be inside the clear window")

    def reset(self) -> None:
        self.state = CLOSURE_ACTIVE
        self.clear_streak = 0
        self.certificate_tail.clear()

    def observe(self, observation: ClosureObservationV1) -> ClosureDecisionV1:
        previous = self.state
        reappearance = bool(
            observation.same_hazard_active
            and previous in {CLOSURE_PENDING, CLOSURE_UNKNOWN}
        )
        if reappearance:
            self.pending_reappearance_count += 1

        if observation.same_hazard_active:
            self.state = CLOSURE_ACTIVE
            self.clear_streak = 0
            self.certificate_tail.clear()
        elif not observation.track_available:
            self.state = CLOSURE_UNKNOWN
            self.clear_streak = 0
            self.certificate_tail.clear()
        else:
            self.state = CLOSURE_PENDING
            self.clear_streak += 1
            self.certificate_tail.append(
                bool(observation.release_certificate_passed)
            )
            self.certificate_tail = self.certificate_tail[
                -self.certificate_tail_steps:
            ]
            certificate_ready = bool(
                len(self.certificate_tail) == self.certificate_tail_steps
                and all(self.certificate_tail)
            )
            if self.clear_streak >= self.clear_window_steps and certificate_ready:
                self.state = CLOSURE_VERIFIED

        return ClosureDecisionV1(
            state=self.state,
            close_allowed=self.state == CLOSURE_VERIFIED,
            pending_reappearance=reappearance,
            clear_streak=int(self.clear_streak),
            certificate_tail=tuple(self.certificate_tail),
        )

    def terminal_certificate(
        self,
        *,
        team_success: bool,
        same_hazard_active: bool,
        final_stack_feasible: bool,
    ) -> bool:
        """Allow only an explicit, safe goal-completion terminal close."""

        allowed = bool(
            team_success and not same_hazard_active and final_stack_feasible
        )
        if allowed:
            self.state = CLOSURE_VERIFIED
        return allowed


def obstacle_track_keys(
    visible_rows: Sequence[Sequence[object]],
) -> set[str]:
    """Return causal obstacle keys without reading simulator identities."""

    keys: set[str] = set()
    for agent_id, row in enumerate(visible_rows):
        for item in row:
            keys.add(
                f"agent:{int(agent_id)}:track:{int(getattr(item, 'track_id'))}"
            )
    return keys
