"""Shared staged safety execution for Standard MAPPO and IA-MAPPO.

Only observable low-level guard, TTC, recovery, and attribution utilities are
exposed here. High-level candidate generation and subgoal selection are not.
"""

from .contracts import (
    ObservableObstacleV1,
    TeamPlanningSnapshotV1,
)
from .closure_consistency import (
    CLOSURE_ACTIVE,
    CLOSURE_PENDING,
    CLOSURE_UNKNOWN,
    CLOSURE_VERIFIED,
    ClosureConsistencyV1,
    ClosureObservationV1,
    obstacle_track_keys,
)
from .execution_consistency import (
    TTCActionCandidateAuditV1,
    TTCActionCandidateEvaluationV1,
    is_safe_progress_headroom,
    registered_ttc_actions_v1,
    registered_ttc_headroom_actions,
    select_ttc_progress_candidate_v2,
)
from .goal_corridor import project_goal_boundary_corridor
from .obstacle_guard import (
    assess_obstacle_actions,
    minimum_predicted_obstacle_clearance_m,
    project_obstacle_actions,
)
from .recovery import (
    StarRecoveryCommandV1,
    build_star_recovery_candidates,
    build_star_recovery_command,
)
from .actions import deterministic_hold_actions

__all__ = [name for name in globals() if not name.startswith("_")]
