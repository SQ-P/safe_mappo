"""Shared low-level safety contracts for Standard MAPPO and IA-MAPPO.

This package deliberately contains no FACS planner or subgoal-selection code.
"""

from .contracts import (
    CAUSAL_SAFETY_SNAPSHOT_SCHEMA,
    INTERVENTION_TRACE_SCHEMA,
    SAFETY_STAGES,
    CausalSafetySnapshotV1,
    EnvironmentShadowSnapshotV1,
    InterventionLevel,
    ObservableObstacleV1,
    StageInterventionRecordV1,
    StageInterventionTraceV1,
    action_modification_level,
)

__all__ = [
    "CAUSAL_SAFETY_SNAPSHOT_SCHEMA",
    "INTERVENTION_TRACE_SCHEMA",
    "SAFETY_STAGES",
    "CausalSafetySnapshotV1",
    "EnvironmentShadowSnapshotV1",
    "InterventionLevel",
    "ObservableObstacleV1",
    "StageInterventionRecordV1",
    "StageInterventionTraceV1",
    "action_modification_level",
]
