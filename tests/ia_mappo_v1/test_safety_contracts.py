from hashlib import sha256

import numpy as np
import pytest

from src.safety.contracts import (
    CausalSafetySnapshotV1,
    InterventionLevel,
    ObservableObstacleV1,
    StageInterventionRecordV1,
    StageInterventionTraceV1,
    action_modification_level,
)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def test_physical_intervention_thresholds_and_reversal():
    level, delta_v, cosine = action_modification_level(
        [0.0, 0.0], [0.1, 0.0], acceleration_scale=1.0, dt=0.5
    )
    assert level == InterventionLevel.MATERIAL
    assert delta_v == pytest.approx(0.05)
    level, delta_v, _ = action_modification_level(
        [0.0, 0.0], [0.5, 0.0], acceleration_scale=1.0, dt=0.5
    )
    assert level == InterventionLevel.LARGE
    assert delta_v == pytest.approx(0.25)
    level, _, cosine = action_modification_level(
        [1.0, 0.0], [-0.01, 0.0], acceleration_scale=1.0, dt=0.5
    )
    assert level == InterventionLevel.DIRECTION_REVERSAL
    assert cosine < 0.0


def test_recovery_trace_cannot_be_actor_risk_label():
    record = StageInterventionRecordV1(
        uav_id=0, stage="obstacle_ttc", input_action=(0.0, 0.0),
        output_action=(0.0, 0.0), direction_cosine=1.0,
        reason="unchanged", safety_margin=None,
        eligible_for_actor_risk=True,
        modification_level=InterventionLevel.NONE, delta_v_mps=0.0,
    )
    with pytest.raises(ValueError, match="not actor-risk labels"):
        StageInterventionTraceV1(
            step=0, snapshot_digest=_digest("snapshot"), records=(record,),
            recovery_active=True,
        )


def test_causal_snapshot_is_hash_stable():
    obstacle = ObservableObstacleV1(
        observer_id=0, track_id=1, relative_position=(2.0, 3.0),
        relative_velocity=(0.1, -0.2), radius=1.0, velocity_valid=True,
    )
    snapshot = CausalSafetySnapshotV1(
        step=2, team_observation=((1.0, 2.0),),
        requested_actions=((0.2, -0.1),), safety_state=(3.0,),
        obstacle_tracks=((obstacle,),), rng_state_digest=_digest("rng"),
        recovery_state_digest=_digest("recovery"),
    )
    assert snapshot.digest == snapshot.digest
    assert len(snapshot.digest) == 64
