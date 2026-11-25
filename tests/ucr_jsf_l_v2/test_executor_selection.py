from __future__ import annotations

from pathlib import Path

from src.safety.ucr_jsf_l_v2.contracts import CandidateEvaluationV2, ConstraintReportV2
from src.safety.ucr_jsf_l_v2.executor import UCRJSFLExecutorV2


def _evaluation(candidate_id, signature, *, emergency=False):
    return CandidateEvaluationV2(
        candidate_id=candidate_id,
        actions=((0.0, 0.0),) * 3,
        constraints=ConstraintReportV2(model_admissible=not emergency),
        progress_score=None if emergency else 1.0,
        actor_deviation=0.5,
        source="actor+actor+actor",
        canonical_action_signature=signature,
        minimum_effective_margin_m=0.25,
        emergency_risk_score=2.0,
    )


def test_normal_and_emergency_ties_ignore_candidate_id_and_input_order():
    lower = ((-0.2, 0.0), (0.0, 0.0), (0.2, 0.0))
    upper = ((-0.1, 0.0), (0.0, 0.0), (0.1, 0.0))
    rows = [_evaluation("zzz", upper), _evaluation("aaa", lower)]
    for ordered in (rows, list(reversed(rows))):
        assert min(ordered, key=UCRJSFLExecutorV2._normal_key).canonical_action_signature == lower

    emergency = [
        _evaluation("zzz", upper, emergency=True),
        _evaluation("aaa", lower, emergency=True),
    ]
    for ordered in (emergency, list(reversed(emergency))):
        assert min(ordered, key=UCRJSFLExecutorV2._emergency_key).canonical_action_signature == lower


def test_candidate_id_is_not_reintroduced_into_selection_keys():
    source = Path("src/safety/ucr_jsf_l_v2/executor.py").read_text(encoding="utf-8")
    normal = source[source.index("def _normal_key"):source.index("def _emergency_key")]
    emergency = source[source.index("def _emergency_key"):source.index("def execute")]
    assert "candidate_id" not in normal
    assert "candidate_id" not in emergency
