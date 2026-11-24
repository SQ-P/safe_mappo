from types import SimpleNamespace

import numpy as np
import pytest

from scripts.audit_ppcsa_gate1_effects import summarize,distribution,input_ulp_scale


def evidence():
    states,records,examples = [],[],[]
    for index,(layout,safe,base_value,gain) in enumerate(((1,True,.01,.001),(1,True,.01,.002),(2,False,0.,.003))):
        state = {"agents":[{"position":[0.,0.]},{"position":[0.,5.]},{"position":[0.,10.]}],
            "arrays":{"goals":np.array([[100.,0.],[100.,5.],[100.,10.]],dtype=np.float32)}}
        states.append({"layout_seed":layout,"step":0 if index!=1 else 20,
            "snapshot":SimpleNamespace(environment_state=state)})
        baseline = {"supported":True,"hard_safe":safe,"executed_progress":base_value,
            "predicted_next_positions":[[1.,0.],[1.,5.],[1.,10.]],"rejection_reasons":[] if safe else ["unknown_obstacle_motion"]}
        candidate = {**baseline,"hard_safe":True,"executed_progress":base_value+gain,
            "predicted_next_positions":[[1.1,0.],[1.1,5.],[1.1,10.]],"rejection_reasons":[]}
        records.append({"state_index":index,"layout":layout,"step":states[-1]["step"],"candidates":[
            {"name":"base","kind":"base","improved":False,"result":baseline},
            {"name":"local_0","kind":"local","improved":True,"result":candidate},
            {"name":"local_1","kind":"local","improved":True,"result":candidate}]})
        examples.extend([{"layout":layout,"state_index":index,"target":base_value+gain,"teacher_eligible":True}]*2)
    return {"states":states,"records":records,"examples":examples}


def test_candidate_state_layout_counts_and_baseline_classes_are_distinct():
    report = summarize(evidence(),{"fit":[1],"calibration":[2],"audit":[3]})
    overall = report["overall"]
    assert overall["positive_local_candidates"] == 6
    assert overall["best_positive_local_per_state_gains"]["count"] == 3
    assert overall["best_positive_local_per_state_gains"]["median"] == pytest.approx(.002)
    assert overall["eligible_value_examples"] == 6 and overall["eligible_value_states"] == 3
    assert overall["eligible_value_layouts"] == [1,2]
    assert overall["best_local_gain_by_baseline"]["hard_safe"]["count"] == 2
    assert overall["best_local_gain_by_baseline"]["uncertifiable"]["count"] == 1
    assert overall["best_gain_top1_state_share"] == pytest.approx(.5)
    assert overall["relative_gain_undefined_zero_baseline"] == 1
    assert report["per_layout"]["3"]["analyzed_states"] == 0
    assert report["partitions"]["audit"]["best_positive_local_per_state_gains"] == {"count":0}


def test_precision_reference_and_empty_nonfinite_evidence():
    goals = np.ones((3,2),dtype=np.float32)*100
    assert input_ulp_scale(np.zeros((3,2)),np.ones((3,2)),goals,np.ones(3)*100)>0
    assert distribution([]) == {"count":0}
    with pytest.raises(ValueError,match="Nonfinite"):
        distribution([np.nan])
    data = evidence()
    data["records"][0]["candidates"][1]["result"]["executed_progress"] = .010000000001
    result = summarize(data,{"fit":[1],"calibration":[2],"audit":[3]})
    assert result["overall"]["positive_local_float32_target_ties"]>=1


def test_missing_normalization_or_overlapping_roles_are_rejected():
    with pytest.raises(ValueError,match="overlap"):
        summarize(evidence(),{"fit":[1],"calibration":[1,2],"audit":[3]})
    data = evidence()
    for state in data["states"]:
        state["step"] = 20
    with pytest.raises(ValueError,match="step-zero"):
        summarize(data,{"fit":[1],"calibration":[2],"audit":[3]})
