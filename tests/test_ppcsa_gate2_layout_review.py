from copy import deepcopy

import pytest

from scripts.review_ppcsa_gate2_layouts import paired,coverage,review_report


def row(layout,step,gain,*,safe=True,risk=False):
    return {"layout":layout,"step":step,"gain":gain,"certifiable":safe,
        "risk_proxy_regression":risk,"hard_safety_regression":False,
        "outcome":"unsafe_or_unsupported" if not safe else ("risk_worsened" if risk else ("improved" if gain>0 else "worsened")),
        "positive_safe":bool(safe and risk is False and gain>0),"rejection_reasons":[]}


def test_pairing_uses_keys_common_safety_and_layout_weighting():
    left = [row(1,0,10.),row(1,1,10.),row(2,0,-1.),row(2,1,100.,risk=True),row(3,0,100.,safe=False,risk=None)]
    right = [row(r["layout"],r["step"],0.) for r in reversed(left)]
    result = paired(left,right,[1,2,3,4])
    assert result["paired_rows"] == 5 and result["common_eligible"] == 3
    assert result["common_gain_difference"]["net"] == 19.
    assert result["common_gain_difference"]["mean"] == pytest.approx(19./3)
    assert result["layout_balanced_mean_gain_difference"] == 4.5
    assert result["right_only_eligible"] == 2
    assert result["per_layout"]["3"]["delta_left_minus_right"]["mean"] is None
    assert result["per_layout"]["4"]["pairs"] == 0


def test_coverage_does_not_count_uncertifiable_or_risk_gain_as_safe_progress():
    report = coverage([row(1,0,1.),row(1,1,-.2),row(1,2,100.,risk=True),row(2,0,100.,safe=False,risk=None)])
    assert report["eligible_progress"]["net"] == pytest.approx(.8)
    assert report["risk_proxy_regressions"] == 1 and report["uncertifiable"] == 1
    assert report["hard_certification_losses"] == 0
    assert report["certifiable_progress_including_risk_worsening"]["net"] == pytest.approx(100.8)
    assert report["raw_progress_including_uncertifiable"]["net"] == pytest.approx(200.8)


def test_pair_keys_and_numerical_evidence_fail_closed():
    a,b = [row(1,0,1.)],[row(1,0,2.)]
    with pytest.raises(ValueError,match="Duplicate"):
        paired(a+a,b,[1])
    with pytest.raises(ValueError,match="keys differ"):
        paired(a,[row(1,1,2.)],[1])
    with pytest.raises(ValueError,match="Nonfinite"):
        paired([row(1,0,float("nan"))],b,[1])
    a[0].update(epsilon=.01,delta_l2=.01)
    b[0].update(epsilon=.01,delta_l2=.02)
    result = paired(a,b,[1],action=True)
    assert result["norm_mismatch_pairs"] == 1 and result["common_eligible"] == 0


def fixture_report():
    models = {}
    for name,multiplier in (("ordinary",1.),("paired",.5)):
        parameters = [row(1,0,.01*multiplier),row(2,0,-.002*multiplier)]
        actions = [{**row(layout,0,gain*multiplier if direction == "critic" else 0.),
            "direction":direction,"epsilon":e,"delta_l2":e}
            for layout,gain in ((1,.01),(2,-.002)) for e in (.01,.05,.10) for direction in ("critic","random","goal")]
        models[name] = {"action_probes":actions,"predicted_positive_but_absorbed":0,
            "parameter_probe":{"heldout_outcomes":parameters,"gradient_norm":1.,"calibration_states":1,
                "calibration_layouts":[0],"per_agent_gradient_cosine":[[1.]],"gradient_cancellation_ratio":1.,"parameter_step":.01}}
    return {"audit_layouts":[1,2],"partitions":{"audit":[1,2]},"probe_epsilons":[.01,.05,.10],"parameter_step":.01,"models":models}


def test_full_review_retains_all_scales_negative_paired_difference_and_inputs():
    source = fixture_report()
    before = deepcopy(source)
    report = review_report(source)
    assert source == before
    assert set(report["paired_minus_ordinary_action"]) == {"0.01","0.05","0.1"}
    assert report["paired_minus_ordinary_parameter"]["common_gain_difference"]["net"] == pytest.approx(-.004)
    assert report["models"]["ordinary"]["parameter_probe"]["per_layout"]["2"]["eligible_progress"]["net"]<0
    assert report["status"] == "RESEARCH_REVIEW_REQUIRED" and not report["actor_training_authorized"]
    assert not report["refit_authorized"] and not report["pilot_authorized"]


def test_fixed_protocol_cannot_silently_select_favorable_scale():
    source = fixture_report()
    source["probe_epsilons"] = [.01]
    with pytest.raises(ValueError,match="protocol"):
        review_report(source)
