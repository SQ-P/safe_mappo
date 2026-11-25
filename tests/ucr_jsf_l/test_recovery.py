from src.safety.ucr_jsf_l.contracts import SafetyMode
from src.safety.ucr_jsf_l.recovery import RecoveryConfigV1, RecoveryControllerV1


def test_recovery_and_stall_modes_have_hysteresis():
    controller = RecoveryControllerV1(
        RecoveryConfigV1(
            recovery_minimum_steps=1,
            recovery_clear_steps=2,
            stall_clear_steps=2,
        )
    )
    assert controller.update(any_stalled=False, certified_candidate=False) is SafetyMode.SAFETY_RECOVERY
    assert controller.update(any_stalled=False, certified_candidate=True) is SafetyMode.SAFETY_RECOVERY
    assert controller.update(any_stalled=False, certified_candidate=True) is SafetyMode.NORMAL
    assert controller.update(any_stalled=True, certified_candidate=True) is SafetyMode.STALL_ESCAPE
    assert controller.update(any_stalled=False, certified_candidate=True) is SafetyMode.STALL_ESCAPE
    assert controller.update(any_stalled=False, certified_candidate=True) is SafetyMode.NORMAL
