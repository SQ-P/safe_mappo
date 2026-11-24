from pathlib import Path

from src.envs.cooperative_uav_env import CooperativeUAVEnv
from src.safety.config_guard import (
    validate_active_config_file,
    validate_active_research_config,
)
from src.utils.io_utils import load_yaml


CONFIGS = {
    23: Path("configs/experiments/standard_mappo_3u_8s2d_seed23.yaml"),
    41: Path("configs/experiments/standard_mappo_3u_8s2d_seed41.yaml"),
    67: Path("configs/experiments/standard_mappo_3u_8s2d_seed67.yaml"),
}


def test_legacy_execution_standard_configs_share_the_registered_contract():
    fingerprints = []
    for seed, path in CONFIGS.items():
        validate_active_config_file(path)
        config = load_yaml(path)
        validate_active_research_config(config, source_path=path)

        assert config["seed"] == seed
        assert config["env"]["num_agents"] == 3
        assert config["env"]["static_obstacle_num"] == 8
        assert config["env"]["dynamic_obstacle_num"] == 2
        assert config["env"]["obstacle_slots"] == 6
        assert config["train"]["total_updates"] == 800
        assert config["train"]["checkpoint_updates"] == [100, 200, 400, 600, 800]
        assert config["subgoal_selector"] == {
            "enabled": False,
            "version": "disabled_archived_v1",
        }
        safety = config["safety_execution"]
        assert safety["obstacle_ttc"]["version"] == "causal_ttc_progress_projection_v2"
        assert safety["safety_state_machine_version"] == "star_scoped_recovery_v4"
        assert safety["candidate_feasibility_version"] == "guard_aligned_observable_v2"
        assert f"seed_{seed}" in config["output"]["output_dir"]
        fingerprints.append((config["env"], config["reward"], safety))

    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_legacy_execution_config_builds_expected_observation_contract():
    config = load_yaml(CONFIGS[23])
    env = CooperativeUAVEnv({
        **config["env"],
        "reward": config["reward"],
        "oracle_subgoal": config["oracle_subgoal"],
        "subgoal_selector": config["subgoal_selector"],
        "safety_execution": config["safety_execution"],
    })
    env.reset(seed=23)
    assert env.local_observation_dim == 52
    assert env.global_state_dim == 156
    assert env.obstacle_execution_guard_version == "causal_ttc_progress_projection_v2"
    assert env.star_safety_state_machine_version == "star_scoped_recovery_v4"
    assert env.star_candidate_feasibility_version == "guard_aligned_observable_v2"
