"""Legal terminal-state reset bank for the A7 PPO exposure experiment.

The bank contains environment snapshots reached through ordinary 4S dynamics.
Its teacher prefix is never inserted into the PPO rollout buffer.  PPO actions
after a sampled reset are drawn from the current Actor and are on-policy.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.envs.cooperative_uav_env import CooperativeUAVEnv


BANK_SCHEMA = "4s_a7_capture_bank_v1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_bucket(observation: np.ndarray, env: CooperativeUAVEnv) -> bool:
    """Broad observable final-goal state; includes varied entry velocities."""
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (2, env.local_observation_dim):
        return False
    if not np.allclose(values[:, 4:6], values[:, 6:8], atol=1e-5, rtol=0):
        return False
    distances = np.linalg.norm(values[:, 4:6] - values[:, 0:2], axis=1)
    return bool(
        3.0 < float(np.max(distances)) < 6.0
        and env.max_episode_steps - env.step_count >= 8
    )


def load_bank_manifest(path: Path, base_task_fingerprint: str) -> dict[str, Any]:
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BANK_SCHEMA:
        raise ValueError("wrong A7 capture bank schema")
    if manifest.get("split") != "train":
        raise ValueError("A7 training may only sample the reserved training bank")
    if manifest.get("task_config_fingerprint") != base_task_fingerprint:
        raise ValueError("A7 bank MDP/PPO contract differs from training task")
    entries = manifest.get("anchors", [])
    if not entries:
        raise ValueError("A7 capture bank is empty")
    root = path.parent
    for entry in entries:
        anchor_path = (root / entry["file"]).resolve()
        if not anchor_path.is_relative_to(root) or not anchor_path.is_file():
            raise ValueError("A7 bank anchor path is invalid")
        if file_sha256(anchor_path) != entry["sha256"]:
            raise ValueError(f"A7 bank anchor changed: {anchor_path}")
    return manifest


def load_anchor(path: Path, entry: dict[str, Any]) -> tuple[CooperativeUAVEnv, np.ndarray, np.ndarray]:
    anchor_path = (path.parent / entry["file"]).resolve()
    if file_sha256(anchor_path) != entry["sha256"]:
        raise ValueError("A7 bank anchor changed after manifest validation")
    # Only load banks generated locally by scripts/a7_prepare_capture_bank.py.
    with anchor_path.open("rb") as stream:
        saved = pickle.load(stream)
    env = saved["env"]
    if type(env) is not CooperativeUAVEnv:
        raise ValueError("A7 anchor does not contain the 4S base environment")
    observation = np.asarray(saved["observation"], dtype=np.float32)
    history = np.asarray(saved["observation_history"], dtype=np.float32)
    if observation.shape != (2, env.local_observation_dim):
        raise ValueError("A7 anchor observation shape mismatch")
    if history.ndim != 3 or history.shape[1:] != observation.shape:
        raise ValueError("A7 anchor GRU history shape mismatch")
    if not np.array_equal(observation, env.build_local_observations()):
        raise ValueError("A7 anchor observation differs from saved environment")
    if not capture_bucket(observation, env):
        raise ValueError("A7 anchor left the preregistered capture bucket")
    if int(env.step_count) != int(entry["step_count"]) or len(history) != env.step_count:
        raise ValueError("A7 anchor physical/observation history length mismatch")
    return env, observation, history


def warm_capture_hidden(trainer: Any, history: np.ndarray) -> torch.Tensor:
    """Rebuild the current Actor's recurrent state from the physical prefix."""
    hidden = trainer.actor.initial_hidden(trainer.policy_rows, trainer.device)
    with torch.no_grad():
        for old_obs in history:
            policy_obs = trainer.policy_observations(old_obs)
            encoded = trainer.actor.encoder(torch.as_tensor(
                policy_obs, dtype=torch.float32, device=trainer.device
            )).unsqueeze(0)
            _, hidden = trainer.actor.gru(encoded, hidden.float())
    return hidden.detach()
