from copy import deepcopy
from pathlib import Path

import torch

from src.algorithms.factory import create_trainer
from src.utils.io_utils import load_yaml


def _train_one_update(config, root: Path, telemetry_level: str):
    candidate = deepcopy(config)
    candidate["runtime"] = {
        "serial_fastpath": {
            "enabled": False,
            "telemetry_level": telemetry_level,
            "evaluation_telemetry_level": "full",
        }
    }
    output = root / telemetry_level
    candidate["output"] = {
        "output_dir": str(output),
        "checkpoint_dir": str(output / "checkpoints"),
        "reject_nonempty": True,
    }
    trainer = create_trainer(candidate, project_root=Path.cwd())
    trainer.train()
    return torch.load(
        output / "checkpoints" / "latest.pt",
        map_location="cpu", weights_only=False,
    )


def test_standard_telemetry_level_does_not_change_checkpoint_tensors(tmp_path):
    config = load_yaml("configs/experiments/ia_mappo_v1_smoke_cpu.yaml")
    full = _train_one_update(config, tmp_path, "full")
    minimal = _train_one_update(config, tmp_path, "train_minimal")
    for component in ("actor", "critic"):
        assert full[component].keys() == minimal[component].keys()
        for name in full[component]:
            torch.testing.assert_close(
                full[component][name], minimal[component][name],
                rtol=0.0, atol=0.0,
            )
