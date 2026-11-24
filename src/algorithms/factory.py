"""Explicit algorithm selection for training, evaluation and diagnostics."""


def create_trainer(config, project_root=None):
    algorithm = str(config.get("algorithm", "ppo")).lower()
    if algorithm == "ppo":
        from src.algorithms.ppo.trainer import PPOTrainer

        trainer_class = PPOTrainer
    elif algorithm == "mappo":
        from src.algorithms.mappo.trainer import MAPPOTrainer

        trainer_class = MAPPOTrainer
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}; expected ppo or mappo")
    return trainer_class(config, project_root=project_root)
