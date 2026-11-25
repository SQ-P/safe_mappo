"""Explicit algorithm selection for training, evaluation and diagnostics."""


def create_trainer(config, project_root=None):
    algorithm = str(config.get("algorithm", "mappo")).lower()
    if algorithm == "mappo":
        train = dict(config.get("train", {}))
        alignment = dict(train.get("safety_alignment", {}))
        intervention = dict(train.get("intervention_aware", {}))
        if alignment.get("mode", "disabled") != "disabled":
            raise ValueError("safety-alignment training is not part of Standard MAPPO")
        if intervention.get("mode", "disabled") != "disabled":
            raise ValueError("intervention-aware training is not part of Standard MAPPO")
        from src.algorithms.mappo.trainer import MAPPOTrainer

        trainer_class = MAPPOTrainer
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}; expected mappo")
    return trainer_class(config, project_root=project_root)
