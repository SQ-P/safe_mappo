"""MAPPO trainer selection wrapper."""

from src.algorithms.multi_agent.trainer import MultiAgentPPOTrainer


class MAPPOTrainer(MultiAgentPPOTrainer):
    ALGORITHM = "mappo"

