"""Fail-closed validation for active IA-MAPPO configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


ARCHIVE_MARKER = "archive/paused_facs_20261109"


def validate_active_config_file(path: str | Path) -> None:
    """Inspect the raw extends chain before YAML merging."""

    current = Path(path).resolve()
    visited: set[Path] = set()
    while True:
        if current in visited:
            raise ValueError("cyclic YAML extends")
        visited.add(current)
        normalized = current.as_posix().lower()
        if ARCHIVE_MARKER in normalized or "/archive/" in normalized:
            raise ValueError("active configuration may not load from the archive")
        with current.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError("configuration must be a mapping")
        parent = payload.get("extends")
        if parent is None:
            return
        parent_text = str(parent).replace("\\", "/").lower()
        if ARCHIVE_MARKER in parent_text or parent_text.startswith("archive/"):
            raise ValueError("active configuration may not extend an archived file")
        current = (current.parent / str(parent)).resolve()


def validate_active_research_config(
    config: Mapping[str, Any], *, source_path: str | Path | None = None
) -> None:
    """Reject archived high-level selectors and Projection-aware losses."""

    selector = dict(config.get("subgoal_selector", {}))
    selector_version = str(selector.get("version", "")).strip().lower()
    if selector_version.startswith("facs"):
        raise ValueError(
            "FACS/FACS-STAR is archived and cannot be used by an active entry point"
        )
    if bool(selector.get("enabled", False)):
        raise ValueError(
            "historical learned/rule subgoal selectors are disabled; use frozen oracle_subgoal"
        )
    if any(
        selector.get(key)
        for key in ("competence_model_artifact", "pairwise_progress_artifact")
    ):
        raise ValueError("historical selector artifacts are archived")
    llm_selector = dict(config.get("llm_subgoal_selector", {}))
    if bool(llm_selector.get("enabled", False)):
        raise ValueError("the historical LLM subgoal selector is archived")
    train = dict(config.get("train", {}))
    if "projection_aware_execution" in train:
        raise ValueError(
            "train.projection_aware_execution is archived; use train.intervention_aware"
        )
    if "candidate_response" in train:
        raise ValueError("historical candidate-response training is archived")
    lineage = config.get("_config_lineage", config.get("extends", ()))
    if isinstance(lineage, (str, Path)):
        lineage = (lineage,)
    for item in lineage or ():
        normalized = str(item).replace("\\", "/").lower()
        if ARCHIVE_MARKER in normalized or normalized.startswith("archive/"):
            raise ValueError("active configuration may not extend an archived file")
    if source_path is not None:
        normalized = str(Path(source_path)).replace("\\", "/").lower()
        if ARCHIVE_MARKER in normalized:
            raise ValueError("archived configuration cannot be launched as active research")

    research_line = str(config.get("research_line", ""))
    if research_line == "ia_mappo_v1":
        if selector_version not in {"", "disabled_archived_v1"}:
            raise ValueError("ia_mappo_v1 requires the disabled selector contract")
        oracle = dict(config.get("oracle_subgoal", {}))
        if not bool(oracle.get("enabled", False)) or not bool(oracle.get("frozen", False)):
            raise ValueError("ia_mappo_v1 requires oracle_subgoal.enabled=true and frozen=true")
        safety = dict(config.get("safety_execution", {}))
        if not bool(safety.get("enabled", False)):
            raise ValueError("ia_mappo_v1 requires the shared safety execution stack")
        if str(safety.get("version")) != "staged_joint_guard_ttc_v1":
            raise ValueError("unsupported ia_mappo_v1 safety_execution.version")


def validate_checkpoint_research_line(
    active_config: Mapping[str, Any], checkpoint_config: Mapping[str, Any]
) -> None:
    """Prevent IA-MAPPO from loading historical weights or normalizers."""

    validate_active_research_config(checkpoint_config)
    active_line = str(active_config.get("research_line", ""))
    saved_line = str(checkpoint_config.get("research_line", ""))
    if active_line == "ia_mappo_v1" and saved_line != "ia_mappo_v1":
        raise ValueError(
            "ia_mappo_v1 must fresh-init or resume only an ia_mappo_v1 checkpoint"
        )
