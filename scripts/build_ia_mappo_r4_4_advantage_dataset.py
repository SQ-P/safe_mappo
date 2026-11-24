"""Build an immutable six-anchor R4.4 advantage dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.intervention_advantage_certification import build_advantage_dataset


def _source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_source, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--role", choices=("development", "fresh_certification"), required=True)
    parser.add_argument("--development-split-source")
    parser.add_argument("--forbidden-panel", action="append", default=[])
    args = parser.parse_args()
    manifest = build_advantage_dataset(
        sources=args.source, output_dir=args.output_dir, role=args.role,
        development_split_source=args.development_split_source,
        forbidden_layout_panels=args.forbidden_panel,
    )
    print(f"R4.4 dataset PASS: role={manifest['role']} rows={manifest['team_step_count']}")


if __name__ == "__main__":
    main()
