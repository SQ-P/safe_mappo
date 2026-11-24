"""Create the non-overwriting formal CSA-vs-Standard paired report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.diagnostics.csa_paper_report import compare_csa_formal
from src.utils.io_utils import save_json


def _read(paths):
    rows, sources = [], []
    for value in paths:
        path = Path(value).resolve()
        sources.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows, sources


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-files", nargs="+", required=True)
    parser.add_argument("--candidate-files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError(f"report already exists: {output}")
    standard, standard_sources = _read(args.standard_files)
    candidate, candidate_sources = _read(args.candidate_files)
    report = compare_csa_formal(
        standard, candidate, bootstrap_samples=args.bootstrap_samples
    )
    report["source_files"] = {
        "standard": standard_sources, "candidate": candidate_sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, output)
    return report


if __name__ == "__main__":
    main()
