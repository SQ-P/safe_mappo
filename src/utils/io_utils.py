"""Small pathlib-based IO helpers for smoke."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_yaml(path: str | Path, _ancestors: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = Path(path).resolve()
    if path in _ancestors:
        raise ValueError("cyclic YAML extends")
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError("configuration must be a mapping")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    base = load_yaml(path.parent / parent, (*_ancestors, path))
    def merge(left, right):
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(left.get(key), dict):
                merge(left[key], value)
            else:
                left[key] = value
        return left
    return merge(base, data)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def append_jsonl(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(obj), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: Iterable[Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows = [to_jsonable(row) for row in rows]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return to_jsonable(obj.tolist())
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return repr(obj)
