"""诊断 Reward 的固定执行契约与受限源码编译器。"""

from __future__ import annotations

import hashlib
import ast
import inspect
import json
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

REWARD_API_VERSION = "1.0"
REWARD_ENTRYPOINT = "diagnostic_reward_fn"
MAX_REWARD_STATE_BYTES = 16_384
ALLOWED_NUMPY_ATTRIBUTES = frozenset({
    "abs", "asarray", "clip", "exp", "float32", "full", "log", "maximum",
    "minimum", "ndarray", "ones", "square", "sqrt", "tanh", "where", "zeros",
})
ALLOWED_NAMED_CALLS = frozenset({
    "abs", "bool", "dict", "float", "int", "len", "max", "min",
    REWARD_ENTRYPOINT,
})


def reward_source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def compile_reward_function(source: str, parameters: Mapping[str, Any]) -> Callable:
    """编译完整 Reward 函数；源码不能自行 import。"""
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != REWARD_ENTRYPOINT:
        raise ValueError("source must contain exactly one reward function")
    if tree.body[0].decorator_list or tree.body[0].args.defaults or tree.body[0].args.kw_defaults:
        raise ValueError("reward function cannot have decorators or defaults")
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While, ast.AsyncFor, ast.ListComp, ast.SetComp,
                             ast.DictComp, ast.GeneratorExp, ast.Lambda)):
            raise ValueError("reward function must use bounded vectorized expressions")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == REWARD_ENTRYPOINT:
            raise ValueError("recursive reward calls are forbidden")

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("diagnostic reward source may not import modules")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError("dunder attribute access is forbidden")
            if isinstance(node.value, ast.Name) and node.value.id == "np":
                if node.attr not in ALLOWED_NUMPY_ATTRIBUTES:
                    raise ValueError(f"numpy attribute is not allowed: {node.attr}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in ALLOWED_NAMED_CALLS:
                raise ValueError(f"function call is not allowed: {node.func.id}")
            if isinstance(node.func, ast.Attribute):
                is_numpy_call = (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "np"
                    and node.func.attr in ALLOWED_NUMPY_ATTRIBUTES
                )
                if not is_numpy_call and node.func.attr != "copy":
                    raise ValueError(f"method call is not allowed: {node.func.attr}")
    namespace: dict[str, Any] = {}
    safe_globals = {
        "__builtins__": {"abs": abs, "bool": bool, "dict": dict, "float": float,
                         "int": int, "len": len, "max": max, "min": min},
        "np": np,
        "PARAMETERS": MappingProxyType(dict(parameters)),
    }
    exec(compile(source, "<diagnostic_reward>", "exec"), safe_globals, namespace)
    reward_fn = namespace.get(REWARD_ENTRYPOINT)
    if not callable(reward_fn):
        raise ValueError(f"missing callable reward entrypoint: {REWARD_ENTRYPOINT}")
    if tuple(inspect.signature(reward_fn).parameters) != (
        "base_reward", "reward_context", "state"
    ):
        raise ValueError(
            "diagnostic_reward_fn signature must be "
            "(base_reward, reward_context, state)"
        )
    return reward_fn


def freeze_reward_context(context: Mapping[str, Any]) -> Mapping[str, Any]:
    """复制并冻结数组，禁止补丁篡改环境状态或上下文。"""
    frozen: dict[str, Any] = {}
    for key, value in context.items():
        if isinstance(value, np.ndarray):
            copied = value.copy()
            copied.setflags(write=False)
            frozen[str(key)] = copied
        else:
            frozen[str(key)] = value
    return MappingProxyType(frozen)


def validate_reward_result(
    result: Any, *, num_agents: int, max_abs_reward: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """验证返回形状、有限性、幅值与有界 state。"""
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("diagnostic_reward_fn must return (shaped_reward, new_state)")
    shaped, state = result
    shaped_array = np.asarray(shaped, dtype=np.float32)
    if shaped_array.shape != (num_agents,):
        raise ValueError(
            f"shaped_reward must have shape ({num_agents},), got {shaped_array.shape}"
        )
    if not np.all(np.isfinite(shaped_array)):
        raise FloatingPointError("diagnostic reward produced NaN/Inf")
    if float(np.max(np.abs(shaped_array))) > float(max_abs_reward):
        raise ValueError("diagnostic reward exceeds configured magnitude limit")
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise TypeError("reward state must be a dict with string keys")
    try:
        encoded = json.dumps(state, default=_json_default, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("reward state must be JSON serializable") from exc
    if len(encoded.encode("utf-8")) > MAX_REWARD_STATE_BYTES:
        raise ValueError("reward state exceeds size limit")
    return shaped_array, dict(state)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)
