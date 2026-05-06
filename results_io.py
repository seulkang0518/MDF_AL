import json
from pathlib import Path

import numpy as np


def _json_path(path_like):
    path = Path(path_like)
    if path.suffix == ".json":
        return path
    return path.with_suffix(".json")


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def save_results_json(results, output_path):
    output_path = _json_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(results)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def load_results_json(input_path):
    input_path = Path(input_path)
    if not input_path.exists() and input_path.suffix != ".json":
        input_path = _json_path(input_path)
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
