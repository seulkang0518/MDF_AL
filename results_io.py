import json
from pathlib import Path

import numpy as np


def metadata_sidecar_path(path_like):
    path = Path(path_like)
    return path.with_name(f"{path.stem}_meta.json")


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


def _is_metadata_value(value):
    if isinstance(value, Path):
        return True
    if isinstance(value, (str, bool, int, float, np.bool_, np.integer, np.floating)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim <= 1 and value.size <= 256
    if isinstance(value, (list, tuple)):
        return len(value) <= 256 and all(
            isinstance(item, (str, bool, int, float, np.bool_, np.integer, np.floating))
            for item in value
        )
    return False


def extract_metadata(results):
    metadata = {}
    for key, value in results.items():
        if _is_metadata_value(value):
            metadata[key] = _to_jsonable(value)
    return metadata


def save_metadata_sidecar(results, output_path):
    sidecar_path = metadata_sidecar_path(output_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    payload = extract_metadata(results)
    payload["result_file"] = str(Path(output_path))
    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return sidecar_path
