import ast
import builtins
import types
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class DatasetExtractionError(Exception):
    """Raised when dataset classes cannot be extracted."""


def _resolve_base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def extract_dataset_classes(code: str) -> List[str]:
    """Return class names that inherit from torch.utils.data.Dataset."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise DatasetExtractionError(f"Failed to parse dataset code: {exc}") from exc

    dataset_classes: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                name = _resolve_base_name(base)
                if name.endswith("Dataset"):
                    dataset_classes.append(node.name)
                    break
    return sorted(set(dataset_classes))


def _exec_user_code(code: str) -> types.ModuleType:
    module = types.ModuleType("user_dataset")
    module.__dict__["__builtins__"] = builtins.__dict__.copy()
    module.__dict__["torch"] = torch
    module.__dict__["Dataset"] = Dataset
    exec(compile(code, "<dataset>", "exec"), module.__dict__)
    return module


def _safe_len(obj) -> Optional[int]:
    try:
        return len(obj)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None


def _preview_item(dataset: Dataset) -> Optional[str]:
    try:
        length = _safe_len(dataset) or 0
        if length == 0:
            return None
        sample = dataset[0]
        if isinstance(sample, torch.Tensor):
            return f"Tensor(shape={tuple(sample.shape)}, dtype={sample.dtype})"
        if isinstance(sample, dict):
            keys = ', '.join(sample.keys())
            return f"Dict keys: {keys}"
        if isinstance(sample, (list, tuple)):
            return f"Sequence(len={len(sample)})"
        return str(type(sample))
    except Exception:  # noqa: BLE001
        return None


def instantiate_datasets(
    code: str,
    class_name: str,
    kwargs_train: Optional[Dict] = None,
    kwargs_val: Optional[Dict] = None,
    kwargs_test: Optional[Dict] = None,
) -> Tuple[Dict[str, Dataset], Dict[str, Dict[str, Optional[str]]]]:
    module = _exec_user_code(code)
    try:
        dataset_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise DatasetExtractionError(f"Class '{class_name}' not found in provided code") from exc

    if not isinstance(dataset_cls, type) or not issubclass(dataset_cls, Dataset):
        raise DatasetExtractionError(
            f"Class '{class_name}' is not a subclass of torch.utils.data.Dataset"
        )

    datasets: Dict[str, Dataset] = {}
    summary: Dict[str, Dict[str, Optional[str]]] = {}

    def _instantiate(split: str, kwargs: Optional[Dict]) -> None:
        if kwargs is None:
            return
        instance = dataset_cls(**kwargs) if kwargs else dataset_cls()
        datasets[split] = instance
        summary[split] = {
            "length": _safe_len(instance),
            "preview": _preview_item(instance),
        }

    _instantiate("train", kwargs_train or {})
    if kwargs_val is not None:
        _instantiate("val", kwargs_val)
    if kwargs_test is not None:
        _instantiate("test", kwargs_test)

    if "train" not in datasets:
        raise DatasetExtractionError("Training dataset instantiation failed.")

    return datasets, summary
