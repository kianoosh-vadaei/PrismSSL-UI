from __future__ import annotations

from typing import Any, Dict, Optional


class ValidationError(ValueError):
    """Raised when an API payload fails validation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _ensure_number(value: Any, field: str, minimum: Optional[float] = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Field '{field}' must be numeric") from exc
    if minimum is not None and number < minimum:
        raise ValidationError(f"Field '{field}' must be ≥ {minimum}")
    return number


def validate_trainer_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    _require("modality" in payload and payload["modality"], "Modality is required")
    trainer_ctor = payload.get("trainer_ctor", {})
    _require(isinstance(trainer_ctor, dict), "trainer_ctor must be an object")
    _require(trainer_ctor.get("method"), "Trainer method is required")
    if "checkpoint_interval" in trainer_ctor:
        _ensure_number(trainer_ctor["checkpoint_interval"], "checkpoint_interval", 1)
    if "num_workers" in trainer_ctor and trainer_ctor["num_workers"] is not None:
        _ensure_number(trainer_ctor["num_workers"], "num_workers", 0)
    return payload


def validate_train_args(args: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_number(args.get("batch_size", 0), "batch_size", 1)
    _ensure_number(args.get("epochs", 0), "epochs", 1)
    _ensure_number(args.get("learning_rate", 0), "learning_rate", 0)
    if args.get("use_hpo"):
        _ensure_number(args.get("n_trials", 0), "n_trials", 1)
        _ensure_number(args.get("tuning_epochs", 0), "tuning_epochs", 1)
    return args


def validate_eval_args(args: Dict[str, Any], has_test: bool) -> Dict[str, Any]:
    _ensure_number(args.get("num_classes", 0), "num_classes", 1)
    _ensure_number(args.get("batch_size", 0), "batch_size", 1)
    _ensure_number(args.get("lr", 0), "lr", 0)
    _ensure_number(args.get("epochs", 0), "epochs", 1)
    _require(has_test, "Instantiate a test dataset before evaluation")
    return args
