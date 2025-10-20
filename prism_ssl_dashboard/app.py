from __future__ import annotations

from typing import Dict, Optional

from flask import Flask, jsonify, render_template, request

from utils.dataset_loader import DatasetExtractionError, extract_dataset_classes, instantiate_datasets
from utils.validators import ValidationError, validate_eval_args, validate_train_args, validate_trainer_payload

app = Flask(__name__, template_folder="templates", static_folder="static")


def _json_response(data, status: int = 200):
    return jsonify(data), status


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/dataset/inspect")
def api_dataset_inspect():
    payload = request.get_json() or {}
    code = payload.get("code", "")
    if not code.strip():
        return _json_response({"error": "Code is required"}, 400)
    try:
        classes = extract_dataset_classes(code)
    except DatasetExtractionError as exc:
        return _json_response({"error": str(exc)}, 400)
    return _json_response({"classes": classes})


@app.post("/api/dataset/instantiate")
def api_dataset_instantiate():
    payload = request.get_json() or {}
    code = payload.get("code", "")
    class_name = payload.get("class_name")
    if not code.strip() or not class_name:
        return _json_response({"error": "Code and class_name are required"}, 400)
    kwargs_train = payload.get("kwargs_train") or {}
    kwargs_val = payload.get("kwargs_val")
    kwargs_test = payload.get("kwargs_test")
    try:
        _, summary = instantiate_datasets(code, class_name, kwargs_train, kwargs_val, kwargs_test)
    except DatasetExtractionError as exc:
        return _json_response({"error": str(exc)}, 400)
    return _json_response({"summary": summary})


@app.post("/api/config/preview")
def api_config_preview():
    payload = request.get_json() or {}
    trainer_payload: Dict[str, object] = payload.get("trainer") or {}
    train_args: Dict[str, object] = payload.get("train_args") or {}
    eval_args: Optional[Dict[str, object]] = payload.get("eval_args")
    dataset_meta: Dict[str, object] = payload.get("dataset") or {}

    errors: Dict[str, str] = {}
    try:
        validate_trainer_payload(trainer_payload)
    except ValidationError as exc:
        errors["trainer"] = str(exc)
    try:
        validate_train_args(train_args)
    except ValidationError as exc:
        errors["train_args"] = str(exc)

    has_test = bool(dataset_meta.get("kwargs_test"))
    if eval_args is not None:
        try:
            validate_eval_args(eval_args, has_test)
        except ValidationError as exc:
            errors["eval_args"] = str(exc)

    if errors:
        return _json_response({"errors": errors}, 400)

    config = {
        "trainer": trainer_payload,
        "train": train_args,
        "dataset": dataset_meta,
    }
    if eval_args is not None:
        config["eval"] = eval_args
    return _json_response({"config": config})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
