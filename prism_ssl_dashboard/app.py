from __future__ import annotations

from typing import Dict

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, join_room

from utils.dataset_loader import DatasetExtractionError, extract_dataset_classes, instantiate_datasets
from utils.training_runner import TrainingManager
from utils.validators import ValidationError, validate_eval_args, validate_train_args, validate_trainer_payload

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "prism-ssl-dashboard"
socketio = SocketIO(app, cors_allowed_origins="*")
training_manager = TrainingManager(socketio)

datasets: Dict[str, object] = {}

def _json_response(data, status=200):
    return jsonify(data), status


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("join")
def _join_room(payload):
    room = payload.get("room", "train")
    join_room(room)


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
        created, summary = instantiate_datasets(code, class_name, kwargs_train, kwargs_val, kwargs_test)
    except DatasetExtractionError as exc:
        return _json_response({"error": str(exc)}, 400)
    datasets.clear()
    datasets.update(created)
    return _json_response({"summary": summary})


@app.post("/api/train/start")
def api_train_start():
    if "train" not in datasets:
        return _json_response({"error": "Instantiate a training dataset first."}, 400)
    payload = request.get_json() or {}
    try:
        validate_trainer_payload(payload)
        validate_train_args(payload.get("train_args", {}))
    except ValidationError as exc:
        return _json_response({"error": str(exc)}, 400)
    if training_manager.is_running():
        return _json_response({"error": "Training already in progress"}, 409)
    try:
        training_manager.start_training(
            modality=payload["modality"],
            trainer_ctor=payload.get("trainer_ctor", {}),
            train_args=payload.get("train_args", {}),
            datasets=datasets,
            use_generic=payload.get("use_generic", False),
            generic_args=payload.get("generic", {}),
        )
    except Exception as exc:  # noqa: BLE001
        return _json_response({"error": str(exc)}, 500)
    return _json_response({"status": "started"})


@app.post("/api/train/stop")
def api_train_stop():
    if not training_manager.is_running():
        return _json_response({"error": "No training run active"}, 400)
    training_manager.stop_training()
    return _json_response({"status": "stopping"})


@app.get("/api/train/status")
def api_train_status():
    status = training_manager.get_status()
    return _json_response(status)


@app.post("/api/evaluate")
def api_evaluate():
    payload = request.get_json() or {}
    eval_args = payload.get("eval_args", {})
    if "train" not in datasets:
        return _json_response({"error": "Instantiate the training dataset first."}, 400)
    has_test = "test" in datasets
    try:
        validate_eval_args(eval_args, has_test)
    except ValidationError as exc:
        return _json_response({"error": str(exc)}, 400)
    try:
        result = training_manager.evaluate(eval_args, datasets)
    except Exception as exc:  # noqa: BLE001
        return _json_response({"error": str(exc)}, 500)
    return _json_response({"message": "Evaluation complete", "result": result})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
