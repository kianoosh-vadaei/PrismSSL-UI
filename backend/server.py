"""FastAPI backend that bridges the PrismSSL Trainer with the web UI.

The frontend posts the full configuration payload (run metadata, dataset/backbone
source snippets, and training hyperparameters).  This service converts that
payload into real Python objects and executes the requested training lifecycle
operations.
"""

from __future__ import annotations

import ctypes
import io
import math
import re
import threading
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from types import ModuleType
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


MODALITY_MODULE = {
    "audio": "audio",
    "vision": "vision",
    "graph": "graph",
    "cross-modal": "cross_modal",
}

EPOCH_PATTERN = re.compile(r"epoch\\s*(\\d+)", re.IGNORECASE)
TRAIN_LOSS_PATTERN = re.compile(r"train(?:ing)?[_\s-]*loss\\s*[:=]\\s*([0-9]*\\.?[0-9]+)", re.IGNORECASE)
VAL_LOSS_PATTERN = re.compile(r"val(?:idation)?[_\s-]*loss\\s*[:=]\\s*([0-9]*\\.?[0-9]+)", re.IGNORECASE)


@dataclass
class TrainerStatus:
    is_running: bool = False
    current_epoch: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "current_epoch": self.current_epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "message": self.message,
        }


class LogSink(io.TextIOBase):
    """Thread-safe text sink that stores log lines and updates the status."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self._buffer = ""

    def write(self, data: str) -> int:  # type: ignore[override]
        if not data:
            return 0
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._callback(line.strip())
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        if self._buffer.strip():
            self._callback(self._buffer.strip())
        self._buffer = ""


class TrainerService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._status = TrainerStatus()
        self._logs: List[str] = []
        self._trainer = None
        self._train_dataset = None
        self._val_dataset = None
        self._train_thread: Optional[threading.Thread] = None
        self._train_thread_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_backbone(self, payload: Dict[str, Any]) -> Dict[str, str]:
        self._create_backbone(payload)
        return {"status": "ok"}

    def init_trainer(self, payload: Dict[str, Any]) -> Dict[str, str]:
        with self._lock:
            self._logs.clear()
            datasets = payload.get("datasets") or {}
            backbone_payload = payload.get("backbone") or {}
            self._train_dataset, self._val_dataset = self._create_datasets(datasets)
            backbone = self._create_backbone(backbone_payload) if backbone_payload else None
            trainer_cls = self._load_trainer_class(payload.get("modality"))
            trainer_kwargs = self._build_trainer_kwargs(payload, backbone)
            self._trainer = trainer_cls(**trainer_kwargs)
            self._status = TrainerStatus(is_running=False, current_epoch=0)
            self._append_log("Trainer initialized successfully.")
        return {"status": "ok"}

    def start_training(self, payload: Dict[str, Any]) -> Dict[str, str]:
        with self._lock:
            if self._trainer is None:
                raise HTTPException(status_code=400, detail="Trainer not initialized.")
            if self._train_thread and self._train_thread.is_alive():
                return {"status": "already_running"}
            self._status = TrainerStatus(is_running=True, current_epoch=0)
            self._logs.clear()
            self._append_log("Launching training run...")
            train_kwargs = self._build_train_kwargs(payload)
            self._train_thread = threading.Thread(
                target=self._run_training,
                args=(train_kwargs,),
                daemon=True,
            )
            self._train_thread.start()
            self._train_thread_id = self._train_thread.ident
        return {"status": "ok"}

    def stop_training(self) -> Dict[str, str]:
        with self._lock:
            if not self._train_thread or not self._train_thread.is_alive():
                return {"status": "idle"}
            if self._train_thread_id is None:
                return {"status": "idle"}
            interrupted = _raise_keyboard_interrupt(self._train_thread_id)
            if interrupted:
                self._append_log("Stop requested by user.")
                return {"status": "stopped"}
            return {"status": "idle"}

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return self._status.to_dict()

    def get_logs(self) -> Dict[str, str]:
        with self._lock:
            return {"logs": "\n".join(self._logs)}

    def clear_logs(self) -> Dict[str, str]:
        with self._lock:
            self._logs.clear()
        return {"status": "cleared"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _run_training(self, train_kwargs: Dict[str, Any]) -> None:
        assert self._trainer is not None
        sink = LogSink(self._handle_log)
        try:
            with redirect_stdout(sink), redirect_stderr(sink):
                self._trainer.train(
                    train_dataset=self._train_dataset,
                    val_dataset=self._val_dataset,
                    **train_kwargs,
                )
            self._append_log("Training completed.")
        except KeyboardInterrupt:
            self._append_log("Training interrupted.")
        except Exception as exc:  # pylint: disable=broad-except
            self._append_log(f"Training failed: {exc}")
        finally:
            sink.flush()
            with self._lock:
                self._status.is_running = False

    def _handle_log(self, line: str) -> None:
        self._append_log(line)
        self._update_status_from_line(line)

    def _append_log(self, message: str) -> None:
        timestamp = datetime.utcnow().isoformat()
        entry = f"[{timestamp}] {message}"
        with self._lock:
            self._logs.append(entry)
            if len(self._logs) > 2000:
                # Keep most recent logs to avoid unbounded memory usage.
                self._logs[:] = self._logs[-2000:]

    def _update_status_from_line(self, line: str) -> None:
        epoch_match = EPOCH_PATTERN.search(line)
        if epoch_match:
            try:
                epoch = int(epoch_match.group(1))
            except ValueError:
                epoch = None
            if epoch is not None:
                self._status.current_epoch = epoch
        train_loss_match = TRAIN_LOSS_PATTERN.search(line)
        if train_loss_match:
            try:
                self._status.train_loss = float(train_loss_match.group(1))
            except ValueError:
                pass
        val_loss_match = VAL_LOSS_PATTERN.search(line)
        if val_loss_match:
            try:
                self._status.val_loss = float(val_loss_match.group(1))
            except ValueError:
                pass

    def _create_datasets(self, datasets_payload: Dict[str, Any]):
        train_payload = datasets_payload.get("train")
        if not train_payload:
            raise HTTPException(status_code=400, detail="Train dataset payload missing.")
        train_class = train_payload.get("class_name")
        if not train_class:
            raise HTTPException(status_code=400, detail="Train dataset class missing.")
        train_source = train_payload.get("source")
        if not train_source:
            raise HTTPException(status_code=400, detail="Train dataset source missing.")
        train_dataset = self._instantiate_from_source(
            train_source,
            train_class,
            train_payload.get("kwargs") or {},
            module_name="train_dataset",
        )
        val_payload = datasets_payload.get("val")
        val_dataset = None
        if val_payload:
            val_class = val_payload.get("class_name")
            if not val_class:
                raise HTTPException(status_code=400, detail="Validation dataset class missing.")
            val_source = val_payload.get("source")
            if not val_source:
                raise HTTPException(status_code=400, detail="Validation dataset source missing.")
            val_dataset = self._instantiate_from_source(
                val_source,
                val_class,
                val_payload.get("kwargs") or {},
                module_name="val_dataset",
            )
        return train_dataset, val_dataset

    def _create_backbone(self, backbone_payload: Dict[str, Any]):
        if not backbone_payload:
            return None
        mode = backbone_payload.get("mode", "none")
        if mode != "upload":
            return None
        class_name = backbone_payload.get("class_name")
        if not class_name:
            raise HTTPException(status_code=400, detail="Backbone class missing.")
        source = backbone_payload.get("source")
        if not source:
            raise HTTPException(status_code=400, detail="Backbone source missing.")
        return self._instantiate_from_source(
            source,
            class_name,
            backbone_payload.get("kwargs") or {},
            module_name="backbone",
        )

    def _instantiate_from_source(
        self,
        source: str,
        class_name: str,
        kwargs: Dict[str, Any],
        module_name: str,
    ):
        if not source or not source.strip():
            raise HTTPException(status_code=400, detail=f"Source code missing for {module_name}.")
        module = ModuleType(module_name)
        exec(compile(source, module_name, "exec"), module.__dict__)  # noqa: S102 - dynamic exec required
        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:  # pylint: disable=raise-missing-from
            raise HTTPException(status_code=400, detail=f"Class {class_name} not found in {module_name} source")
        try:
            return cls(**kwargs)
        except TypeError as exc:  # pylint: disable=raise-missing-from
            raise HTTPException(status_code=400, detail=f"Failed to instantiate {class_name}: {exc}")

    def _load_trainer_class(self, modality: Optional[str]):
        if not modality:
            raise HTTPException(status_code=400, detail="Modality not provided.")
        module_key = MODALITY_MODULE.get(modality.lower())
        if not module_key:
            raise HTTPException(status_code=400, detail=f"Unsupported modality '{modality}'.")
        module_path = f"PrismSSL.{module_key}.Trainer"
        try:
            module = import_module(module_path)
        except ModuleNotFoundError as exc:  # pylint: disable=raise-missing-from
            raise HTTPException(status_code=400, detail=f"Unable to import module {module_path}.")
        try:
            return getattr(module, "Trainer")
        except AttributeError as exc:  # pylint: disable=raise-missing-from
            raise HTTPException(status_code=400, detail=f"Trainer class missing in {module_path}.")

    def _build_trainer_kwargs(self, payload: Dict[str, Any], backbone: Any) -> Dict[str, Any]:
        run_cfg = payload.get("run") or {}
        kwargs = {
            "method": payload.get("method"),
            "backbone": backbone,
            "variant": payload.get("variant") or "base",
            "save_dir": run_cfg.get("save_dir") or ".",
            "checkpoint_interval": _sanitize_int(run_cfg.get("checkpoint_interval"), default=10),
            "reload_checkpoint": bool(run_cfg.get("reload_checkpoint")),
            "verbose": bool(run_cfg.get("verbose", True)),
            "mixed_precision_training": bool(run_cfg.get("mixed_precision", True)),
            "wandb_project": run_cfg.get("wandb_project"),
            "wandb_entity": run_cfg.get("wandb_entity"),
            "wandb_mode": run_cfg.get("wandb_mode", "online"),
            "wandb_run_name": run_cfg.get("wandb_run_name") or run_cfg.get("name"),
            "wandb_config": run_cfg.get("wandb_config"),
            "wandb_notes": run_cfg.get("wandb_notes"),
            "wandb_tags": run_cfg.get("wandb_tags"),
            "use_data_parallel": bool(run_cfg.get("use_data_parallel")),
            "num_workers": _sanitize_int(run_cfg.get("num_workers")),
        }
        return kwargs

    def _build_train_kwargs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        train_cfg = payload.get("train") or {}
        kwargs = {
            "batch_size": _sanitize_int(train_cfg.get("batch_size"), default=16),
            "start_epoch": _sanitize_int(train_cfg.get("start_epoch"), default=0),
            "epochs": _sanitize_int(train_cfg.get("epochs"), default=1),
            "start_iteration": _sanitize_int(train_cfg.get("start_iteration"), default=0),
            "learning_rate": _sanitize_float(train_cfg.get("lr"), default=1e-4),
            "weight_decay": _sanitize_float(train_cfg.get("weight_decay"), default=1e-2),
            "optimizer": train_cfg.get("optimizer", "adamw"),
            "use_hpo": bool(train_cfg.get("use_hpo")),
            "n_trials": _sanitize_int(train_cfg.get("n_trials"), default=20),
            "tuning_epochs": _sanitize_int(train_cfg.get("tuning_epochs"), default=5),
            "use_embedding_logger": bool(train_cfg.get("use_embedding_logger")),
        }
        extra_kwargs = train_cfg.get("extra_kwargs") or {}
        kwargs.update(extra_kwargs)
        return kwargs


def _sanitize_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int,)):
        return value
    if isinstance(value, float) and not math.isnan(value):
        return int(value)
    return default


def _sanitize_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    return default


def _raise_keyboard_interrupt(thread_id: int) -> bool:
    """Attempt to asynchronously raise KeyboardInterrupt in the target thread."""

    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(KeyboardInterrupt))
    if res == 0:
        return False
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
        return False
    return True


service = TrainerService()
app = FastAPI(title="PrismSSL Trainer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/trainer/init")
async def init_trainer(payload: Dict[str, Any]):
    return service.init_trainer(payload)


@app.post("/api/trainer/build_backbone")
async def build_backbone(payload: Dict[str, Any]):
    return service.build_backbone(payload)


@app.post("/api/train/start")
async def start_train(payload: Dict[str, Any]):
    return service.start_training(payload)


@app.post("/api/train/stop")
async def stop_train():
    return service.stop_training()


@app.get("/api/status")
async def fetch_status():
    return service.get_status()


@app.get("/api/logs")
async def fetch_logs():
    return service.get_logs()


@app.post("/api/logs/clear")
async def clear_logs():
    return service.clear_logs()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
