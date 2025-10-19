from __future__ import annotations

import contextlib
import importlib
import io
import logging
import threading
import time
import traceback
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from flask_socketio import SocketIO
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForPreTraining


def bert_loss_fn(outputs, batch):
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss
    prediction_scores = outputs.prediction_logits
    target = batch.get("labels")
    if target is None:
        target = batch["input_ids"]
    loss = torch.nn.functional.cross_entropy(
        prediction_scores.view(-1, prediction_scores.size(-1)), target.view(-1)
    )
    return loss


@dataclass
class TrainingStatus:
    running: bool = False
    progress: float = 0.0
    message: str = "Idle"
    current_epoch: int = 0
    current_iteration: int = 0
    start_time: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def elapsed_seconds(self) -> Optional[float]:
        if not self.start_time:
            return None
        return max(time.time() - self.start_time, 0)


class TrainingManager:
    def __init__(self, socketio: SocketIO) -> None:
        self.socketio = socketio
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.status = TrainingStatus()
        self.last_trainer_ctor: Optional[Dict[str, Any]] = None
        self.last_modality: Optional[str] = None
        self.use_generic: bool = False
        self.generic_args: Dict[str, Any] = {}
        self._current_trainer: Optional[Any] = None

    def start_training(
        self,
        modality: str,
        trainer_ctor: Dict[str, Any],
        train_args: Dict[str, Any],
        datasets: Dict[str, Any],
        use_generic: bool = False,
        generic_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            if self.status.running:
                raise RuntimeError("Training already running")
            self.status = TrainingStatus(running=True, start_time=time.time(), message="Training")
            self._stop_event.clear()
            self.last_trainer_ctor = trainer_ctor
            self.last_modality = modality
            self.use_generic = use_generic
            self.generic_args = generic_args or {}
            train_args_copy = dict(train_args)
            self._thread = threading.Thread(
                target=self._run_training,
                args=(modality, dict(trainer_ctor), train_args_copy, datasets, use_generic, self.generic_args),
                daemon=True,
            )
            self._thread.start()

    def stop_training(self) -> None:
        self._stop_event.set()
        trainer = self._current_trainer
        if trainer is not None:
            for attr in ("request_stop", "stop_training", "stop"):
                if hasattr(trainer, attr):
                    try:
                        getattr(trainer, attr)()
                    except Exception:  # noqa: BLE001
                        pass
                    break

    def is_running(self) -> bool:
        return self.status.running

    def get_status(self) -> Dict[str, Any]:
        elapsed = self.status.elapsed_seconds()
        return {
            "running": self.status.running,
            "progress": self.status.progress,
            "current_epoch": self.status.current_epoch,
            "current_iter": self.status.current_iteration,
            "message": self.status.message,
            "elapsed_seconds": elapsed,
        }

    def evaluate(self, eval_args: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
        if self.use_generic:
            raise RuntimeError("GenericSSLTrainer evaluation is not implemented")
        if not self.last_trainer_ctor or not self.last_modality:
            raise RuntimeError("Start a training run before evaluation")
        Trainer = self._import_trainer(self.last_modality)
        trainer = Trainer(**self.last_trainer_ctor)
        kwargs = dict(eval_args)
        extra = kwargs.pop("extra_kwargs", {}) if isinstance(kwargs.get("extra_kwargs"), dict) else {}
        try:
            result = trainer.evaluate(
                train_dataset=datasets["train"],
                test_dataset=datasets["test"],
                **kwargs,
                **extra,
            )
        except TypeError:
            # Some trainer implementations may not accept **extra
            result = trainer.evaluate(
                train_dataset=datasets["train"],
                test_dataset=datasets["test"],
                **kwargs,
            )
        return {"result": result}

    # Internal helpers

    def _import_trainer(self, modality: str):
        module = importlib.import_module(f"PrismSSL.{modality}.Trainer")
        return getattr(module, "Trainer")

    def _emit_progress(self, progress: float, eta: Optional[float] = None) -> None:
        self.status.progress = max(0.0, min(progress, 1.0))
        payload = {"progress": self.status.progress}
        if eta is not None:
            payload["eta_seconds"] = eta
        self.socketio.emit("train_progress", payload, to="train")

    def _emit_status(self, label: str, class_name: str) -> None:
        self.socketio.emit(
            "train_status",
            {
                "status": {
                    "label": label,
                    "className": class_name,
                },
                "elapsed_seconds": self.status.elapsed_seconds() or 0,
            },
            to="train",
        )

    def _run_training(
        self,
        modality: str,
        trainer_ctor: Dict[str, Any],
        train_args: Dict[str, Any],
        datasets: Dict[str, Any],
        use_generic: bool,
        generic_args: Dict[str, Any],
    ) -> None:
        with self._capture_output():
            try:
                self._emit_status("Training", "training")
                self.socketio.emit(
                    "train_log",
                    {"message": f"Starting training (generic={use_generic})"},
                    to="train",
                )
                if use_generic:
                    self._run_generic_training(train_args, datasets, generic_args)
                else:
                    self._run_prism_training(modality, trainer_ctor, train_args, datasets)
                if self._stop_event.is_set():
                    self.socketio.emit("train_log", {"message": "Training stopped."}, to="train")
                    self._emit_status("Stopped", "error")
                else:
                    self.socketio.emit("train_done", {"message": "Training complete."}, to="train")
                    self._emit_status("Done", "done")
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.socketio.emit("train_error", {"error": str(exc)}, to="train")
                self._emit_status("Error", "error")
            finally:
                self.status.running = False
                self._emit_progress(1.0 if not self._stop_event.is_set() else self.status.progress)

    def _run_prism_training(
        self,
        modality: str,
        trainer_ctor: Dict[str, Any],
        train_args: Dict[str, Any],
        datasets: Dict[str, Any],
    ) -> None:
        Trainer = self._import_trainer(modality)
        trainer = Trainer(**trainer_ctor)
        self._current_trainer = trainer
        local_args = dict(train_args)
        extra_kwargs = local_args.pop("extra_kwargs", {}) if isinstance(local_args.get("extra_kwargs"), dict) else {}
        kwargs = {**local_args}
        kwargs.update(extra_kwargs)
        self._emit_progress(0.01)
        self.socketio.emit("train_log", {"message": "Invoking PrismSSL trainer."}, to="train")
        try:
            trainer.train(
                train_dataset=datasets["train"],
                val_dataset=datasets.get("val"),
                **kwargs,
            )
        except TypeError:
            trainer.train(
                train_dataset=datasets["train"],
                **kwargs,
            )
        finally:
            self._current_trainer = None
        self._emit_progress(1.0)

    def _run_generic_training(
        self,
        train_args: Dict[str, Any],
        datasets: Dict[str, Any],
        generic_args: Dict[str, Any],
    ) -> None:
        model_name = generic_args.get("model_name", "bert-base-uncased")
        use_lora = generic_args.get("use_lora", True)
        epochs = int(generic_args.get("epochs", 10))
        batch_size = int(train_args.get("batch_size", 16))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertForPreTraining.from_pretrained(model_name)
        if use_lora:
            self.socketio.emit(
                "train_log",
                {"message": "LoRA toggle enabled (placeholder - no modification applied)."},
                to="train",
            )
        dataloader = self._build_generic_dataloader(datasets["train"], tokenizer, batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.get("learning_rate", 1e-4))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        total_steps = max(len(dataloader), 1) * epochs
        current_step = 0
        for epoch in range(epochs):
            if self._stop_event.is_set():
                break
            self.socketio.emit("train_log", {"message": f"Epoch {epoch + 1}/{epochs}"}, to="train")
            for batch in dataloader:
                if self._stop_event.is_set():
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = bert_loss_fn(outputs, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                current_step += 1
                progress = current_step / total_steps
                self.status.current_epoch = epoch
                self.status.current_iteration = current_step
                self._emit_progress(progress, None)
                self.socketio.emit(
                    "train_log",
                    {"message": f"step={current_step} loss={loss.item():.4f}"},
                    to="train",
                )
            if self._stop_event.is_set():
                break
        self._emit_progress(1.0 if not self._stop_event.is_set() else current_step / total_steps)

    def _build_generic_dataloader(self, dataset, tokenizer, batch_size: int) -> DataLoader:
        def collate(batch):
            first = batch[0]
            if isinstance(first, str):
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                enc["labels"] = enc["input_ids"].clone()
                return enc
            if isinstance(first, dict):
                collated: Dict[str, Any] = {}
                for key in first.keys():
                    values = [sample[key] for sample in batch]
                    if torch.is_tensor(values[0]):
                        collated[key] = torch.stack(values)
                    else:
                        collated[key] = torch.tensor(values)
                if "labels" not in collated and "input_ids" in collated:
                    collated["labels"] = collated["input_ids"].clone()
                return collated
            raise RuntimeError("Dataset samples must be str or dict for GenericSSLTrainer")

        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    @contextlib.contextmanager
    def _capture_output(self):
        stdout = sys.stdout
        stderr = sys.stderr
        redirect_out = SocketIOStreamRedirect(self.socketio, "train", stdout)
        redirect_err = SocketIOStreamRedirect(self.socketio, "train", stderr)
        logging_handler = SocketIOLoggingHandler(self.socketio, "train")
        logging_handler.setLevel(logging.NOTSET)
        root_logger = logging.getLogger()
        logging_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        sys.stdout = redirect_out
        sys.stderr = redirect_err
        root_logger.addHandler(logging_handler)
        try:
            yield
        finally:
            redirect_out.flush()
            redirect_err.flush()
            root_logger.removeHandler(logging_handler)
            logging_handler.close()
            sys.stdout = stdout
            sys.stderr = stderr


class SocketIOStreamRedirect(io.TextIOBase):
    def __init__(self, socketio: SocketIO, room: str, original):
        super().__init__()
        self.socketio = socketio
        self.room = room
        self.original = original
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            data = str(data)
        if self.original is not None:
            self.original.write(data)
        if not data:
            return 0
        with self._lock:
            self._buffer += data
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._emit(line)
        return len(data)

    def flush(self) -> None:  # noqa: D401
        if self.original is not None:
            self.original.flush()
        with self._lock:
            if self._buffer:
                self._emit(self._buffer)
                self._buffer = ""

    def _emit(self, text: str) -> None:
        message = text.rstrip("\r")
        if not message:
            return
        self.socketio.emit("train_log", {"message": message}, to=self.room)

    def writable(self) -> bool:  # noqa: D401
        return True


class SocketIOLoggingHandler(logging.Handler):
    def __init__(self, socketio: SocketIO, room: str) -> None:
        super().__init__()
        self.socketio = socketio
        self.room = room

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            message = self.format(record)
        except Exception:  # noqa: BLE001
            message = record.getMessage()
        if not message:
            return
        self.socketio.emit("train_log", {"message": message}, to=self.room)
