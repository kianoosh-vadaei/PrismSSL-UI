from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class PretextTrainingError(Exception):
    """Raised when background pretext training cannot be launched."""


def _write_config_snapshot(config: Dict[str, Any]) -> Path:
    trainer = config.get("trainer", {}) or {}
    ctor = trainer.get("trainer_ctor", {}) or {}
    save_dir = Path(ctor.get("save_dir") or "./checkpoints").expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    snapshot_path = save_dir / f"pretext_config_{timestamp}.json"
    snapshot_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return snapshot_path


def _simulate_training(config: Dict[str, Any]) -> None:
    train_args = config.get("train", {}) or {}
    epochs = int(train_args.get("epochs") or 1)
    simulated_epochs = max(1, min(epochs, 5))
    for epoch in range(simulated_epochs):
        print(
            f"[PretextTraining] Simulating epoch {epoch + 1}/{simulated_epochs}",
            flush=True,
        )
        time.sleep(1)
    if epochs > simulated_epochs:
        print(
            f"[PretextTraining] Simulation truncated {epochs - simulated_epochs} additional epoch(s).",
            flush=True,
        )
    print("[PretextTraining] Background training finished.", flush=True)


def run_pretext_training(config: Dict[str, Any]) -> None:
    """Entry point executed in a background process.

    The function stores the provided configuration snapshot to disk. If the
    environment variable ``PRISMSSL_TRAIN_COMMAND`` is defined it will be used
    to launch an external training command. Otherwise a short simulated
    training loop runs so the behaviour is observable out-of-the-box.
    """

    try:
        snapshot_path = _write_config_snapshot(config)
        print(
            f"[PretextTraining] Configuration snapshot written to {snapshot_path}",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - best-effort logging only
        raise PretextTrainingError(f"Unable to persist configuration: {exc}") from exc

    command = os.environ.get("PRISMSSL_TRAIN_COMMAND")
    if not command:
        _simulate_training(config)
        return

    expanded_command = command.format(config=str(snapshot_path))
    print(f"[PretextTraining] Launching command: {expanded_command}", flush=True)

    # Import here to avoid unnecessary overhead for the simulation path.
    import subprocess

    try:
        completed = subprocess.run(expanded_command, shell=True, check=True)
        print(
            f"[PretextTraining] External command finished with return code {completed.returncode}",
            flush=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime guard
        raise PretextTrainingError(
            f"Training command failed with exit code {exc.returncode}"
        ) from exc
