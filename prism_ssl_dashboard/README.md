# Prism-SSL Training Dashboard

A modern AI-themed dashboard for configuring, launching, and monitoring Prism-SSL training runs from the browser. The app combines a Flask + Socket.IO backend with a responsive HTML/CSS/JS frontend that streams real-time logs, manages datasets, and supports both Prism-SSL modality trainers and an optional GenericSSLTrainer path leveraging Transformers.

## Features
- Paste custom PyTorch `Dataset` code, detect subclasses, and instantiate train/val/test datasets.
- Configure trainer, training, and evaluation parameters with validation and helpful defaults.
- Live training console with progress bar, status chips, and cooperative stop handling.
- Optional GenericSSLTrainer path using BERT with LoRA toggle.
- Theme toggle (light/dark), config persistence via localStorage, and JSON import/export.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Notes
- Executing dataset code executes arbitrary Python on the server. Only run trusted code in an isolated environment.
- Prism-SSL training often requires GPUs for practical runtimes.
- Ensure required data files are accessible relative to the server process when instantiating datasets.
