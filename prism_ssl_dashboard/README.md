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

## Using the dashboard

1. **Provide your dataset implementation**
   - Paste the Python file contents that contain one or more `torch.utils.data.Dataset` subclasses into the *Dataset Code* panel.
   - Click **Detect Classes** to parse the file. A dropdown with the detected dataset classes appears if the code is valid.
   - Choose the class you want to instantiate for the train/val/test splits and fill in JSON kwargs (for example `{"root": "./data", "split": "train"}`).
   - Press **Instantiate Datasets**. The backend executes the code in an isolated module, instantiates the selected class, and returns dataset lengths or errors.

2. **Configure the trainer**
   - Open the **Configure Trainer** tab and set the Prism-SSL modality, constructor parameters, and optional Weights & Biases settings. Defaults are pre-populated; hover tooltips explain each field.
   - Toggle **Use GenericSSLTrainer (Transformers)** if you want to run the BERT-based example path instead of a modality-specific Prism-SSL trainer.

3. **Set training parameters**
   - In the **Train / Tune** tab, adjust batch size, epochs, optimizer choice, learning rate, hyperparameter optimization options, and any JSON passthrough kwargs you need.
   - Press **Start Training**. A background worker begins training and the log console streams Socket.IO events with timestamps, progress, and status updates. Use **Stop** to request a graceful shutdown and **Resume** to continue with the preserved state.

4. **Run evaluation (optional)**
   - After training, switch to the **Evaluate** tab. Provide evaluation-specific arguments (e.g., `num_classes`) and click **Run Evaluation**. Results and logs appear in the console.

5. **Manage configurations**
   - The dashboard automatically persists your last-used configuration in `localStorage`. Use the **Export Config** and **Import Config** controls to share or reload setups.

### Tips
- The status chip above the log console reflects the current worker state (Idle, Training, Stopped, Done, or Error).
- Toggle the theme switch in the header to alternate between light and dark AI-themed palettes. Your preference is saved locally.
- If `use_hpo` is enabled, make sure to provide valid values for `n_trials` and `tuning_epochs`; the frontend prevents submission until constraints are satisfied.

## Notes
- Executing dataset code executes arbitrary Python on the server. Only run trusted code in an isolated environment.
- Prism-SSL training often requires GPUs for practical runtimes.
- Ensure required data files are accessible relative to the server process when instantiating datasets.
