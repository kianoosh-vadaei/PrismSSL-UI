# PrismSSL UI

A standalone HTML/CSS/JS dashboard for orchestrating PrismSSL self-supervised training
runs.  The interface mirrors the original Streamlit experience with sidebar
configuration panels, Ace-powered editors for datasets/backbones, live log
streaming, and theme toggling.

## Frontend usage

1. Open `public/index.html` in a browser or serve the `public/` directory with
   your static file server of choice.
2. Configure the run, dataset/backbone code snippets, and hyperparameters from
   the sidebar panels.
3. Toggle **Use Mock Backend** to try the UI without a running service.  Disable
   it to forward requests to the real API endpoints.

## Backend service

A FastAPI server that wraps the official `PrismSSL.<modality>.Trainer` classes is
provided in `backend/server.py`.  It accepts the exact payload emitted by the
frontend and performs the following steps:

- Executes uploaded dataset/backbone source snippets and instantiates them.
- Creates the appropriate PrismSSL `Trainer` implementation for the selected
  modality/method.
- Launches training in a background thread, capturing stdout/stderr into the log
  buffer exposed to the UI.

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

> **Note:** PrismSSL pulls in PyTorch; ensure you select the appropriate wheel
> for your environment if GPU acceleration is required.

### Run the API

```bash
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

The service implements the contract expected by the UI:

- `POST /api/trainer/init`
- `POST /api/trainer/build_backbone`
- `POST /api/train/start`
- `POST /api/train/stop`
- `GET /api/status`
- `GET /api/logs`
- `POST /api/logs/clear`

Keep the backend running while interacting with the frontend (with the mock
backend toggle disabled) to route requests into real PrismSSL training jobs.
