# Python worker runtime

The dashboard no longer depends on a standalone API. Instead, a Web Worker
(`python-worker.js`) loads Pyodide and executes the assembled configuration
payload entirely inside the browser.

## Event flow

1. The UI collects all run metadata, dataset/backbone snippets, and
   hyperparameters.
2. Pressing **Launch training** sends the payload to the worker via `postMessage`.
3. The worker executes the snippets, simulates a PrismSSL-style training loop,
   and emits progress events (`status`, `metrics`, `log`).
4. Activity logs, metric summaries, and progress bars update live in the
   workspace.
5. The **Run Python check** action uses the same worker to execute ad-hoc Python
   snippets and displays stdout/stderr inline.

Use this file as a reference if you plan to replace the synthetic loop with
real PrismSSL logic.
