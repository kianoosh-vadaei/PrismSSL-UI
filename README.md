# PrismSSL Toolbox UI

A standalone HTML/CSS/JS dashboard for orchestrating PrismSSL-style training
runs without any backend services. The interface ships as a single-page
application, renders a toolbox-style control panel, and executes Python payloads
inside a Web Worker powered by Pyodide.

## Getting started

1. Serve the `public/` directory with any static file server or open
   `public/index.html` directly in a browser.
2. Wait for the **Python runtime ready** badge in the header. The first load can
   take a few seconds while Pyodide downloads.
3. Configure the run metadata, dataset/backbone snippets, and hyperparameters
   from the toolbox column.
4. Press **Launch training** to send the configuration into the worker.
   Training progress, synthetic metrics, and activity logs appear in the
   workspace panels.
5. Use the **Run Python check** action to execute an ad-hoc diagnostic snippet
   and verify the embedded Python interpreter is responsive.

The dashboard persists all form fields and editor content in `localStorage`,
allowing you to refresh the page without losing your blueprint.

## Architecture

- **No backend dependencies** — a `python-worker.js` module hosts Pyodide and
  simulates the PrismSSL trainer loop entirely in the browser. The worker
  consumes the complete payload assembled from the UI and streams progress
  events back to the main thread.
- **Ace-powered editors** for train/validation datasets and backbone code.
  Snippets are executed inside the worker to validate that Python is running.
- **Theme-aware design** following the AI-inspired dark/light palettes described
  in the original brief.

## Development tips

- Customise the diagnostic snippet in the workspace to quickly test new Python
  logic. The console output and exceptions are surfaced inline.
- The worker currently emits synthetic metrics; integrate real PrismSSL logic by
  expanding `public/python-worker.js` with the appropriate package imports once
  the libraries are available in Pyodide.

## Testing

No automated tests are bundled. Open the page in a modern browser (Chromium,
Firefox, or Safari) to validate behaviour manually.
