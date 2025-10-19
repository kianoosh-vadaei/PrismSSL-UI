const WORKER_URL = new URL("./python-worker.js", import.meta.url);

export class PythonRuntime {
  constructor() {
    this.worker = new Worker(WORKER_URL, { type: "module" });
    this.listeners = new Set();
    this.pendingDiagnostics = new Map();
    this.diagnosticCounter = 0;
    this.ready = false;
    this.initialized = false;

    this.worker.onmessage = (event) => {
      this.handleMessage(event.data);
    };

    this.worker.onerror = (error) => {
      this.emit({ type: "python-error", error: error.message || String(error) });
    };
  }

  emit(event) {
    this.listeners.forEach((listener) => {
      try {
        listener(event);
      } catch (error) {
        console.warn("PythonRuntime listener error", error);
      }
    });
  }

  handleMessage(message) {
    switch (message?.type) {
      case "ready": {
        this.ready = true;
        this.emit({ type: "ready" });
        break;
      }
      case "python-event": {
        this.emit({ type: "python-event", payload: message.payload });
        break;
      }
      case "runtime-state": {
        this.emit({ type: "runtime-state", payload: message.payload });
        break;
      }
      case "python-error": {
        this.emit({ type: "python-error", error: message.error });
        break;
      }
      case "train-complete": {
        this.emit({ type: "train-complete", payload: message.payload });
        break;
      }
      case "diagnostic-result": {
        const entry = this.pendingDiagnostics.get(message.id);
        if (entry) {
          entry.resolve(message.result);
          this.pendingDiagnostics.delete(message.id);
        }
        this.emit({ type: "diagnostic-result", payload: message.result });
        break;
      }
      case "diagnostic-error": {
        const entry = this.pendingDiagnostics.get(message.id);
        if (entry) {
          entry.reject(message.error);
          this.pendingDiagnostics.delete(message.id);
        }
        this.emit({ type: "diagnostic-error", error: message.error });
        break;
      }
      default:
        break;
    }
  }

  async init() {
    if (this.initialized) {
      return;
    }
    this.initialized = true;
    this.worker.postMessage({ type: "init" });
  }

  on(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  startTraining(config) {
    this.worker.postMessage({ type: "start-training", payload: config });
  }

  stopTraining() {
    this.worker.postMessage({ type: "stop-training" });
  }

  runDiagnostic(code) {
    const id = ++this.diagnosticCounter;
    return new Promise((resolve, reject) => {
      this.pendingDiagnostics.set(id, { resolve, reject });
      this.worker.postMessage({ type: "run-diagnostic", id, payload: { code } });
    });
  }

  destroy() {
    this.listeners.clear();
    this.worker.terminate();
  }
}

