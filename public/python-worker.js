importScripts("https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js");

let pyodideInstance = null;
let ready = false;
let stopFlag = false;

async function ensurePyodide() {
  if (pyodideInstance) {
    return pyodideInstance;
  }
  pyodideInstance = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/",
  });

  const emit = pyodideInstance.toPy((message) => {
    let payload = message;
    if (message && typeof message.toJs === "function") {
      payload = message.toJs({ dict_converter: Object.fromEntries });
      if (typeof message.destroy === "function") {
        message.destroy();
      }
    }
    self.postMessage({ type: "python-event", payload });
  });
  const checkStop = pyodideInstance.toPy(() => stopFlag);

  pyodideInstance.globals.set("emit", emit);
  pyodideInstance.globals.set("check_stop", checkStop);

  ready = true;
  self.postMessage({ type: "ready" });
  return pyodideInstance;
}

function resetStopFlag() {
  stopFlag = false;
}

function escapeBase64(value) {
  if (!value) return "";
  return btoa(unescape(encodeURIComponent(value)));
}

function buildTrainingScript(payload) {
  const configJson = JSON.stringify(payload);
  const configB64 = btoa(unescape(encodeURIComponent(configJson)));

  return `
import asyncio
import base64
import json
import math
import random
from js import emit, check_stop

config = json.loads(base64.b64decode("${configB64}").decode("utf-8"))
python_section = config.get("python", {})

def _decode_source(key):
    raw = python_section.get(key)
    if not raw:
        return ""
    return base64.b64decode(raw).decode("utf-8")

emit({"event": "status", "message": "Python runtime received the blueprint.", "run_name": config["global"]["runName"]})

for label, key in (("train dataset", "train_source"), ("validation dataset", "val_source"), ("backbone", "backbone_source")):
    source_code = _decode_source(key)
    if not source_code.strip():
        continue
    emit({"event": "log", "message": f"Compiling {label} snippet..."})
    exec(source_code, globals(), globals())
    emit({"event": "log", "message": f"Loaded {label}."})

total_epochs = int(config["train"]["epochs"])
if total_epochs <= 0:
    total_epochs = 1

async def simulate_training():
    emit({"event": "status", "message": "Training loop started."})
    for epoch in range(1, total_epochs + 1):
        if check_stop():
            emit({"event": "status", "message": "Training cancelled by user.", "epoch": epoch})
            return {"status": "stopped", "epoch": epoch}
        await asyncio.sleep(0.25)
        train_loss = round(max(0.0005, 1.0 / (epoch + 1) + random.uniform(-0.02, 0.02)), 4)
        val_loss = round(max(0.0005, 1.1 / (epoch + 1) + random.uniform(-0.03, 0.03)), 4)
        emit({
            "event": "metrics",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "total_epochs": total_epochs,
        })
        emit({
            "event": "log",
            "message": f"Epoch {epoch}/{total_epochs} — train_loss={train_loss:.4f}, val_loss={val_loss:.4f}",
        })
    emit({"event": "status", "message": "Training finished successfully.", "epoch": total_epochs})
    return {"status": "completed", "epoch": total_epochs}

result = await simulate_training()
result
`;
}

function buildDiagnosticScript(code) {
  const codeB64 = escapeBase64(code);
  return `
import base64
import io
import json
import sys

source = base64.b64decode("${codeB64}").decode("utf-8")
buffer = io.StringIO()
status = "ok"
output = ""

original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = buffer
sys.stderr = buffer

try:
    namespace = {}
    exec(source, namespace, namespace)
except Exception as exc:
    status = "error"
    output = buffer.getvalue() + f"\n{type(exc).__name__}: {exc}"
else:
    output = buffer.getvalue() or "(no output)"
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr

json.dumps({"status": status, "output": output})
`;
}

async function runTraining(payload) {
  const pyodide = await ensurePyodide();
  resetStopFlag();
  self.postMessage({ type: "runtime-state", payload: { status: "running" } });
  try {
    const script = buildTrainingScript(payload);
    const result = await pyodide.runPythonAsync(script);
    let response = result;
    if (result && typeof result.toJs === "function") {
      response = result.toJs({ dict_converter: Object.fromEntries });
      if (typeof result.destroy === "function") {
        result.destroy();
      }
    }
    self.postMessage({ type: "train-complete", payload: response });
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    self.postMessage({ type: "python-error", error: message });
  } finally {
    self.postMessage({ type: "runtime-state", payload: { status: "idle" } });
  }
}

async function runDiagnostic(id, code) {
  const pyodide = await ensurePyodide();
  try {
    const script = buildDiagnosticScript(code);
    const result = await pyodide.runPythonAsync(script);
    let payload = result;
    if (result && typeof result.toJs === "function") {
      payload = result.toJs({ dict_converter: Object.fromEntries });
      if (typeof result.destroy === "function") {
        result.destroy();
      }
    }
    self.postMessage({ type: "diagnostic-result", id, result: payload });
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    self.postMessage({ type: "diagnostic-error", id, error: message });
  }
}

self.onmessage = async (event) => {
  const { type, payload, id } = event.data || {};
  switch (type) {
    case "init":
      await ensurePyodide();
      break;
    case "start-training":
      await runTraining(payload);
      break;
    case "stop-training":
      stopFlag = true;
      break;
    case "run-diagnostic":
      await runDiagnostic(id, payload?.code || "");
      break;
    default:
      break;
  }
};

