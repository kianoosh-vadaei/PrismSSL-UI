import { PythonRuntime } from "./api.js";
import { stripAnsi } from "./ansi.js";
import { createAceEditor, updateEditorTheme } from "./editors.js";

const METHOD_OPTIONS = {
  audio: ["Wav2Vec2", "HuBERT", "SimCLR", "COLA", "EAT"],
  vision: ["SimCLR", "BYOL", "MoCoV2", "DINO", "BarlowTwins", "SwAV", "SimSiam"],
  graph: ["GraphCL"],
  "cross-modal": ["CLIP", "SLIP", "ALBEF", "SIMVLM", "UNITER_VQA", "VSE", "CLAP", "AUDIO_CLIP", "WAV2CLIP"],
};

const STORAGE_KEY = "toolbox-config-v1";
const THEME_KEY = "mkssl-theme";
const EDITOR_KEYS = {
  train: "toolbox-train-editor",
  val: "toolbox-val-editor",
  backbone: "toolbox-backbone-editor",
};

const DEFAULT_CONFIG = () => ({
  runName: `toolbox_${new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 15)}`,
  saveDir: "checkpoints",
  checkpointInterval: 10,
  reloadCheckpoint: false,
  mixedPrecision: true,
  verbose: true,
  useDataParallel: false,
  numWorkers: 8,
  modality: "audio",
  method: "Wav2Vec2",
  variant: "base",
  trainClass: "DummyDataset",
  trainKwargs: "{\n  \"length\": 500,\n  \"input_dim\": 64\n}",
  valClass: "",
  valKwargs: "",
  backboneMode: "none",
  backboneClass: "MyBackbone",
  backboneKwargs: "{\n  \"hidden_dim\": 256\n}",
  batchSize: 16,
  epochs: 10,
  lr: 0.001,
  weightDecay: 0.01,
  optimizer: "adamw",
  useHpo: false,
  nTrials: 20,
  tuningEpochs: 5,
  embeddingLogger: false,
  wandbProject: "",
  wandbEntity: "",
  wandbMode: "offline",
  wandbRunName: "",
  wandbNotes: "",
  wandbTags: "[\n  \"experiment\",\n  \"toolbox\"\n]",
  wandbConfig: "{\n  \"dropout\": 0.1\n}",
});

const DEFAULT_TRAIN_CODE = `import torch\nfrom torch.utils.data import Dataset\n\nclass DummyDataset(Dataset):\n    def __init__(self, length=500, input_dim=64):\n        self.length = length\n        self.input_dim = input_dim\n\n    def __len__(self):\n        return self.length\n\n    def __getitem__(self, idx):\n        return torch.randn(self.input_dim), torch.randint(0, 10, (1,))\n`;

const DEFAULT_VAL_CODE = `import torch\nfrom torch.utils.data import Dataset\n\nclass DummyValDataset(Dataset):\n    def __len__(self):\n        return 200\n\n    def __getitem__(self, idx):\n        return torch.randn(64), torch.randint(0, 10, (1,))\n`;

const DEFAULT_BACKBONE_CODE = `import torch.nn as nn\n\nclass MyBackbone(nn.Module):\n    def __init__(self, hidden_dim=256):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(64, hidden_dim),\n            nn.ReLU(),\n            nn.Linear(hidden_dim, hidden_dim)\n        )\n\n    def forward(self, x):\n        return self.net(x)\n`;

const store = {
  config: loadStoredConfig(),
  validation: {
    trainKwargsError: "",
    valKwargsError: "",
    backboneKwargsError: "",
    wandbConfigError: "",
    wandbTagsError: "",
  },
  runtime: {
    isRunning: false,
    currentEpoch: 0,
    trainLoss: null,
    valLoss: null,
    totalEpochs: 0,
  },
  logs: [],
};

const runtime = new PythonRuntime();
let editors = { train: null, val: null, backbone: null };

const persistStore = debounce(() => {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(store.config));
  } catch (error) {
    console.warn("Failed to persist configuration", error);
  }
}, 200);

document.addEventListener("DOMContentLoaded", () => {
  runtime.on(handleRuntimeEvent);
  runtime.init();
  initTheme();
  initFormBindings();
  initEditors();
  initButtons();
  populateMethodOptions(store.config.modality);
  refreshMetrics();
  refreshPythonStatus("loading", "Booting Python environment…");
});

function loadStoredConfig() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return DEFAULT_CONFIG();
    }
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_CONFIG(), ...parsed };
  } catch (error) {
    console.warn("Unable to read stored configuration", error);
    return DEFAULT_CONFIG();
  }
}

function debounce(fn, wait = 200) {
  let handle;
  return (...args) => {
    clearTimeout(handle);
    handle = setTimeout(() => fn(...args), wait);
  };
}

function $(selector) {
  return document.querySelector(selector);
}

function initTheme() {
  const stored = window.localStorage.getItem(THEME_KEY) || "dark";
  applyTheme(stored);
  const toggle = $("#theme-toggle");
  toggle?.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    applyTheme(next);
  });
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  window.localStorage.setItem(THEME_KEY, theme);
  const icon = document.querySelector("[data-theme-icon]");
  if (icon) {
    icon.textContent = theme === "dark" ? "🌙" : "☀️";
  }
  updateEditorTheme(Object.values(editors), theme);
}

function initEditors() {
  editors.train = createAceEditor({
    elementId: "train-editor",
    storageKey: EDITOR_KEYS.train,
    defaultValue: DEFAULT_TRAIN_CODE,
  });
  editors.val = createAceEditor({
    elementId: "val-editor",
    storageKey: EDITOR_KEYS.val,
    defaultValue: DEFAULT_VAL_CODE,
  });
  editors.backbone = createAceEditor({
    elementId: "backbone-editor",
    storageKey: EDITOR_KEYS.backbone,
    defaultValue: DEFAULT_BACKBONE_CODE,
  });
}

function initButtons() {
  $("#btn-start")?.addEventListener("click", () => {
    startTraining();
  });
  $("#btn-stop")?.addEventListener("click", () => {
    runtime.stopTraining();
    appendLog("Stop requested by user.");
  });
  $("#btn-clear-logs")?.addEventListener("click", () => {
    store.logs = [];
    renderLogs();
  });
  $("#btn-run-diagnostic")?.addEventListener("click", runDiagnosticSnippet);
}

function initFormBindings() {
  const bindings = [
    { id: "run-name", key: "runName" },
    { id: "save-dir", key: "saveDir" },
    { id: "variant", key: "variant" },
    { id: "checkpoint-interval", key: "checkpointInterval", cast: Number },
    { id: "num-workers", key: "numWorkers", cast: Number },
    { id: "batch-size", key: "batchSize", cast: Number },
    { id: "epochs", key: "epochs", cast: Number },
    { id: "learning-rate", key: "lr", cast: Number },
    { id: "weight-decay", key: "weightDecay", cast: Number },
    { id: "optimizer", key: "optimizer" },
    { id: "n-trials", key: "nTrials", cast: Number },
    { id: "tuning-epochs", key: "tuningEpochs", cast: Number },
    { id: "train-class", key: "trainClass" },
    { id: "val-class", key: "valClass" },
    { id: "backbone-class", key: "backboneClass" },
    { id: "wandb-project", key: "wandbProject" },
    { id: "wandb-entity", key: "wandbEntity" },
    { id: "wandb-mode", key: "wandbMode" },
    { id: "wandb-run-name", key: "wandbRunName" },
    { id: "wandb-notes", key: "wandbNotes" },
  ];

  bindings.forEach(({ id, key, cast }) => {
    const el = document.getElementById(id);
    if (!el) return;
    const value = store.config[key];
    el.value = value ?? "";
    el.addEventListener("input", (event) => {
      const raw = event.target.value;
      store.config[key] = cast ? cast(raw || 0) : raw;
      persistStore();
      refreshMetrics();
    });
  });

  const checkboxBindings = [
    { id: "reload-checkpoint", key: "reloadCheckpoint" },
    { id: "mixed-precision", key: "mixedPrecision" },
    { id: "verbose-mode", key: "verbose" },
    { id: "data-parallel", key: "useDataParallel" },
    { id: "use-hpo", key: "useHpo" },
    { id: "embedding-logger", key: "embeddingLogger" },
  ];

  checkboxBindings.forEach(({ id, key }) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.checked = Boolean(store.config[key]);
    el.addEventListener("change", (event) => {
      store.config[key] = event.target.checked;
      persistStore();
    });
  });

  const jsonBindings = [
    { id: "train-kwargs", key: "trainKwargs", errorKey: "trainKwargsError" },
    { id: "val-kwargs", key: "valKwargs", errorKey: "valKwargsError" },
    { id: "backbone-kwargs", key: "backboneKwargs", errorKey: "backboneKwargsError" },
    { id: "wandb-config", key: "wandbConfig", errorKey: "wandbConfigError" },
    { id: "wandb-tags", key: "wandbTags", errorKey: "wandbTagsError" },
  ];

  jsonBindings.forEach(({ id, key, errorKey }) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = store.config[key] ?? "";
    el.addEventListener("input", (event) => {
      const value = event.target.value;
      store.config[key] = value;
      validateJsonField(value, errorKey, key === "wandbTags");
      persistStore();
    });
    validateJsonField(store.config[key], errorKey, key === "wandbTags");
  });

  const modalitySelect = document.getElementById("modality");
  const methodSelect = document.getElementById("method");
  if (modalitySelect) {
    modalitySelect.value = store.config.modality;
    modalitySelect.addEventListener("change", (event) => {
      const modality = event.target.value;
      store.config.modality = modality;
      populateMethodOptions(modality);
      persistStore();
      refreshMetrics();
    });
  }
  if (methodSelect) {
    methodSelect.addEventListener("change", (event) => {
      store.config.method = event.target.value;
      persistStore();
      refreshMetrics();
    });
  }

  const backboneMode = document.getElementById("backbone-mode");
  if (backboneMode) {
    backboneMode.value = store.config.backboneMode;
    backboneMode.addEventListener("change", (event) => {
      store.config.backboneMode = event.target.value;
      persistStore();
    });
  }
}

function validateJsonField(value, errorKey, expectArray = false) {
  const trimmed = (value || "").trim();
  if (!trimmed) {
    store.validation[errorKey] = "";
    renderValidationErrors();
    return;
  }
  try {
    const parsed = JSON.parse(trimmed);
    if (expectArray && !Array.isArray(parsed)) {
      throw new Error("Expected a JSON array");
    }
    store.validation[errorKey] = "";
  } catch (error) {
    store.validation[errorKey] = error.message;
  }
  renderValidationErrors();
}

function renderValidationErrors() {
  const map = {
    trainKwargsError: "train-kwargs-error",
    valKwargsError: "val-kwargs-error",
    backboneKwargsError: "backbone-kwargs-error",
    wandbConfigError: "wandb-config-error",
    wandbTagsError: "wandb-tags-error",
  };
  Object.entries(map).forEach(([key, id]) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = store.validation[key] || "";
    el.hidden = !store.validation[key];
  });
}

function populateMethodOptions(modality) {
  const methodSelect = document.getElementById("method");
  if (!methodSelect) return;
  const options = METHOD_OPTIONS[modality] || [];
  methodSelect.innerHTML = options
    .map((method) => `<option value="${method}">${method}</option>`)
    .join("");
  if (!options.includes(store.config.method)) {
    store.config.method = options[0] || "";
  }
  methodSelect.value = store.config.method;
}

function refreshMetrics() {
  const { method, modality, batchSize, epochs } = store.config;
  const metricMap = {
    "metric-method": method,
    "metric-modality": modality,
    "metric-batch": batchSize,
    "metric-epochs": epochs,
  };
  Object.entries(metricMap).forEach(([id, value]) => {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = value ?? "—";
    }
  });
}

function refreshPythonStatus(status, message) {
  const el = document.getElementById("python-status");
  if (!el) return;
  el.dataset.state = status;
  el.textContent = message;
}

function startTraining() {
  if (store.runtime.isRunning) {
    appendLog("Training is already running.");
    return;
  }
  const payload = buildTrainingPayload();
  if (!payload) {
    appendLog("Unable to start training: please resolve validation errors.");
    return;
  }
  store.runtime = { ...store.runtime, isRunning: true, currentEpoch: 0, trainLoss: null, valLoss: null, totalEpochs: payload.train.epochs };
  updateRunControls();
  appendLog("Dispatching configuration to Python runtime…");
  runtime.startTraining(payload);
}

function buildTrainingPayload() {
  const errors = {};
  const parseJson = (value, { expectArray = false } = {}) => {
    const trimmed = (value || "").trim();
    if (!trimmed) return null;
    const parsed = JSON.parse(trimmed);
    if (expectArray && !Array.isArray(parsed)) {
      throw new Error("Expected a JSON array");
    }
    return parsed;
  };
  const safeParse = (value, errorKey, options) => {
    try {
      const parsed = parseJson(value, options);
      store.validation[errorKey] = "";
      return parsed;
    } catch (error) {
      store.validation[errorKey] = error.message;
      errors[errorKey] = error.message;
      return null;
    }
  };

  const trainKwargs = safeParse(store.config.trainKwargs, "trainKwargsError");
  const valKwargs = safeParse(store.config.valKwargs, "valKwargsError");
  const backboneKwargs = safeParse(store.config.backboneKwargs, "backboneKwargsError");
  const wandbConfig = safeParse(store.config.wandbConfig, "wandbConfigError");
  const wandbTags = safeParse(store.config.wandbTags, "wandbTagsError", { expectArray: true });

  renderValidationErrors();
  if (Object.keys(errors).length > 0) {
    return null;
  }

  const trainSource = editors.train?.getValue?.() ?? "";
  const valSource = editors.val?.getValue?.() ?? "";
  const backboneSource = editors.backbone?.getValue?.() ?? "";

  const encodeSource = (value) => {
    if (!value) return "";
    return btoa(unescape(encodeURIComponent(value)));
  };

  return {
    global: {
      runName: store.config.runName,
      saveDir: store.config.saveDir,
      checkpointInterval: Number(store.config.checkpointInterval) || 0,
      reloadCheckpoint: store.config.reloadCheckpoint,
      mixedPrecision: store.config.mixedPrecision,
      verbose: store.config.verbose,
      useDataParallel: store.config.useDataParallel,
      numWorkers: Number(store.config.numWorkers) || 0,
    },
    selection: {
      modality: store.config.modality,
      method: store.config.method,
      variant: store.config.variant,
    },
    dataset: {
      train: { className: store.config.trainClass, kwargs: trainKwargs },
      val: { className: store.config.valClass, kwargs: valKwargs },
    },
    backbone: {
      mode: store.config.backboneMode,
      className: store.config.backboneClass,
      kwargs: backboneKwargs,
    },
    train: {
      batchSize: Number(store.config.batchSize) || 0,
      epochs: Number(store.config.epochs) || 1,
      learningRate: Number(store.config.lr) || 0,
      weightDecay: Number(store.config.weightDecay) || 0,
      optimizer: store.config.optimizer,
      useHpo: store.config.useHpo,
      nTrials: Number(store.config.nTrials) || 0,
      tuningEpochs: Number(store.config.tuningEpochs) || 0,
      embeddingLogger: store.config.embeddingLogger,
    },
    wandb: {
      project: store.config.wandbProject,
      entity: store.config.wandbEntity,
      mode: store.config.wandbMode,
      runName: store.config.wandbRunName,
      notes: store.config.wandbNotes,
      config: wandbConfig,
      tags: wandbTags,
    },
    python: {
      train_source: encodeSource(trainSource),
      val_source: encodeSource(valSource),
      backbone_source: encodeSource(backboneSource),
    },
  };
}

function updateRunControls() {
  const running = store.runtime.isRunning;
  const startButton = $("#btn-start");
  const stopButton = $("#btn-stop");
  if (startButton) {
    startButton.disabled = running;
  }
  if (stopButton) {
    stopButton.disabled = !running;
  }
}

function handleRuntimeEvent(event) {
  switch (event.type) {
    case "ready":
      refreshPythonStatus("ready", "Python runtime ready");
      appendLog("Pyodide initialized successfully.");
      break;
    case "runtime-state": {
      const status = event.payload?.status;
      if (status === "running") {
        refreshPythonStatus("running", "Python executing workload…");
      } else if (status === "idle") {
        refreshPythonStatus("idle", "Python idle");
        store.runtime.isRunning = false;
        updateRunControls();
      }
      break;
    }
    case "python-event":
      processPythonEvent(event.payload);
      break;
    case "python-error":
      appendLog(`Python error: ${event.error}`);
      refreshPythonStatus("error", `Python error: ${event.error}`);
      store.runtime.isRunning = false;
      updateRunControls();
      break;
    case "train-complete":
      appendLog(`Training finished: ${JSON.stringify(event.payload)}`);
      store.runtime.isRunning = false;
      updateRunControls();
      break;
    case "diagnostic-result":
      renderDiagnosticOutput(event.payload);
      break;
    case "diagnostic-error":
      renderDiagnosticOutput({ status: "error", output: event.error });
      break;
    default:
      break;
  }
}

function processPythonEvent(payload) {
  if (!payload) return;
  switch (payload.event) {
    case "status":
      if (payload.message) {
        appendLog(payload.message);
      }
      if (typeof payload.epoch === "number") {
        store.runtime.currentEpoch = payload.epoch;
        updateProgress();
      }
      break;
    case "metrics":
      store.runtime.currentEpoch = payload.epoch || store.runtime.currentEpoch;
      store.runtime.trainLoss = payload.train_loss ?? store.runtime.trainLoss;
      store.runtime.valLoss = payload.val_loss ?? store.runtime.valLoss;
      store.runtime.totalEpochs = payload.total_epochs ?? store.runtime.totalEpochs;
      updateMetricsDisplay();
      updateProgress();
      break;
    case "log":
      if (payload.message) {
        appendLog(payload.message);
      }
      break;
    default:
      appendLog(`Python event: ${JSON.stringify(payload)}`);
      break;
  }
}

function updateMetricsDisplay() {
  const { currentEpoch, trainLoss, valLoss } = store.runtime;
  const epochEl = document.getElementById("monitor-epoch");
  const trainEl = document.getElementById("monitor-train-loss");
  const valEl = document.getElementById("monitor-val-loss");
  if (epochEl) epochEl.textContent = currentEpoch ? `Epoch ${currentEpoch}` : "—";
  if (trainEl) trainEl.textContent = typeof trainLoss === "number" ? trainLoss.toFixed(4) : "—";
  if (valEl) valEl.textContent = typeof valLoss === "number" ? valLoss.toFixed(4) : "—";
}

function updateProgress() {
  const { currentEpoch, totalEpochs } = store.runtime;
  const fill = document.getElementById("progress-fill");
  if (!fill) return;
  const progress = totalEpochs ? Math.min(100, Math.round((currentEpoch / totalEpochs) * 100)) : 0;
  fill.style.width = `${progress}%`;
  fill.dataset.value = `${progress}%`;
}

function appendLog(message) {
  const clean = stripAnsi(message || "");
  store.logs.push(`[${new Date().toLocaleTimeString()}] ${clean}`);
  if (store.logs.length > 500) {
    store.logs = store.logs.slice(-500);
  }
  renderLogs();
}

function renderLogs() {
  const container = document.getElementById("log-output");
  if (!container) return;
  container.textContent = store.logs.join("\n");
  container.scrollTop = container.scrollHeight;
}

async function runDiagnosticSnippet() {
  const textarea = document.getElementById("diagnostic-snippet");
  if (!textarea) return;
  const code = textarea.value || "print('hello from python')";
  renderDiagnosticOutput({ status: "running", output: "Running diagnostic…" });
  try {
    const result = await runtime.runDiagnostic(code);
    renderDiagnosticOutput(result);
  } catch (error) {
    renderDiagnosticOutput({ status: "error", output: error.message || String(error) });
  }
}

function renderDiagnosticOutput(result) {
  const outputEl = document.getElementById("diagnostic-output");
  if (!outputEl) return;
  const status = result?.status || "unknown";
  outputEl.dataset.status = status;
  outputEl.textContent = result?.output || JSON.stringify(result, null, 2);
}

