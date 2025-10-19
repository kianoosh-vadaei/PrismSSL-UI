import { ApiClient } from "./api.js";
import { stripAnsi } from "./ansi.js";
import { createAceEditor, updateEditorTheme } from "./editors.js";

const METHOD_OPTIONS = {
  audio: ["Wav2Vec2", "HuBERT", "SimCLR", "COLA", "EAT"],
  vision: ["SimCLR", "BYOL", "MoCoV2", "DINO", "BarlowTwins", "SwAV", "SimSiam"],
  graph: ["GraphCL"],
  "cross-modal": ["CLIP", "SLIP", "ALBEF", "SIMVLM", "UNITER_VQA", "VSE", "CLAP", "AUDIO_CLIP", "WAV2CLIP"],
};

const STORAGE_KEY = "mkssl-config-v1";
const THEME_KEY = "mkssl-theme";
const MOCK_KEY = "mkssl-use-mock";

const DEFAULT_CONFIG = () => ({
  runName: `run_${new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 15)}`,
  saveDir: "checkpoints",
  checkpointInterval: 10,
  reloadCheckpoint: false,
  mixedPrecision: true,
  verbose: true,
  useDataParallel: false,
  numWorkers: 8,
  wandbProject: "",
  wandbEntity: "",
  wandbMode: "online",
  wandbRunName: "",
  wandbNotes: "",
  wandbTags: "",
  wandbConfig: "",
  modality: "audio",
  method: "Wav2Vec2",
  variant: "",
  trainClass: "DummyDataset",
  trainKwargs: "{\n  \"length\": 500,\n  \"input_dim\": 64\n}",
  valClass: "",
  valKwargs: "",
  backboneMode: "none",
  backboneClass: "MyBackbone",
  backboneKwargs: "{\n  \"hidden_dim\": 256\n}",
  batchSize: 16,
  epochs: 100,
  lr: 0.001,
  weightDecay: 0.01,
  optimizer: "adam",
  useHpo: false,
  nTrials: 20,
  tuningEpochs: 5,
  embeddingLogger: false,
});

const DEFAULT_TRAIN_CODE = `import torch\nfrom torch.utils.data import Dataset\n\nclass DummyDataset(Dataset):\n    def __init__(self, length=500, input_dim=64):\n        self.length = length\n        self.input_dim = input_dim\n\n    def __len__(self):\n        return self.length\n\n    def __getitem__(self, idx):\n        return torch.randn(self.input_dim), torch.randint(0, 10, (1,))\n`;

const DEFAULT_VAL_CODE = `# Optional validation dataset placeholder\nimport torch\nfrom torch.utils.data import Dataset\n\nclass DummyValDataset(Dataset):\n    def __len__(self):\n        return 200\n\n    def __getitem__(self, idx):\n        return torch.randn(64), torch.randint(0, 10, (1,))\n`;

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
  },
};

const JSON_FIELD_LABELS = {
  trainKwargsError: "Train dataset kwargs",
  valKwargsError: "Validation dataset kwargs",
  backboneKwargsError: "Backbone kwargs",
  wandbConfigError: "W&B config",
  wandbTagsError: "W&B tags",
};

const api = new ApiClient();
let editors = { train: null, val: null, backbone: null };
let statusInterval = null;
let logsInterval = null;
const persistStore = debounce(() => {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(store.config));
  } catch (error) {
    console.warn("Unable to persist configuration", error);
  }
}, 200);

function loadStoredConfig() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return DEFAULT_CONFIG();
    }
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_CONFIG(), ...parsed };
  } catch (error) {
    console.warn("Failed to load config", error);
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

function parseJsonOrNull(value) {
  if (!value || !value.trim()) return null;
  try {
    return JSON.parse(value);
  } catch (error) {
    console.warn("Failed to parse JSON value", error);
    return null;
  }
}

function $(selector) {
  return document.querySelector(selector);
}

function initTheme() {
  const storedTheme = window.localStorage.getItem(THEME_KEY) || "dark";
  applyTheme(storedTheme);
  const toggle = $("#theme-toggle");
  toggle?.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
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

function initMockToggle() {
  const checkbox = $("#mock-api");
  if (!checkbox) return;
  const stored = window.localStorage.getItem(MOCK_KEY);
  const enabled = stored === null ? true : stored === "true";
  checkbox.checked = enabled;
  api.setMockMode(enabled);
  checkbox.addEventListener("change", () => {
    const value = checkbox.checked;
    api.setMockMode(value);
    window.localStorage.setItem(MOCK_KEY, String(value));
    setStatusMessage(value ? "Mock backend enabled." : "Using live API endpoints.", "info");
  });
}

function initEditors() {
  editors.train = createAceEditor({
    elementId: "train-editor",
    storageKey: "mkssl-editor-train",
    defaultValue: DEFAULT_TRAIN_CODE,
  });
  editors.val = createAceEditor({
    elementId: "val-editor",
    storageKey: "mkssl-editor-val",
    defaultValue: DEFAULT_VAL_CODE,
  });
  editors.backbone = createAceEditor({
    elementId: "backbone-editor",
    storageKey: "mkssl-editor-backbone",
    defaultValue: DEFAULT_BACKBONE_CODE,
  });
}

function bindInputs() {
  bindTextInput("#run-name", "runName");
  bindTextInput("#save-dir", "saveDir");
  bindNumberInput("#checkpoint-interval", "checkpointInterval");
  bindCheckbox("#reload-ckpt", "reloadCheckpoint");
  bindCheckbox("#mixed-precision", "mixedPrecision");
  bindCheckbox("#verbose", "verbose");
  bindCheckbox("#use-dp", "useDataParallel");
  bindNumberInput("#num-workers", "numWorkers");
  bindTextInput("#wandb-project", "wandbProject");
  bindTextInput("#wandb-entity", "wandbEntity");
  bindSelect("#wandb-mode", "wandbMode");
  bindTextInput("#wandb-run-name", "wandbRunName");
  bindTextInput("#wandb-notes", "wandbNotes");
  bindJsonTextarea("#wandb-tags", "wandbTags", "wandbTagsError", { optional: true });
  bindJsonTextarea("#wandb-config", "wandbConfig", "wandbConfigError", { optional: true });
  bindSelect("#modality", "modality", () => {
    updateMethodOptions();
    updateMetricsSummary();
  });
  bindSelect("#method", "method", updateMetricsSummary);
  bindTextInput("#variant", "variant");
  bindTextInput("#train-class", "trainClass");
  bindJsonTextarea("#train-kwargs", "trainKwargs", "trainKwargsError");
  bindTextInput("#val-class", "valClass");
  bindJsonTextarea("#val-kwargs", "valKwargs", "valKwargsError", { optional: true });
  bindRadioGroup("input[name='backbone-mode']", "backboneMode", handleBackboneModeChange);
  bindTextInput("#backbone-class", "backboneClass");
  bindJsonTextarea("#backbone-kwargs", "backboneKwargs", "backboneKwargsError", { optional: true });
  bindNumberInput("#batch-size", "batchSize");
  bindNumberInput("#epochs", "epochs");
  bindNumberInput("#lr", "lr", { allowFloat: true });
  bindNumberInput("#weight-decay", "weightDecay", { allowFloat: true });
  bindSelect("#optimizer", "optimizer");
  bindCheckbox("#use-hpo", "useHpo", handleHpoToggle);
  bindNumberInput("#n-trials", "nTrials");
  bindNumberInput("#tuning-epochs", "tuningEpochs");
  bindCheckbox("#embedding-logger", "embeddingLogger");
  setInitialValues();
  handleBackboneModeChange();
  handleHpoToggle();
  updateMethodOptions();
  updateMetricsSummary();
}

function setInitialValues() {
  const entries = [
    ["#run-name", store.config.runName],
    ["#save-dir", store.config.saveDir],
    ["#checkpoint-interval", store.config.checkpointInterval],
    ["#reload-ckpt", store.config.reloadCheckpoint],
    ["#mixed-precision", store.config.mixedPrecision],
    ["#verbose", store.config.verbose],
    ["#use-dp", store.config.useDataParallel],
    ["#num-workers", store.config.numWorkers],
    ["#wandb-project", store.config.wandbProject],
    ["#wandb-entity", store.config.wandbEntity],
    ["#wandb-mode", store.config.wandbMode],
    ["#wandb-run-name", store.config.wandbRunName],
    ["#wandb-notes", store.config.wandbNotes],
    ["#wandb-tags", store.config.wandbTags],
    ["#wandb-config", store.config.wandbConfig],
    ["#modality", store.config.modality],
    ["#method", store.config.method],
    ["#variant", store.config.variant],
    ["#train-class", store.config.trainClass],
    ["#train-kwargs", store.config.trainKwargs],
    ["#val-class", store.config.valClass],
    ["#val-kwargs", store.config.valKwargs],
    ["input[name='backbone-mode']", store.config.backboneMode],
    ["#backbone-class", store.config.backboneClass],
    ["#backbone-kwargs", store.config.backboneKwargs],
    ["#batch-size", store.config.batchSize],
    ["#epochs", store.config.epochs],
    ["#lr", store.config.lr],
    ["#weight-decay", store.config.weightDecay],
    ["#optimizer", store.config.optimizer],
    ["#use-hpo", store.config.useHpo],
    ["#n-trials", store.config.nTrials],
    ["#tuning-epochs", store.config.tuningEpochs],
    ["#embedding-logger", store.config.embeddingLogger],
  ];
  for (const [selector, value] of entries) {
    const el = $(selector);
    if (!el) continue;
    if (el.type === "checkbox") {
      el.checked = Boolean(value);
    } else if (el.type === "radio") {
      const radio = document.querySelector(`input[name='backbone-mode'][value='${value}']`);
      if (radio) radio.checked = true;
    } else {
      el.value = value ?? "";
    }
  }
}

function bindTextInput(selector, key) {
  const el = $(selector);
  if (!el) return;
  el.value = store.config[key] ?? "";
  el.addEventListener("input", () => {
    store.config[key] = el.value;
    persistStore();
    if (["runName", "saveDir", "variant"].includes(key)) {
      updateMetricsSummary();
    }
  });
}

function bindNumberInput(selector, key, { allowFloat = false } = {}) {
  const el = $(selector);
  if (!el) return;
  el.value = store.config[key];
  el.addEventListener("input", () => {
    const raw = el.value;
    const parsed = allowFloat ? parseFloat(raw) : parseInt(raw, 10);
    if (!Number.isNaN(parsed)) {
      store.config[key] = parsed;
      persistStore();
      updateMetricsSummary();
    }
  });
}

function bindCheckbox(selector, key, onChange) {
  const el = $(selector);
  if (!el) return;
  el.checked = Boolean(store.config[key]);
  el.addEventListener("change", () => {
    store.config[key] = el.checked;
    persistStore();
    if (onChange) onChange();
  });
}

function bindSelect(selector, key, onChange) {
  const el = $(selector);
  if (!el) return;
  el.value = store.config[key];
  el.addEventListener("change", () => {
    store.config[key] = el.value;
    persistStore();
    if (onChange) onChange();
  });
}

function bindRadioGroup(selector, key, onChange) {
  const radios = document.querySelectorAll(selector);
  radios.forEach((radio) => {
    radio.checked = radio.value === store.config[key];
    radio.addEventListener("change", () => {
      if (radio.checked) {
        store.config[key] = radio.value;
        persistStore();
        if (onChange) onChange(radio.value);
      }
    });
  });
}

function bindJsonTextarea(selector, key, errorKey, { optional = false } = {}) {
  const textarea = $(selector);
  if (textarea) {
    textarea.value = store.config[key] ?? "";
    textarea.addEventListener("input", () => {
      store.config[key] = textarea.value;
      validateJsonField(key, errorKey, optional);
      persistStore();
    });
  }
  validateJsonField(key, errorKey, optional);
}

function validateJsonField(key, errorKey, optional) {
  const textarea = document.querySelector(getSelectorForKey(key));
  const errorEl = document.querySelector(`#${errorKey.replace(/([A-Z])/g, "-$1").toLowerCase()}`) || document.querySelector(`#${errorKey}`);
  if (!textarea || !errorEl) return;
  const value = textarea.value.trim();
  if (!value) {
    store.validation[errorKey] = "";
    errorEl.textContent = "";
    updateValidationSummary();
    return;
  }
  try {
    JSON.parse(value);
    store.validation[errorKey] = "";
    errorEl.textContent = "";
  } catch (error) {
    const label = JSON_FIELD_LABELS[errorKey] || "JSON field";
    const message = `Invalid JSON in ${label}.`;
    store.validation[errorKey] = message;
    errorEl.textContent = message;
  }
  updateValidationSummary();
}

function getSelectorForKey(key) {
  switch (key) {
    case "trainKwargs":
      return "#train-kwargs";
    case "valKwargs":
      return "#val-kwargs";
    case "backboneKwargs":
      return "#backbone-kwargs";
    case "wandbConfig":
      return "#wandb-config";
    case "wandbTags":
      return "#wandb-tags";
    default:
      return `#${key}`;
  }
}

function handleBackboneModeChange() {
  const mode = store.config.backboneMode;
  const section = $("#backbone-config");
  if (!section) return;
  if (mode === "upload") {
    section.hidden = false;
    section.setAttribute("aria-hidden", "false");
  } else {
    section.hidden = true;
    section.setAttribute("aria-hidden", "true");
  }
}

function handleHpoToggle() {
  const enabled = Boolean(store.config.useHpo);
  const trials = $("#n-trials");
  const tuning = $("#tuning-epochs");
  if (trials) {
    trials.disabled = !enabled;
  }
  if (tuning) {
    tuning.disabled = !enabled;
  }
}

function updateMethodOptions() {
  const select = $("#method");
  if (!select) return;
  const options = METHOD_OPTIONS[store.config.modality] || [];
  select.innerHTML = "";
  const targetMethod = options.includes(store.config.method) ? store.config.method : options[0];
  options.forEach((method) => {
    const option = document.createElement("option");
    option.value = method;
    option.textContent = method;
    option.selected = method === targetMethod;
    select.append(option);
  });
  store.config.method = targetMethod;
  persistStore();
  updateMetricsSummary();
}

function updateMetricsSummary() {
  const modalityEl = $("#metric-modality");
  const methodEl = $("#metric-method");
  const batchEl = $("#metric-batch");
  const epochsEl = $("#metric-epochs");
  if (modalityEl) modalityEl.textContent = store.config.modality;
  if (methodEl) methodEl.textContent = store.config.method;
  if (batchEl) batchEl.textContent = store.config.batchSize;
  if (epochsEl) epochsEl.textContent = store.config.epochs;
}

function wireControls() {
  $("#init-trainer")?.addEventListener("click", onInitializeTrainer);
  $("#start-training")?.addEventListener("click", onStartTraining);
  $("#stop-training")?.addEventListener("click", onStopTraining);
  $("#clear-logs")?.addEventListener("click", onClearLogs);
  $("#validate-backbone")?.addEventListener("click", onValidateBackbone);
}

async function onValidateBackbone() {
  if (store.config.backboneMode !== "upload") {
    setStatusMessage("Backbone mode is set to default.", "info");
    return;
  }
  const { backboneClass, backboneKwargs } = store.config;
  const source = editors.backbone ? editors.backbone.getValue() : "";
  try {
    const kwargs = backboneKwargs.trim() ? JSON.parse(backboneKwargs) : {};
    await api.buildBackbone({
      mode: "upload",
      class_name: backboneClass,
      kwargs,
      source,
    });
    setStatusMessage(`Backbone ${backboneClass} validated successfully.`, "success");
    $("#backbone-status").textContent = "Backbone ready ✅";
  } catch (error) {
    console.error(error);
    setStatusMessage(`Backbone error: ${error.message}`, "error");
    $("#backbone-status").textContent = `Error: ${error.message}`;
  }
}

function collectDatasets() {
  const trainSource = editors.train ? editors.train.getValue() : "";
  const valSource = editors.val ? editors.val.getValue() : "";
  if (!trainSource.trim()) {
    throw new Error("Provide train dataset code first.");
  }
  if (!store.config.trainClass.trim()) {
    throw new Error("Train dataset class is required.");
  }
  let trainKwargs = {};
  let valKwargs = {};
  if (store.config.trainKwargs.trim()) {
    trainKwargs = JSON.parse(store.config.trainKwargs);
  }
  if (store.config.valKwargs.trim()) {
    valKwargs = JSON.parse(store.config.valKwargs);
  }
  if (store.config.valClass?.trim() && !valSource.trim()) {
    throw new Error("Provide validation dataset code or clear the class name.");
  }
  return {
    train: {
      source: trainSource,
      class_name: store.config.trainClass,
      kwargs: trainKwargs,
    },
    val: store.config.valClass?.trim()
      ? {
          source: valSource,
          class_name: store.config.valClass,
          kwargs: valKwargs,
        }
      : null,
  };
}

function collectBackbone() {
  if (store.config.backboneMode !== "upload") {
    return { mode: "none" };
  }
  const source = editors.backbone ? editors.backbone.getValue() : "";
  if (!source.trim()) {
    throw new Error("Provide backbone source code.");
  }
  if (!store.config.backboneClass.trim()) {
    throw new Error("Backbone class is required.");
  }
  const kwargs = store.config.backboneKwargs.trim()
    ? JSON.parse(store.config.backboneKwargs)
    : {};
  return {
    mode: "upload",
    source,
    class_name: store.config.backboneClass,
    kwargs,
  };
}

function collectTrainConfig() {
  return {
    batch_size: Number(store.config.batchSize),
    epochs: Number(store.config.epochs),
    lr: Number(store.config.lr),
    weight_decay: Number(store.config.weightDecay),
    optimizer: store.config.optimizer,
    use_hpo: Boolean(store.config.useHpo),
    n_trials: Number(store.config.nTrials),
    tuning_epochs: Number(store.config.tuningEpochs),
    use_embedding_logger: Boolean(store.config.embeddingLogger),
  };
}

function assemblePayload() {
  const datasets = collectDatasets();
  const backbone = collectBackbone();
  return {
    run: {
      name: store.config.runName,
      save_dir: store.config.saveDir,
      checkpoint_interval: Number(store.config.checkpointInterval),
      reload_checkpoint: Boolean(store.config.reloadCheckpoint),
      mixed_precision: Boolean(store.config.mixedPrecision),
      verbose: Boolean(store.config.verbose),
      use_data_parallel: Boolean(store.config.useDataParallel),
      num_workers: Number(store.config.numWorkers),
      wandb_project: store.config.wandbProject?.trim() || null,
      wandb_entity: store.config.wandbEntity?.trim() || null,
      wandb_mode: store.config.wandbMode || "online",
      wandb_run_name: store.config.wandbRunName?.trim() || null,
      wandb_notes: store.config.wandbNotes?.trim() || null,
      wandb_tags: parseJsonOrNull(store.config.wandbTags),
      wandb_config: parseJsonOrNull(store.config.wandbConfig),
    },
    modality: store.config.modality,
    method: store.config.method,
    variant: store.config.variant || null,
    datasets,
    backbone,
    train: collectTrainConfig(),
  };
}

async function onInitializeTrainer() {
  try {
    const payload = assemblePayload();
    await api.initTrainer(payload);
    setStatusMessage("Trainer is ready ✅", "success");
  } catch (error) {
    console.error(error);
    setStatusMessage(`Trainer init failed: ${error.message}`, "error");
  }
}

async function onStartTraining() {
  try {
    const payload = assemblePayload();
    const response = await api.startTraining(payload);
    if (response?.status === "already_running") {
      setStatusMessage("Training already running.", "info");
      return;
    }
    setStatusMessage("Training started.", "success");
    store.runtime.isRunning = true;
    ensureStatusPolling();
    ensureLogPolling();
    updateStatusBanner();
  } catch (error) {
    console.error(error);
    setStatusMessage(error.message, "error");
  }
}

async function onStopTraining() {
  try {
    const response = await api.stopTraining();
    if (response?.status === "stopped") {
      setStatusMessage("Stop requested.", "warn");
    } else {
      setStatusMessage("Trainer already idle.", "info");
    }
    store.runtime.isRunning = false;
    stopStatusPolling();
    updateStatusBanner();
  } catch (error) {
    console.error(error);
    setStatusMessage(error.message, "error");
  }
}

async function onClearLogs() {
  try {
    await api.clearLogs();
    $("#logs").value = "";
    $("#log-updated").textContent = "Last updated: just now";
    setStatusMessage("Logs cleared.", "info");
  } catch (error) {
    console.error(error);
    setStatusMessage(error.message, "error");
  }
}

function setStatusMessage(message, type = "info") {
  const el = $("#status-message");
  if (!el) return;
  el.textContent = message;
  el.classList.remove("status-message--success", "status-message--warn");
  if (type === "success") {
    el.classList.add("status-message--success");
  } else if (type === "warn" || type === "error") {
    el.classList.add("status-message--warn");
  }
}

function updateStatusBanner() {
  const running = store.runtime.isRunning;
  setStatusMessage(running ? "🟢 Training running in background..." : "🟡 Training idle.");
}

function ensureStatusPolling() {
  if (statusInterval) return;
  statusInterval = setInterval(fetchStatus, 1500);
  fetchStatus();
}

function stopStatusPolling() {
  if (statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
}

function ensureLogPolling() {
  if (logsInterval) return;
  logsInterval = setInterval(fetchLogs, 2000);
  fetchLogs();
}

function stopLogPolling() {
  if (logsInterval) {
    clearInterval(logsInterval);
    logsInterval = null;
  }
}

async function fetchStatus() {
  try {
    const data = await api.getStatus();
    if (!data) return;
    store.runtime.isRunning = Boolean(data.is_running);
    store.runtime.currentEpoch = data.current_epoch ?? 0;
    store.runtime.trainLoss = data.train_loss;
    store.runtime.valLoss = data.val_loss;
    renderLiveMetrics();
    if (!store.runtime.isRunning) {
      stopStatusPolling();
      updateStatusBanner();
    } else {
      setStatusMessage("🟢 Training running in background...");
    }
  } catch (error) {
    console.error(error);
  }
}

async function fetchLogs() {
  try {
    const data = await api.getLogs();
    if (!data) return;
    const cleaned = stripAnsi(data.logs || "");
    const textarea = $("#logs");
    if (textarea && textarea.value !== cleaned) {
      textarea.value = cleaned;
      textarea.scrollTop = textarea.scrollHeight;
      $("#log-updated").textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
    }
  } catch (error) {
    console.error(error);
  }
}

function renderLiveMetrics() {
  $("#live-epoch").textContent = store.runtime.currentEpoch;
  $("#live-train-loss").textContent = store.runtime.trainLoss != null ? Number(store.runtime.trainLoss).toFixed(4) : "-";
  $("#live-val-loss").textContent = store.runtime.valLoss != null ? Number(store.runtime.valLoss).toFixed(4) : "-";
}

function updateValidationSummary() {
  const summary = Object.values(store.validation).find(Boolean);
  const el = $("#log-status");
  if (!el) return;
  if (summary) {
    el.textContent = `⚠️ ${summary}`;
  } else {
    el.textContent = "";
  }
}

function initializeApp() {
  initTheme();
  initMockToggle();
  initEditors();
  bindInputs();
  wireControls();
  updateStatusBanner();
  ensureLogPolling();
  fetchStatus();
}

window.addEventListener("beforeunload", () => {
  stopStatusPolling();
  stopLogPolling();
});

document.addEventListener("DOMContentLoaded", initializeApp);
