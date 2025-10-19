const DEFAULT_BASE_URL = "";

function delay(ms = 250) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class MockBackend {
  constructor() {
    this.reset();
  }

  reset() {
    this.isInitialized = false;
    this.backboneValidated = false;
    this.isRunning = false;
    this.currentEpoch = 0;
    this.maxEpochs = 0;
    this.trainLoss = null;
    this.valLoss = null;
    this.optimizer = "adam";
    this.logs = [];
    this.interval = null;
  }

  appendLog(line) {
    const timestamp = new Date().toISOString();
    this.logs.push(`[mock ${timestamp}] ${line}`);
    if (this.logs.length > 1000) {
      this.logs = this.logs.slice(-1000);
    }
  }

  async initTrainer(payload) {
    await delay(200);
    this.isInitialized = true;
    this.maxEpochs = payload?.train?.epochs ?? 0;
    this.optimizer = payload?.train?.optimizer ?? "adam";
    this.appendLog("Trainer initialized with mock backend.");
    return { status: "ok" };
  }

  async buildBackbone(payload) {
    await delay(200);
    if (!payload?.class_name) {
      throw new Error("Backbone class name required");
    }
    const classPattern = new RegExp(`class\\s+${payload.class_name}`);
    if (payload?.source && !classPattern.test(payload.source)) {
      throw new Error(`Class ${payload.class_name} not found in source`);
    }
    this.backboneValidated = true;
    this.appendLog(`Backbone ${payload.class_name} validated.`);
    return { status: "ok" };
  }

  async startTraining(payload) {
    await delay(250);
    if (!this.isInitialized) {
      throw new Error("Initialize trainer first");
    }
    if (this.isRunning) {
      return { status: "already_running" };
    }
    this.isRunning = true;
    this.currentEpoch = 0;
    this.trainLoss = null;
    this.valLoss = null;
    this.maxEpochs = payload?.train?.epochs ?? this.maxEpochs ?? 10;
    this.appendLog(`Starting mock training for ${this.maxEpochs} epochs using ${this.optimizer}.`);
    if (this.interval) {
      clearInterval(this.interval);
    }
    this.interval = setInterval(() => {
      if (!this.isRunning) {
        return;
      }
      this.currentEpoch += 1;
      const progress = Math.min(this.currentEpoch / Math.max(this.maxEpochs, 1), 1);
      this.trainLoss = Number((1.0 - progress + Math.random() * 0.05).toFixed(4));
      this.valLoss = Number((1.1 - progress + Math.random() * 0.08).toFixed(4));
      this.appendLog(`Epoch ${this.currentEpoch}: train=${this.trainLoss} val=${this.valLoss}`);
      if (this.currentEpoch >= this.maxEpochs) {
        this.appendLog("Training completed successfully.");
        this.isRunning = false;
        clearInterval(this.interval);
        this.interval = null;
      }
    }, 1200);
    return { status: "ok" };
  }

  async stopTraining() {
    await delay(150);
    if (!this.isRunning) {
      return { status: "idle" };
    }
    this.isRunning = false;
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    this.appendLog("Training stop requested (mock).");
    return { status: "stopped" };
  }

  async fetchStatus() {
    await delay(150);
    return {
      is_running: this.isRunning,
      current_epoch: this.currentEpoch,
      train_loss: this.trainLoss,
      val_loss: this.valLoss,
    };
  }

  async fetchLogs() {
    await delay(150);
    return { logs: this.logs.join("\n") };
  }

  async clearLogs() {
    await delay(150);
    this.logs = [];
    return { status: "cleared" };
  }
}

export class ApiClient {
  constructor({ baseUrl = DEFAULT_BASE_URL } = {}) {
    this.baseUrl = baseUrl;
    this.mock = new MockBackend();
    this.useMock = false;
  }

  setMockMode(enabled) {
    this.useMock = Boolean(enabled);
  }

  async request(path, { method = "GET", body } = {}) {
    if (this.useMock) {
      return this.requestMock(path, { method, body });
    }
    const url = `${this.baseUrl}${path}`;
    const headers = { "Content-Type": "application/json" };
    const options = { method, headers };
    if (body !== undefined) {
      options.body = JSON.stringify(body);
    }
    const response = await fetch(url, options);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed: ${response.status}`);
    }
    if (response.status === 204) {
      return null;
    }
    const data = await response.json();
    return data;
  }

  async requestMock(path, { method, body }) {
    switch (`${method} ${path}`) {
      case "POST /api/trainer/init":
        return this.mock.initTrainer(body);
      case "POST /api/trainer/build_backbone":
        return this.mock.buildBackbone(body);
      case "POST /api/train/start":
        return this.mock.startTraining(body);
      case "POST /api/train/stop":
        return this.mock.stopTraining(body);
      case "GET /api/status":
        return this.mock.fetchStatus();
      case "GET /api/logs":
        return this.mock.fetchLogs();
      case "POST /api/logs/clear":
        return this.mock.clearLogs();
      default:
        throw new Error(`Unsupported mock endpoint: ${method} ${path}`);
    }
  }

  initTrainer(payload) {
    return this.request("/api/trainer/init", { method: "POST", body: payload });
  }

  buildBackbone(payload) {
    return this.request("/api/trainer/build_backbone", { method: "POST", body: payload });
  }

  startTraining(payload) {
    return this.request("/api/train/start", { method: "POST", body: payload });
  }

  stopTraining() {
    return this.request("/api/train/stop", { method: "POST" });
  }

  getStatus() {
    return this.request("/api/status", { method: "GET" });
  }

  getLogs() {
    return this.request("/api/logs", { method: "GET" });
  }

  clearLogs() {
    return this.request("/api/logs/clear", { method: "POST" });
  }
}
