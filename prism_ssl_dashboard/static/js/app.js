const state = {
    datasetSummary: {},
    lastConfig: null,
    socket: null,
    runStart: null,
    stopRequested: false,
};

function $(selector) {
    return document.querySelector(selector);
}

function getTheme() {
    return localStorage.getItem('prism-theme') || 'light';
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const toggle = $('#theme-toggle');
    toggle.textContent = theme === 'light' ? '🌞' : '🌙';
    localStorage.setItem('prism-theme', theme);
}

function toggleTheme() {
    const current = getTheme();
    applyTheme(current === 'light' ? 'dark' : 'light');
}

function showToast(message, type = 'info', timeout = 4000) {
    const container = $('#toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('hide');
        toast.addEventListener('transitionend', () => toast.remove(), { once: true });
        toast.style.opacity = '0';
    }, timeout);
}

function openModal(content, title = 'Notice') {
    $('#modal-title').textContent = title;
    $('#modal-body').innerHTML = content;
    $('#modal').classList.remove('hidden');
}

function closeModal() {
    $('#modal').classList.add('hidden');
}

function setupTabs() {
    document.querySelectorAll('.tab').forEach((tab) => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach((panel) => panel.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tab.dataset.target).classList.add('active');
        });
    });
}

function parseJson(value, fallback = null) {
    if (!value || value.trim() === '') return fallback;
    try {
        return JSON.parse(value);
    } catch (err) {
        throw new Error(err.message || 'Invalid JSON');
    }
}

async function detectDatasetClasses() {
    const code = $('#dataset-code').value;
    if (!code.trim()) {
        showToast('Paste dataset code before detection.', 'error');
        return;
    }
    try {
        const res = await fetch('/api/dataset/inspect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code }),
        });
        if (!res.ok) throw new Error('Failed to inspect dataset.');
        const data = await res.json();
        const select = $('#dataset-class');
        select.innerHTML = '<option value="" disabled selected>Select a class</option>';
        if (data.classes && data.classes.length) {
            data.classes.forEach((cls) => {
                const option = document.createElement('option');
                option.value = cls;
                option.textContent = cls;
                select.appendChild(option);
            });
            showToast(`Detected ${data.classes.length} dataset class(es).`, 'success');
        } else {
            showToast('No dataset subclasses detected.', 'error');
        }
    } catch (err) {
        console.error(err);
        showToast(err.message, 'error');
    }
}

function renderDatasetSummary(summary) {
    const container = $('#dataset-summary');
    container.innerHTML = '';
    Object.entries(summary).forEach(([key, info]) => {
        const card = document.createElement('div');
        card.className = 'card';
        const details = [`Length: ${info.length ?? 'unknown'}`];
        if (info.preview) {
            details.push(`Preview: ${info.preview}`);
        }
        card.innerHTML = `<strong>${key} dataset</strong><br>${details.join('<br>')}`;
        container.appendChild(card);
    });
}

async function instantiateDatasets() {
    const code = $('#dataset-code').value;
    const className = $('#dataset-class').value;
    if (!code.trim() || !className) {
        showToast('Provide code and select a dataset class before instantiation.', 'error');
        return;
    }
    let kwargsTrain, kwargsVal, kwargsTest;
    try {
        kwargsTrain = parseJson($('#dataset-kwargs-train').value, {});
        kwargsVal = parseJson($('#dataset-kwargs-val').value, null);
        kwargsTest = parseJson($('#dataset-kwargs-test').value, null);
    } catch (err) {
        showToast(`JSON error: ${err.message}`, 'error');
        return;
    }
    try {
        const payload = {
            code,
            class_name: className,
            kwargs_train: kwargsTrain,
            kwargs_val: kwargsVal,
            kwargs_test: kwargsTest,
        };
        const res = await fetch('/api/dataset/instantiate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to instantiate dataset');
        state.datasetSummary = data.summary || {};
        renderDatasetSummary(state.datasetSummary);
        persistConfig({
            dataset: {
                code,
                class_name: className,
                kwargs_train: kwargsTrain,
                kwargs_val: kwargsVal,
                kwargs_test: kwargsTest,
            },
        });
        showToast('Datasets instantiated successfully.', 'success');
    } catch (err) {
        console.error(err);
        showToast(err.message, 'error');
    }
}

function sanitizeString(value) {
    if (value === undefined || value === null) return null;
    const trimmed = value.trim();
    return trimmed === '' ? null : trimmed;
}

function collectTrainerConfig() {
    const modality = sanitizeString($('#modality').value);
    const method = sanitizeString($('#method').value);
    if (!modality || !method) {
        throw new Error('Modality and method are required.');
    }
    const trainerCtor = {
        method,
        variant: sanitizeString($('#variant').value) || 'base',
        save_dir: sanitizeString($('#save_dir').value) || '.',
        checkpoint_interval: Number($('#checkpoint_interval').value) || 10,
        reload_checkpoint: $('#reload_checkpoint').checked,
        verbose: $('#verbose').checked,
        mixed_precision_training: $('#mixed_precision_training').checked,
        use_data_parallel: $('#use_data_parallel').checked,
    };
    const numWorkers = $('#num_workers').value;
    if (numWorkers) trainerCtor.num_workers = Number(numWorkers);

    const wandbProject = sanitizeString($('#wandb_project').value);
    const wandbEntity = sanitizeString($('#wandb_entity').value);
    const wandbRunName = sanitizeString($('#wandb_run_name').value);
    const wandbNotes = sanitizeString($('#wandb_notes').value);
    const wandbTags = sanitizeString($('#wandb_tags').value);

    trainerCtor.wandb_mode = $('#wandb_mode').value || 'online';
    if (wandbProject) trainerCtor.wandb_project = wandbProject;
    if (wandbEntity) trainerCtor.wandb_entity = wandbEntity;
    if (wandbRunName) trainerCtor.wandb_run_name = wandbRunName;
    if (wandbNotes) trainerCtor.wandb_notes = wandbNotes;
    if (wandbTags) trainerCtor.wandb_tags = wandbTags.split(',').map((tag) => tag.trim()).filter(Boolean);

    const wandbConfigRaw = $('#wandb_config').value;
    if (wandbConfigRaw.trim()) {
        trainerCtor.wandb_config = parseJson(wandbConfigRaw);
    }

    return {
        modality,
        trainer_ctor: trainerCtor,
        use_generic: $('#use_generic').checked,
        generic: {
            model_name: $('#generic_model_name').value,
            epochs: Number($('#generic_epochs').value) || 10,
            use_lora: $('#generic_use_lora').checked,
        },
    };
}

function collectTrainArgs() {
    const epochs = Number($('#epochs').value);
    const batchSize = Number($('#batch_size').value);
    const learningRate = Number($('#learning_rate').value);
    if (epochs < 1) throw new Error('Epochs must be ≥ 1');
    if (batchSize < 1) throw new Error('Batch size must be ≥ 1');
    if (learningRate <= 0) throw new Error('Learning rate must be > 0');

    const args = {
        batch_size: batchSize,
        epochs,
        start_epoch: Number($('#start_epoch').value) || 0,
        start_iteration: Number($('#start_iteration').value) || 0,
        learning_rate: learningRate,
        weight_decay: Number($('#weight_decay').value) || 0,
        optimizer: sanitizeString($('#optimizer').value) || 'adamw',
        use_hpo: $('#use_hpo').checked,
        n_trials: Number($('#n_trials').value) || 20,
        tuning_epochs: Number($('#tuning_epochs').value) || 5,
        use_embedding_logger: $('#use_embedding_logger').checked,
    };

    if (args.use_hpo) {
        if (args.n_trials < 1) throw new Error('HPO trials must be ≥ 1');
        if (args.tuning_epochs < 1) throw new Error('Tuning epochs must be ≥ 1');
    }

    const raw = $('#train_kwargs').value;
    if (raw.trim()) {
        args.extra_kwargs = parseJson(raw, {});
    }

    return args;
}

function collectEvalArgs() {
    const numClasses = Number($('#eval_num_classes').value);
    if (!numClasses || numClasses < 1) throw new Error('Evaluation requires num_classes ≥ 1');
    const args = {
        num_classes: numClasses,
        batch_size: Number($('#eval_batch_size').value) || 64,
        lr: Number($('#eval_lr').value) || 0.001,
        epochs: Number($('#eval_epochs').value) || 10,
        freeze_backbone: $('#eval_freeze_backbone').checked,
    };
    const raw = $('#eval_kwargs').value;
    if (raw.trim()) {
        args.extra_kwargs = parseJson(raw, {});
    }
    return args;
}

function persistConfig(partial) {
    const updated = { ...(state.lastConfig || {}), ...partial };
    state.lastConfig = updated;
    localStorage.setItem('prism-dashboard-config', JSON.stringify(updated));
}

function restoreConfig() {
    const stored = localStorage.getItem('prism-dashboard-config');
    if (!stored) return;
    try {
        const config = JSON.parse(stored);
        state.lastConfig = config;
        const trainer = config.trainer || {};
        if (trainer.modality) $('#modality').value = trainer.modality;
        if (trainer.trainer_ctor) {
            const c = trainer.trainer_ctor;
            $('#method').value = c.method ?? '';
            $('#variant').value = c.variant ?? 'base';
            $('#save_dir').value = c.save_dir ?? './checkpoints';
            $('#checkpoint_interval').value = c.checkpoint_interval ?? 10;
            $('#reload_checkpoint').checked = !!c.reload_checkpoint;
            $('#verbose').checked = c.verbose !== false;
            $('#mixed_precision_training').checked = c.mixed_precision_training !== false;
            $('#use_data_parallel').checked = !!c.use_data_parallel;
            if (c.num_workers !== undefined) $('#num_workers').value = c.num_workers;
            $('#wandb_mode').value = c.wandb_mode ?? 'online';
            $('#wandb_project').value = c.wandb_project ?? '';
            $('#wandb_entity').value = c.wandb_entity ?? '';
            $('#wandb_run_name').value = c.wandb_run_name ?? '';
            $('#wandb_notes').value = c.wandb_notes ?? '';
            $('#wandb_tags').value = (c.wandb_tags || []).join(', ');
            $('#wandb_config').value = c.wandb_config ? JSON.stringify(c.wandb_config, null, 2) : '';
        }
        $('#use_generic').checked = trainer.use_generic ?? false;
        if (trainer.use_generic && trainer.generic) {
            $('#generic-settings').hidden = false;
            $('#generic_model_name').value = trainer.generic.model_name ?? 'bert-base-uncased';
            $('#generic_epochs').value = trainer.generic.epochs ?? 10;
            $('#generic_use_lora').checked = trainer.generic.use_lora ?? true;
        }
        const train = config.train || {};
        if (train.batch_size) $('#batch_size').value = train.batch_size;
        if (train.epochs) $('#epochs').value = train.epochs;
        if (train.learning_rate) $('#learning_rate').value = train.learning_rate;
        if (train.weight_decay !== undefined) $('#weight_decay').value = train.weight_decay;
        $('#optimizer').value = train.optimizer ?? 'adamw';
        $('#start_epoch').value = train.start_epoch ?? 0;
        $('#start_iteration').value = train.start_iteration ?? 0;
        $('#use_hpo').checked = !!train.use_hpo;
        $('#n_trials').value = train.n_trials ?? 20;
        $('#tuning_epochs').value = train.tuning_epochs ?? 5;
        $('#use_embedding_logger').checked = !!train.use_embedding_logger;
        $('#train_kwargs').value = train.extra_kwargs ? JSON.stringify(train.extra_kwargs, null, 2) : '';

        const evalArgs = config.eval || {};
        if (evalArgs.num_classes) $('#eval_num_classes').value = evalArgs.num_classes;
        if (evalArgs.batch_size) $('#eval_batch_size').value = evalArgs.batch_size;
        if (evalArgs.lr) $('#eval_lr').value = evalArgs.lr;
        if (evalArgs.epochs) $('#eval_epochs').value = evalArgs.epochs;
        $('#eval_freeze_backbone').checked = evalArgs.freeze_backbone ?? true;
        $('#eval_kwargs').value = evalArgs.extra_kwargs ? JSON.stringify(evalArgs.extra_kwargs, null, 2) : '';

        if (config.dataset) {
            $('#dataset-code').value = config.dataset.code || '';
            $('#dataset-kwargs-train').value = config.dataset.kwargs_train ? JSON.stringify(config.dataset.kwargs_train, null, 2) : '';
            $('#dataset-kwargs-val').value = config.dataset.kwargs_val ? JSON.stringify(config.dataset.kwargs_val, null, 2) : '';
            $('#dataset-kwargs-test').value = config.dataset.kwargs_test ? JSON.stringify(config.dataset.kwargs_test, null, 2) : '';
        }
    } catch (err) {
        console.warn('Unable to restore config', err);
    }
}

function gatherFullConfig() {
    return {
        trainer: collectTrainerConfig(),
        train: collectTrainArgs(),
        eval: collectEvalArgs(),
        dataset: {
            code: $('#dataset-code').value,
            class_name: $('#dataset-class').value,
            kwargs_train: parseJson($('#dataset-kwargs-train').value, {}),
            kwargs_val: parseJson($('#dataset-kwargs-val').value, null),
            kwargs_test: parseJson($('#dataset-kwargs-test').value, null),
        },
    };
}

function exportConfig() {
    try {
        const config = gatherFullConfig();
        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'prism-dashboard-config.json';
        link.click();
        URL.revokeObjectURL(link.href);
    } catch (err) {
        showToast(err.message, 'error');
    }
}

function importConfig(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
        try {
            const config = JSON.parse(reader.result);
            localStorage.setItem('prism-dashboard-config', JSON.stringify(config));
            restoreConfig();
            showToast('Configuration imported.', 'success');
        } catch (err) {
            showToast('Failed to import configuration.', 'error');
        }
    };
    reader.readAsText(file);
}

function setStatus(status, cls = '') {
    const chip = $('#run-status');
    chip.textContent = status;
    chip.className = `status-chip ${cls}`.trim();
}

function updateProgress(progress) {
    $('#progress-bar').style.width = `${Math.max(0, Math.min(progress, 1)) * 100}%`;
}

function appendLog(line) {
    const output = $('#log-output');
    const timestamp = new Date().toLocaleTimeString();
    output.textContent += `[${timestamp}] ${line}\n`;
    if ($('#auto-scroll').checked) {
        output.scrollTop = output.scrollHeight;
    }
}

function resetConsole() {
    updateProgress(0);
    $('#log-output').textContent = '';
    $('#elapsed').textContent = 'Elapsed: 00:00';
    $('#eta').textContent = 'ETA: --';
}

function secondsToClock(seconds) {
    const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
    const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
}

function handleTimer() {
    if (!state.runStart) return;
    const elapsed = (Date.now() - state.runStart) / 1000;
    $('#elapsed').textContent = `Elapsed: ${secondsToClock(elapsed)}`;
    requestAnimationFrame(handleTimer);
}

function setupSocket() {
    state.socket = io();
    state.socket.on('connect', () => {
        state.socket.emit('join', { room: 'train' });
    });
    state.socket.on('train_log', (payload) => {
        appendLog(payload.message);
    });
    state.socket.on('train_progress', (payload) => {
        if (typeof payload.progress === 'number') updateProgress(payload.progress);
        if (payload.eta_seconds !== undefined) {
            $('#eta').textContent = `ETA: ${payload.eta_seconds === null ? '--' : secondsToClock(payload.eta_seconds)}`;
        }
    });
    state.socket.on('train_status', (payload) => {
        if (payload.status) setStatus(payload.status.label, payload.status.className);
        if (payload.elapsed_seconds !== undefined) {
            $('#elapsed').textContent = `Elapsed: ${secondsToClock(payload.elapsed_seconds)}`;
        }
    });
    state.socket.on('train_done', (payload) => {
        setStatus('Done', 'done');
        $('#stop-train').disabled = true;
        $('#start-train').disabled = false;
        $('#resume-train').disabled = false;
        state.runStart = null;
        showToast(payload.message || 'Training complete!', 'success');
    });
    state.socket.on('train_error', (payload) => {
        setStatus('Error', 'error');
        $('#stop-train').disabled = true;
        $('#start-train').disabled = false;
        $('#resume-train').disabled = false;
        state.runStart = null;
        showToast(payload.error || 'Training failed', 'error');
    });
}

async function startTraining() {
    try {
        const trainer = collectTrainerConfig();
        const trainArgs = collectTrainArgs();
        let datasetConfig;
        try {
            datasetConfig = {
                code: $('#dataset-code').value,
                class_name: $('#dataset-class').value,
                kwargs_train: parseJson($('#dataset-kwargs-train').value, {}),
                kwargs_val: parseJson($('#dataset-kwargs-val').value, null),
                kwargs_test: parseJson($('#dataset-kwargs-test').value, null),
            };
        } catch (err) {
            showToast(`Dataset kwargs invalid: ${err.message}`, 'error');
            return;
        }
        persistConfig({
            trainer,
            train: trainArgs,
            dataset: datasetConfig,
            eval: state.lastConfig?.eval ?? {},
        });
        if (!state.datasetSummary.train) {
            showToast('Instantiate the training dataset first.', 'error');
            return;
        }
        const payload = {
            modality: trainer.modality,
            trainer_ctor: trainer.trainer_ctor,
            train_args: trainArgs,
            use_generic: trainer.use_generic,
            generic: trainer.generic,
        };
        const res = await fetch('/api/train/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to start training');
        resetConsole();
        setStatus('Training', 'training');
        $('#start-train').disabled = true;
        $('#stop-train').disabled = false;
        $('#resume-train').disabled = true;
        state.runStart = Date.now();
        requestAnimationFrame(handleTimer);
        showToast('Training started.', 'success');
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function stopTraining() {
    try {
        const res = await fetch('/api/train/stop', { method: 'POST' });
        if (!res.ok) throw new Error('Failed to send stop signal');
        showToast('Stop signal sent.', 'info');
        state.stopRequested = true;
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function resumeTraining() {
    if (!state.lastConfig) {
        showToast('No configuration available to resume.', 'error');
        return;
    }
    try {
        const res = await fetch('/api/train/status');
        const status = await res.json();
        if (status.running) {
            showToast('Training already running.', 'info');
            return;
        }
        $('#start-train').disabled = true;
        $('#stop-train').disabled = false;
        $('#resume-train').disabled = true;
        const payload = {
            modality: state.lastConfig.trainer.modality,
            trainer_ctor: state.lastConfig.trainer.trainer_ctor,
            train_args: state.lastConfig.train,
            use_generic: state.lastConfig.trainer.use_generic,
            generic: state.lastConfig.trainer.generic,
            resume: true,
        };
        const resStart = await fetch('/api/train/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await resStart.json();
        if (!resStart.ok) throw new Error(data.error || 'Failed to resume training');
        resetConsole();
        setStatus('Training', 'training');
        state.runStart = Date.now();
        requestAnimationFrame(handleTimer);
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function runEvaluation(event) {
    event.preventDefault();
    try {
        const evalArgs = collectEvalArgs();
        persistConfig({ eval: evalArgs });
        const res = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ eval_args: evalArgs }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Evaluation failed');
        showToast(data.message || 'Evaluation complete.', 'success');
    } catch (err) {
        showToast(err.message, 'error');
    }
}

function bindEvents() {
    $('#theme-toggle').addEventListener('click', toggleTheme);
    $('#detect-dataset').addEventListener('click', detectDatasetClasses);
    $('#clear-dataset').addEventListener('click', () => {
        $('#dataset-code').value = '';
        $('#dataset-class').innerHTML = '<option value="" disabled selected>Select a class</option>';
        $('#dataset-summary').innerHTML = '';
    });
    $('#instantiate-dataset').addEventListener('click', instantiateDatasets);
    $('#use_generic').addEventListener('change', (e) => {
        $('#generic-settings').hidden = !e.target.checked;
    });
    $('#export-config').addEventListener('click', exportConfig);
    $('#import-config').addEventListener('change', importConfig);
    $('#start-train').addEventListener('click', startTraining);
    $('#stop-train').addEventListener('click', stopTraining);
    $('#resume-train').addEventListener('click', resumeTraining);
    $('#clear-logs').addEventListener('click', () => {
        $('#log-output').textContent = '';
    });
    $('#modal-close').addEventListener('click', closeModal);
    $('#modal-confirm').addEventListener('click', closeModal);
    $('#start-eval').addEventListener('click', runEvaluation);
}

window.addEventListener('DOMContentLoaded', () => {
    applyTheme(getTheme());
    setupTabs();
    restoreConfig();
    bindEvents();
    setupSocket();
});
