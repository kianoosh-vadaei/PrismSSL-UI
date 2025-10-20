const STORAGE_KEY = 'prism-console-config-v2';

const state = {
    datasetSummary: {},
    lastConfig: null,
    trainingStatus: 'idle',
    trainingPoller: null,
};

function $(selector) {
    return document.querySelector(selector);
}

function showMessage(message, type = 'neutral') {
    const area = $('#message-area');
    if (!area) return;
    area.textContent = message || '';
    area.className = '';
    if (!message) return;
    if (type === 'success') {
        area.classList.add('message-success');
    } else if (type === 'error') {
        area.classList.add('message-error');
    }
}

function stopTrainingPoller() {
    if (state.trainingPoller) {
        clearInterval(state.trainingPoller);
        state.trainingPoller = null;
    }
}

function startTrainingPoller() {
    if (state.trainingPoller) return;
    state.trainingPoller = setInterval(() => {
        checkTrainingStatus({ notifyOnComplete: true });
    }, 4000);
}

function applyTrainingStatus(running, options = {}) {
    const previous = state.trainingStatus;
    state.trainingStatus = running ? 'running' : 'idle';
    const button = $('#start-pretext-training');
    if (button) {
        const idleLabel = button.dataset.labelIdle || 'Start pretext training';
        const runningLabel = button.dataset.labelRunning || 'Training…';
        if (running) {
            button.disabled = true;
            button.textContent = runningLabel;
        } else {
            button.disabled = false;
            button.textContent = idleLabel;
        }
    }
    if (running) {
        startTrainingPoller();
    } else {
        stopTrainingPoller();
        if (options.notifyOnComplete && previous === 'running') {
            showMessage('Pretext training finished.', 'success');
        }
    }
}

function checkTrainingStatus(options = {}) {
    return fetch('/api/train/status')
        .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
        .then(({ ok, data }) => {
            if (!ok) throw new Error(data.error || 'Unable to determine training status.');
            applyTrainingStatus(Boolean(data.running), {
                notifyOnComplete: Boolean(options.notifyOnComplete),
            });
        })
        .catch((error) => {
            console.warn('Unable to fetch training status', error);
        });
}

function getStoredTheme() {
    return localStorage.getItem('prism-theme') || 'light';
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('prism-theme', theme);
    const toggle = $('#theme-toggle');
    if (toggle) {
        toggle.textContent = theme === 'dark' ? 'Dark' : 'Light';
    }
}

function toggleTheme() {
    const current = getStoredTheme();
    applyTheme(current === 'light' ? 'dark' : 'light');
}

function parseJson(text, fallback = null) {
    if (!text || !text.trim()) return fallback;
    try {
        return JSON.parse(text);
    } catch (error) {
        throw new Error(error.message || 'Invalid JSON');
    }
}

function detectDatasetClasses() {
    const code = $('#dataset-code').value;
    if (!code.trim()) {
        showMessage('Paste dataset code before running detection.', 'error');
        return;
    }
    fetch('/api/dataset/inspect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
    })
        .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
        .then(({ ok, data }) => {
            if (!ok) throw new Error(data.error || 'Unable to inspect dataset.');
            const select = $('#dataset-class');
            select.innerHTML = '<option value="" disabled selected>Select a class</option>';
            (data.classes || []).forEach((cls) => {
                const option = document.createElement('option');
                option.value = cls;
                option.textContent = cls;
                select.appendChild(option);
            });
            showMessage(`Detected ${data.classes.length} dataset class(es).`, 'success');
        })
        .catch((error) => {
            showMessage(error.message, 'error');
        });
}

function renderDatasetSummary(summary) {
    const container = $('#dataset-summary');
    container.innerHTML = '';
    const entries = Object.entries(summary || {});
    if (!entries.length) {
        const empty = document.createElement('p');
        empty.textContent = 'No dataset preview available yet.';
        container.appendChild(empty);
        return;
    }
    entries.forEach(([split, info]) => {
        const card = document.createElement('div');
        card.className = 'summary-card';
        const details = [];
        if (info.length !== undefined && info.length !== null) {
            details.push(`Length: ${info.length}`);
        }
        if (info.preview) {
            details.push(`Preview: ${info.preview}`);
        }
        card.innerHTML = `<strong>${split}</strong><br>${details.join('<br>') || 'No additional information'}`;
        container.appendChild(card);
    });
}

function instantiateDatasets() {
    const code = $('#dataset-code').value;
    const className = $('#dataset-class').value;
    if (!code.trim() || !className) {
        showMessage('Provide dataset code and select a class before previewing.', 'error');
        return;
    }
    let kwargsTrain;
    let kwargsVal;
    let kwargsTest;
    try {
        kwargsTrain = parseJson($('#dataset-kwargs-train').value, {});
        kwargsVal = parseJson($('#dataset-kwargs-val').value, null);
        kwargsTest = parseJson($('#dataset-kwargs-test').value, null);
    } catch (error) {
        showMessage(`Dataset argument error: ${error.message}`, 'error');
        return;
    }
    const payload = {
        code,
        class_name: className,
        kwargs_train: kwargsTrain,
        kwargs_val: kwargsVal,
        kwargs_test: kwargsTest,
    };
    fetch('/api/dataset/instantiate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    })
        .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
        .then(({ ok, data }) => {
            if (!ok) throw new Error(data.error || 'Dataset instantiation failed');
            state.datasetSummary = data.summary || {};
            renderDatasetSummary(state.datasetSummary);
            persistConfig({ dataset: { ...payload }, dataset_summary: state.datasetSummary });
            showMessage('Dataset summary prepared.', 'success');
        })
        .catch((error) => {
            showMessage(error.message, 'error');
        });
}

function sanitize(value) {
    if (value === undefined || value === null) return null;
    const trimmed = String(value).trim();
    return trimmed === '' ? null : trimmed;
}

function collectTrainerConfig() {
    const modality = sanitize($('#modality').value);
    const method = sanitize($('#method').value);
    if (!modality || !method) {
        throw new Error('Modality and method are required.');
    }
    const trainerCtor = {
        method,
        variant: sanitize($('#variant').value) || 'base',
        save_dir: sanitize($('#save_dir').value) || './checkpoints',
        checkpoint_interval: Number($('#checkpoint_interval').value) || 1,
        reload_checkpoint: $('#reload_checkpoint').checked,
        verbose: $('#verbose').checked,
        mixed_precision_training: $('#mixed_precision_training').checked,
        use_data_parallel: $('#use_data_parallel').checked,
    };
    const workers = $('#num_workers').value;
    if (workers) {
        trainerCtor.num_workers = Number(workers);
    }
    const wandbProject = sanitize($('#wandb_project').value);
    const wandbEntity = sanitize($('#wandb_entity').value);
    const wandbRunName = sanitize($('#wandb_run_name').value);
    const wandbNotes = sanitize($('#wandb_notes').value);
    const wandbTags = sanitize($('#wandb_tags').value);
    trainerCtor.wandb_mode = $('#wandb_mode').value || 'online';
    if (wandbProject) trainerCtor.wandb_project = wandbProject;
    if (wandbEntity) trainerCtor.wandb_entity = wandbEntity;
    if (wandbRunName) trainerCtor.wandb_run_name = wandbRunName;
    if (wandbNotes) trainerCtor.wandb_notes = wandbNotes;
    if (wandbTags) {
        trainerCtor.wandb_tags = wandbTags.split(',').map((tag) => tag.trim()).filter(Boolean);
    }
    const wandbConfig = $('#wandb_config').value;
    if (wandbConfig.trim()) {
        trainerCtor.wandb_config = parseJson(wandbConfig, {});
    }
    const useGeneric = $('#use_generic').checked;
    return {
        modality,
        trainer_ctor: trainerCtor,
        use_generic: useGeneric,
        generic: useGeneric
            ? {
                  model_name: $('#generic_model_name').value,
                  epochs: Number($('#generic_epochs').value) || 1,
                  use_lora: $('#generic_use_lora').checked,
              }
            : {},
    };
}

function collectTrainArgs() {
    const epochs = Number($('#epochs').value);
    const batchSize = Number($('#batch_size').value);
    const learningRate = Number($('#learning_rate').value);
    if (Number.isNaN(epochs) || epochs < 1) throw new Error('Epochs must be at least 1.');
    if (Number.isNaN(batchSize) || batchSize < 1) throw new Error('Batch size must be at least 1.');
    if (Number.isNaN(learningRate) || learningRate <= 0) throw new Error('Learning rate must be greater than 0.');
    const args = {
        batch_size: batchSize,
        epochs,
        start_epoch: Number($('#start_epoch').value) || 0,
        start_iteration: Number($('#start_iteration').value) || 0,
        learning_rate: learningRate,
        weight_decay: Number($('#weight_decay').value) || 0,
        optimizer: $('#optimizer').value || 'adamw',
        use_hpo: $('#use_hpo').checked,
        n_trials: Number($('#n_trials').value) || 20,
        tuning_epochs: Number($('#tuning_epochs').value) || 5,
        use_embedding_logger: $('#use_embedding_logger').checked,
    };
    if (args.use_hpo) {
        if (args.n_trials < 1) throw new Error('Number of HPO trials must be at least 1.');
        if (args.tuning_epochs < 1) throw new Error('Tuning epochs must be at least 1.');
    }
    const extra = $('#train_kwargs').value;
    if (extra.trim()) {
        args.extra_kwargs = parseJson(extra, {});
    }
    return args;
}

function collectEvalArgs() {
    if (!$('#include-eval').checked) {
        return null;
    }
    const args = {
        num_classes: Number($('#eval_num_classes').value) || 0,
        batch_size: Number($('#eval_batch_size').value) || 0,
        lr: Number($('#eval_lr').value) || 0,
        epochs: Number($('#eval_epochs').value) || 0,
        freeze_backbone: $('#eval_freeze_backbone').checked,
    };
    const extra = $('#eval_kwargs').value;
    if (extra.trim()) {
        args.extra_kwargs = parseJson(extra, {});
    }
    return args;
}

function collectDatasetConfig() {
    const code = $('#dataset-code').value;
    const className = $('#dataset-class').value;
    if (!code.trim()) throw new Error('Dataset code is required.');
    if (!className) throw new Error('Select a dataset class.');
    return {
        code,
        class_name: className,
        kwargs_train: parseJson($('#dataset-kwargs-train').value, {}),
        kwargs_val: parseJson($('#dataset-kwargs-val').value, null),
        kwargs_test: parseJson($('#dataset-kwargs-test').value, null),
    };
}

function gatherFullConfig() {
    const trainer = collectTrainerConfig();
    const train = collectTrainArgs();
    const dataset = collectDatasetConfig();
    const evalArgs = collectEvalArgs();
    const config = { trainer, train, dataset };
    if (evalArgs) {
        config.eval = evalArgs;
    }
    return config;
}

function persistConfig(partial) {
    const updated = { ...(state.lastConfig || {}), ...partial };
    state.lastConfig = updated;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
}

function restoreConfig() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return;
    try {
        const config = JSON.parse(stored);
        state.lastConfig = config;
        if (config.dataset) {
            $('#dataset-code').value = config.dataset.code || '';
            const select = $('#dataset-class');
            select.innerHTML = '<option value="" disabled>Select a class</option>';
            if (config.dataset.class_name) {
                const option = document.createElement('option');
                option.value = config.dataset.class_name;
                option.textContent = config.dataset.class_name;
                option.selected = true;
                select.appendChild(option);
            }
            $('#dataset-kwargs-train').value = config.dataset.kwargs_train
                ? JSON.stringify(config.dataset.kwargs_train, null, 2)
                : '';
            $('#dataset-kwargs-val').value = config.dataset.kwargs_val
                ? JSON.stringify(config.dataset.kwargs_val, null, 2)
                : '';
            $('#dataset-kwargs-test').value = config.dataset.kwargs_test
                ? JSON.stringify(config.dataset.kwargs_test, null, 2)
                : '';
        }
        state.datasetSummary = config.dataset_summary || {};
        if (config.trainer) {
            $('#modality').value = config.trainer.modality || '';
            const ctor = config.trainer.trainer_ctor || {};
            $('#method').value = ctor.method || '';
            $('#variant').value = ctor.variant || 'base';
            $('#save_dir').value = ctor.save_dir || './checkpoints';
            $('#checkpoint_interval').value = ctor.checkpoint_interval ?? 1;
            $('#num_workers').value = ctor.num_workers ?? '';
            $('#reload_checkpoint').checked = !!ctor.reload_checkpoint;
            $('#verbose').checked = ctor.verbose !== false;
            $('#mixed_precision_training').checked = ctor.mixed_precision_training !== false;
            $('#use_data_parallel').checked = !!ctor.use_data_parallel;
            $('#wandb_project').value = ctor.wandb_project || '';
            $('#wandb_entity').value = ctor.wandb_entity || '';
            $('#wandb_run_name').value = ctor.wandb_run_name || '';
            $('#wandb_mode').value = ctor.wandb_mode || 'online';
            $('#wandb_notes').value = ctor.wandb_notes || '';
            $('#wandb_tags').value = (ctor.wandb_tags || []).join(', ');
            $('#wandb_config').value = ctor.wandb_config ? JSON.stringify(ctor.wandb_config, null, 2) : '';
        }
        $('#use_generic').checked = !!config.trainer?.use_generic;
        if (config.trainer?.use_generic && config.trainer.generic) {
            $('#generic-settings').hidden = false;
            $('#generic_model_name').value = config.trainer.generic.model_name || 'bert-base-uncased';
            $('#generic_epochs').value = config.trainer.generic.epochs || 1;
            $('#generic_use_lora').checked = config.trainer.generic.use_lora !== false;
        }
        if (config.train) {
            $('#batch_size').value = config.train.batch_size || 16;
            $('#epochs').value = config.train.epochs || 1;
            $('#learning_rate').value = config.train.learning_rate || 0.0001;
            $('#weight_decay').value = config.train.weight_decay ?? 0.01;
            $('#optimizer').value = config.train.optimizer || 'adamw';
            $('#start_epoch').value = config.train.start_epoch || 0;
            $('#start_iteration').value = config.train.start_iteration || 0;
            $('#use_hpo').checked = !!config.train.use_hpo;
            $('#use_embedding_logger').checked = !!config.train.use_embedding_logger;
            $('#n_trials').value = config.train.n_trials || 20;
            $('#tuning_epochs').value = config.train.tuning_epochs || 5;
            $('#train_kwargs').value = config.train.extra_kwargs
                ? JSON.stringify(config.train.extra_kwargs, null, 2)
                : '';
        }
        if (config.eval) {
            $('#include-eval').checked = true;
            $('#eval_num_classes').value = config.eval.num_classes || 1;
            $('#eval_batch_size').value = config.eval.batch_size || 64;
            $('#eval_lr').value = config.eval.lr || 0.001;
            $('#eval_epochs').value = config.eval.epochs || 1;
            $('#eval_freeze_backbone').checked = config.eval.freeze_backbone !== false;
            $('#eval_kwargs').value = config.eval.extra_kwargs ? JSON.stringify(config.eval.extra_kwargs, null, 2) : '';
        } else {
            $('#include-eval').checked = false;
            $('#evaluation-fields').style.display = 'none';
        }
    } catch (error) {
        console.warn('Failed to restore configuration', error);
    }
}

function exportConfig() {
    try {
        const config = gatherFullConfig();
        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'prismssl-config.json';
        link.click();
        URL.revokeObjectURL(link.href);
        showMessage('Configuration exported.', 'success');
    } catch (error) {
        showMessage(error.message, 'error');
    }
}

function importConfig(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
        try {
            const config = JSON.parse(reader.result);
            localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
            restoreConfig();
            showMessage('Configuration imported.', 'success');
        } catch (error) {
            showMessage('Unable to import configuration.', 'error');
        }
    };
    reader.readAsText(file);
}

function updateEvaluationVisibility() {
    const container = $('#evaluation-fields');
    if ($('#include-eval').checked) {
        container.style.display = '';
    } else {
        container.style.display = 'none';
    }
}

function updateGenericVisibility() {
    $('#generic-settings').hidden = !$('#use_generic').checked;
}

function generateRunSheet() {
    showMessage('');
    try {
        const config = gatherFullConfig();
        persistConfig(config);
        fetch('/api/config/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                trainer: config.trainer,
                train_args: config.train,
                eval_args: config.eval || null,
                dataset: config.dataset,
            }),
        })
            .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
            .then(({ ok, data }) => {
                if (!ok) throw new Error(Object.values(data.errors || {}).join('\n') || 'Validation failed.');
                $('#config-preview').textContent = JSON.stringify(data.config, null, 2);
                showMessage('Configuration validated. You can now run training from the terminal.', 'success');
            })
            .catch((error) => {
                showMessage(error.message, 'error');
            });
    } catch (error) {
        showMessage(error.message, 'error');
    }
}

function startPretextTraining() {
    showMessage('');
    let config;
    try {
        config = gatherFullConfig();
        persistConfig(config);
    } catch (error) {
        showMessage(error.message, 'error');
        return;
    }
    const button = $('#start-pretext-training');
    if (button) {
        button.disabled = true;
    }
    fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config }),
    })
        .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
        .then(({ ok, data }) => {
            if (!ok) {
                const message =
                    Object.values(data.errors || {}).join('\n') ||
                    data.error ||
                    'Unable to start training.';
                throw new Error(message);
            }
            applyTrainingStatus(true);
            showMessage('Pretext training started in the background.', 'success');
        })
        .catch((error) => {
            applyTrainingStatus(false, { notifyOnComplete: false });
            showMessage(error.message, 'error');
        });
}

function clearDataset() {
    $('#dataset-code').value = '';
    $('#dataset-class').innerHTML = '<option value="" disabled selected>Select a class</option>';
    $('#dataset-summary').innerHTML = '';
    state.datasetSummary = {};
}

function bindEvents() {
    $('#theme-toggle').addEventListener('click', toggleTheme);
    $('#detect-dataset').addEventListener('click', detectDatasetClasses);
    $('#clear-dataset').addEventListener('click', clearDataset);
    $('#instantiate-dataset').addEventListener('click', instantiateDatasets);
    $('#export-config').addEventListener('click', exportConfig);
    $('#import-config').addEventListener('change', importConfig);
    $('#generate-run-sheet').addEventListener('click', generateRunSheet);
    $('#start-pretext-training').addEventListener('click', startPretextTraining);
    $('#use_generic').addEventListener('change', updateGenericVisibility);
    $('#include-eval').addEventListener('change', updateEvaluationVisibility);
}

window.addEventListener('DOMContentLoaded', () => {
    applyTheme(getStoredTheme());
    restoreConfig();
    renderDatasetSummary(state.datasetSummary);
    updateGenericVisibility();
    updateEvaluationVisibility();
    bindEvents();
    checkTrainingStatus();
});
