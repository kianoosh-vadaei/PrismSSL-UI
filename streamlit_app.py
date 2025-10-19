# streamlit_app.py — UI with clean log capture shown in st.text_area + auto-refresh while training

import json
import types
import tempfile
import importlib.util
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

# NEW: capture + clean logs + refresh
import sys, io, re, time

import streamlit as st
import torch.nn as nn
from torch.utils.data import Dataset

# ========= NEW: inline code editor =========
try:
    from streamlit_ace import st_ace
    _ACE_AVAILABLE = True
except Exception:
    _ACE_AVAILABLE = False

# ========================
# Page Setup & Theming
# ========================
st.set_page_config(page_title="MK-SSL • Research Toolbox", page_icon="🧪", layout="wide")
st.markdown("""
<style>
    :root { --mk-card: rgba(255,255,255,0.05); --mk-border: rgba(255,255,255,0.12);
            --mk-grad-a: #7c3aed; --mk-grad-b: #06b6d4; }
    .mk-hero { padding: 22px 24px; border-radius: 18px;
               background: linear-gradient(135deg, var(--mk-grad-a), var(--mk-grad-b));
               color: white; font-weight: 800; letter-spacing:.4px; font-size: 1.8rem;
               box-shadow: 0 14px 32px rgba(0,0,0,.28); }
    .mk-subtle { margin-top: 8px; opacity:.95; font-size:1.05rem; }
    .mk-card { background: var(--mk-card); border: 1px solid var(--mk-border);
               border-radius: 18px; padding: 16px 18px;
               box-shadow: inset 0 1px 0 rgba(255,255,255,0.02); }
    .stMetric label, .stSelectbox label, .stNumberInput label,
    .stTextInput label, .stTextArea label, .stSlider label { font-size:1.05rem !important; }
    .stButton>button { border-radius: 12px; padding: 10px 16px; font-size:1.05rem; }
</style>
""", unsafe_allow_html=True)

# ========================
# Logging capture (stderr tee + ANSI strip)
# ========================
ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")

class TeeStderr(io.TextIOBase):
    """Mirror sys.stderr while keeping a buffer we can read in the UI."""
    def __init__(self, *mirrors):
        self.buf = io.StringIO()
        self.mirrors = list(mirrors)
    def write(self, s):
        self.buf.write(s)
        for m in self.mirrors:
            try:
                m.write(s); m.flush()
            except Exception:
                pass
        return len(s)
    def flush(self):
        for m in self.mirrors:
            try: m.flush()
            except Exception: pass
    def getvalue(self) -> str:
        return self.buf.getvalue()
    def clear(self):
        self.buf = io.StringIO()

def _init_state():
    ss = st.session_state
    ss.setdefault("is_running", False)
    ss.setdefault("current_epoch", 0)
    ss.setdefault("train_loss", None)
    ss.setdefault("val_loss", None)
    ss.setdefault("trainer", None)
    ss.setdefault("backbone_obj", None)
    ss.setdefault("train_dataset_obj", None)
    ss.setdefault("val_dataset_obj", None)
    ss.setdefault("train_thread", None)
    # logging capture
    ss.setdefault("orig_stderr", sys.stderr)
    ss.setdefault("tee", None)
_init_state()

def ensure_log_capture():
    """Install a tee on sys.stderr so colorlog handlers write into our buffer."""
    if st.session_state.tee is None:
        st.session_state.tee = TeeStderr(st.session_state.orig_stderr)
        sys.stderr = st.session_state.tee

def clear_logs():
    """Reset the buffer but keep capture installed."""
    if st.session_state.tee is None:
        ensure_log_capture()
    else:
        st.session_state.tee.clear()

def get_clean_logs() -> str:
    """Return buffered logs with ANSI sequences stripped (text_area-safe)."""
    text = st.session_state.tee.getvalue() if st.session_state.tee else ""
    text = ANSI_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # compress excessive blank lines
    return text

def _rerun():
    """Streamlit version-safe rerun."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ========================
# Utils (MAIN THREAD ONLY)
# ========================
def parse_json_or_none(text: str):
    if not text or not text.strip():
        return None
    return json.loads(text)

# ---- NEW: import helpers that accept either bytes or pasted text ----
def _import_module_from_content(content: bytes, filename: str = "uploaded_module.py") -> types.ModuleType:
    tmp_dir = tempfile.mkdtemp(prefix="mkssl_")
    tmp_path = Path(tmp_dir) / filename
    tmp_path.write_bytes(content)
    spec = importlib.util.spec_from_file_location(tmp_path.stem, tmp_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from file: {tmp_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

def import_module_from_file(file_obj, filename: str = "uploaded_module.py") -> types.ModuleType:
    content = file_obj.read()
    return _import_module_from_content(content, filename=filename)

def import_module_from_text(text: str, filename: str = "uploaded_module.py") -> types.ModuleType:
    return _import_module_from_content(text.encode("utf-8"), filename=filename)

def build_backbone(backbone_mode: str, uploaded_or_text, class_name: str, kwargs_json: str) -> Optional[nn.Module]:
    if backbone_mode == "Default (None)":
        return None
    if uploaded_or_text is None or (isinstance(uploaded_or_text, str) and not uploaded_or_text.strip()):
        raise ValueError("Provide Python code defining your nn.Module class.")
    # Accept either a pasted string or a file-like object:
    if isinstance(uploaded_or_text, str):
        mod = import_module_from_text(uploaded_or_text, filename="backbone.py")
    else:
        mod = import_module_from_file(uploaded_or_text, filename="backbone.py")
    if not hasattr(mod, class_name):
        raise AttributeError(f"Class '{class_name}' not found in uploaded code.")
    cls = getattr(mod, class_name)
    kwargs = parse_json_or_none(kwargs_json) or {}
    obj = cls(**kwargs)
    if not isinstance(obj, nn.Module):
        raise TypeError(f"'{class_name}' is not a torch.nn.Module.")
    return obj

def build_dataset(uploaded_or_text, class_name: str, kwargs_json: str, required: bool = True) -> Optional[Dataset]:
    if uploaded_or_text is None or (isinstance(uploaded_or_text, str) and not uploaded_or_text.strip()):
        if required:
            raise ValueError("Provide Python code defining your Dataset class.")
        return None
    if isinstance(uploaded_or_text, str):
        mod = import_module_from_text(uploaded_or_text, filename="dataset.py")
    else:
        mod = import_module_from_file(uploaded_or_text, filename="dataset.py")
    if not hasattr(mod, class_name):
        raise AttributeError(f"Dataset class '{class_name}' not found.")
    cls = getattr(mod, class_name)
    kwargs = parse_json_or_none(kwargs_json) or {}
    obj = cls(**kwargs)
    if not isinstance(obj, Dataset):
        raise TypeError(f"'{class_name}' is not a torch.utils.data.Dataset.")
    return obj

def get_trainer_class(modality: str):
    if modality == "audio":
        from MK_SSL.audio.Trainer import Trainer
        return Trainer
    if modality == "vision":
        from MK_SSL.vision.Trainer import Trainer
        return Trainer
    if modality == "graph":
        from MK_SSL.graph.Trainer import Trainer
        return Trainer
    if modality == "cross-modal":
        from MK_SSL.multimodal.Trainer import Trainer
        return Trainer
    raise ValueError(f"Unsupported modality: {modality}")

# ========================
# Sidebar — Configs (MAIN THREAD)
# ========================
with st.sidebar:
    st.markdown("### ⚙️ **Global Setup** ✨")
    run_name = st.text_input("🏷️ Run Name", value=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    save_dir = st.text_input("📁 Save Directory", value="checkpoints")
    checkpoint_interval = st.number_input("🧷 Checkpoint Interval", min_value=1, max_value=100000, value=10)
    reload_ckpt = st.toggle("🔁 Reload Checkpoint", value=False)
    mixed_precision = st.toggle("⚡ Mixed Precision", value=True)
    verbose = st.toggle("🔊 Verbose Logging", value=True)
    use_dp = st.toggle("🧩 DataParallel", value=False)
    num_workers = st.number_input("🧵 Num Workers", min_value=0, max_value=2048, value=8)

    st.markdown("---")
    modality = st.selectbox("🎛️ Modality", ["audio", "vision", "graph", "cross-modal"])
    def methods_for(m):
        return {
            "audio": ["Wav2Vec2", "HuBERT", "SimCLR", "COLA", "EAT"],
            "vision": ["SimCLR", "BYOL", "MoCoV2", "DINO", "BarlowTwins", "SwAV", "SimSiam"],
            "graph": ["GraphCL"],
            "cross-modal": ["CLIP", "SLIP", "ALBEF", "SIMVLM", "UNITER_VQA", "VSE", "CLAP", "AUDIO_CLIP", "WAV2CLIP"],
        }[m]
    method = st.selectbox("🧬 Method", methods_for(modality))
    variant = st.text_input("🧪 Variant (optional)", value="")

    st.markdown("---")
    st.markdown("### 📚 **Datasets (PyTorch Dataset .py)**")

    # ===== REPLACED: file_uploader -> Ace editor (keeps the same placement/labels) =====
    if not _ACE_AVAILABLE:
        st.warning("Install `streamlit-ace` to enable the inline code editor. Falling back to plain text.")
    st.markdown("**📄 Train Dataset .py**")
    tr_py = st_ace(language="python", theme="dracula", height=180, key="train_py") if _ACE_AVAILABLE else st.text_area("Train Dataset .py", key="train_py_fallback", height=180, label_visibility="collapsed")

    tr_cls = st.text_input("🏷️ Train Dataset Class", key="train_cls")
    tr_kwargs = st.text_area("⚙️ Train Dataset kwargs (JSON)", key="train_kwargs", placeholder='{"length": 500, "input_dim": 64}')

    st.markdown("**📄 Val Dataset .py (optional)**")
    val_py = st_ace(language="python", theme="dracula", height=160, key="val_py") if _ACE_AVAILABLE else st.text_area("Val Dataset .py (optional)", key="val_py_fallback", height=160, label_visibility="collapsed")

    val_cls = st.text_input("🏷️ Val Dataset Class (optional)", key="val_cls")
    val_kwargs = st.text_area("⚙️ Val Dataset kwargs (JSON, optional)", key="val_kwargs")

    st.markdown("---")
    backbone_mode = st.radio("Backbone Source", ["Default (None)", "Upload .py (nn.Module)"])

    # ===== REPLACED: backbone uploader -> Ace editor (only shown when selected) =====
    if backbone_mode == "Upload .py (nn.Module)":
        st.markdown("**📄 Backbone .py**")
        bb_py = st_ace(language="python", theme="dracula", height=180, key="bb_py") if _ACE_AVAILABLE else st.text_area("Backbone .py", key="bb_py_fallback", height=180, label_visibility="collapsed")
        bb_cls = st.text_input("🏷️ Backbone Class", key="bb_cls")
        bb_kwargs = st.text_area("⚙️ Backbone kwargs (JSON)", key="bb_kwargs", placeholder='{"hidden_dim": 256}')
        if st.button("🧪 Validate & Build Backbone"):
            try:
                st.session_state.backbone_obj = build_backbone("Upload .py (nn.Module)", bb_py, bb_cls, bb_kwargs or "")
                st.success(f"Built backbone: {st.session_state.backbone_obj.__class__.__name__}")
            except Exception as e:
                st.session_state.backbone_obj = None
                st.error(f"Backbone error: {e}")
    else:
        st.session_state.backbone_obj = None

    st.markdown("---")
    batch_size = st.number_input("📦 Batch Size", min_value=1, max_value=8192, value=16)
    epochs = st.number_input("⏳ Epochs", min_value=1, max_value=100000, value=100)
    lr = st.number_input("🪙 Learning Rate", min_value=1e-8, max_value=10.0, value=1e-3, format="%.8f")
    weight_decay = st.number_input("⚖️ Weight Decay", min_value=0.0, max_value=1.0, value=1e-2, format="%.6f")
    optimizer_name = st.selectbox("🧯 Optimizer", ["adam","adamw","sgd"])

    st.markdown("---")
    use_hpo = st.toggle("🧪 Use HPO (Optuna)", value=False)
    n_trials = st.number_input("🎯 Trials", min_value=1, max_value=10000, value=20, disabled=not use_hpo)
    tuning_epochs = st.number_input("🧵 Tuning Epochs", min_value=1, max_value=10000, value=5, disabled=not use_hpo)
    use_embedding_logger = st.toggle("🧿 Embedding Logger", value=False)

# ========================
# Dashboard Header & Chips (MAIN THREAD)
# ========================
st.markdown('<div class="mk-hero">🧪 MK-SSL Research Toolbox — Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="mk-subtle">Configure on the left, then initialize the trainer and start training. Logs appear below.</div>', unsafe_allow_html=True)
st.write("")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🧠 Modality", modality); c2.metric("🧬 Method", method)
c3.metric("📦 Batch Size", batch_size); c4.metric("⏳ Max Epochs", epochs)

# Metric placeholders (no auto-updates without explicit wiring)
m1, m2, m3 = st.columns(3)
m1.metric("🎯 Current Epoch", st.session_state.current_epoch)
m2.metric("📉 Train Loss", "-" if st.session_state.train_loss is None else f"{st.session_state.train_loss:.4f}")
m3.metric("✅ Val Loss", "-" if st.session_state.val_loss is None else f"{st.session_state.val_loss:.4f}")

# ========================
# Controls (MAIN THREAD)
# ========================
col_init, col_start, col_stop, col_logs = st.columns([1.4, 1.2, 1.0, 1.0])

with col_init:
    if st.button("🧱 Initialize Trainer", use_container_width=True):
        try:
            # install capture BEFORE trainer logger attaches its colorlog.StreamHandler
            ensure_log_capture()

            st.session_state.train_dataset_obj = build_dataset(tr_py, tr_cls, tr_kwargs or "", required=True)
            st.session_state.val_dataset_obj = build_dataset(val_py, val_cls, val_kwargs or "", required=False)
            Trainer = get_trainer_class(modality)
            trainer = Trainer(
                method=method,
                backbone=st.session_state.backbone_obj,
                variant=variant or None,
                save_dir=save_dir,
                checkpoint_interval=int(checkpoint_interval),
                reload_checkpoint=bool(reload_ckpt),
                verbose=bool(verbose),
                mixed_precision_training=bool(mixed_precision),
                use_data_parallel=bool(use_dp),
                num_workers=int(num_workers),
            )
            st.session_state.trainer = trainer
            st.success("Trainer is ready ✅")
        except Exception as e:
            st.session_state.trainer = None
            st.error(f"Trainer init failed: {e}")

def _train_worker(trainer, train_ds, val_ds, train_kwargs: dict):
    """Runs in background; training library does its own logging to the handler installed earlier."""
    try:
        trainer.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            **train_kwargs,
        )
    except Exception as e:
        # still goes to our tee via stderr if training logger writes there; this print is a fallback
        print(f"[Trainer thread] Error: {e}")

with col_start:
    start = st.button("▶️ **Start / Resume Training**", use_container_width=True)

with col_stop:
    stop = st.button("⏹️ **Stop**", use_container_width=True)

with col_logs:
    if st.button("🧹 Clear logs", use_container_width=True):
        clear_logs()

# Start training
if start:
    if st.session_state.trainer is None:
        st.error("Initialize trainer first.")
    elif st.session_state.train_dataset_obj is None:
        st.error("Provide train dataset code first.")
    else:
        if not st.session_state.is_running:
            ensure_log_capture()  # make sure the tee is installed before thread starts
            st.session_state.is_running = True
            trainer_ref = st.session_state.trainer
            train_ds_ref = st.session_state.train_dataset_obj
            val_ds_ref = st.session_state.val_dataset_obj
            train_kwargs = dict(
                batch_size=int(batch_size),
                epochs=int(epochs),
                lr=float(lr),
                weight_decay=float(weight_decay),
                optimizer=optimizer_name,
                use_hpo=bool(use_hpo),
                n_trials=int(n_trials) if use_hpo else 0,
                tuning_epochs=int(tuning_epochs) if use_hpo else 0,
                use_embedding_logger=bool(use_embedding_logger),
                logger_loader=None,
            )
            th = threading.Thread(
                target=_train_worker,
                args=(trainer_ref, train_ds_ref, val_ds_ref, train_kwargs),
                daemon=True,
                name="mkssl-train-thread",
            )
            st.session_state.train_thread = th
            th.start()

            # kick an immediate refresh so early logs show up right away
            time.sleep(0.05)
            _rerun()
        else:
            st.info("Training already running.")

if stop:
    st.warning("Stop requested (works only if the library supports cooperative cancel).")

# Status (flip is_running to False when thread ends)
thr = st.session_state.train_thread
if st.session_state.is_running and (thr is not None) and (not thr.is_alive()):
    st.session_state.is_running = False

st.info("🟢 Training running in background..." if st.session_state.is_running else "🟡 Training idle.")

# ========================
# Live Logs (captured + cleaned → text_area)
# ========================
st.markdown("#### 📜 **Live Logs**")
logs_clean = get_clean_logs()
st.text_area("Logs", value=logs_clean or "(No logs yet — initialize or start training.)", height=380, label_visibility="collapsed")

# ========================
# Auto-refresh while training is active
# ========================
if st.session_state.is_running:
    time.sleep(0.5)   # gentle throttle
    _rerun()
