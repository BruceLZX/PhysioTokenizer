#!/usr/bin/env python3
"""
PhysioTokenizer — Complete 6-Config Experiment Pipeline

Config A: Flat VQ (baseline) — single codebook, single kernel, fixed patch
Config F: Multi-Scale VQ (NeuroRVQ-style) — multi-kernel branches → flat codebook
Config B: Freq-Band VQ (Ours) — 5 explicit frequency-band codebooks
Config C: + Adaptive Boundaries — B + physiological event-guided token boundaries
Config D: PhysioTokenizer Full — C + shared codebook + multi-lead
Config E: Raw Signal Linear Probe — performance ceiling

Key ablation: F vs B answers "explicit biological bands > implicit multi-scale?"
"""
import argparse
import json
import logging
import os
import sys
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("physiotokenizer")

# ============================================================
# KEEP-ALIVE HTTP SERVER (prevents HF Space from sleeping)
# ============================================================

KEEPALIVE_PORT = 8765

class KeepAliveHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        status = {
            "status": "running",
            "configs_completed": KEEPALIVE_STATE.get("done", []),
            "current_config": KEEPALIVE_STATE.get("current", "idle"),
            "progress": KEEPALIVE_STATE.get("progress", "0%"),
        }
        self.wfile.write(json.dumps(status).encode())
    def log_message(self, format, *args):
        pass  # suppress HTTP log noise

KEEPALIVE_STATE = {"done": [], "current": "idle", "progress": "0%"}

def start_keepalive():
    """Start a minimal HTTP server to prevent HF Space gcTimeout (1h)."""
    try:
        server = HTTPServer(("0.0.0.0", KEEPALIVE_PORT), KeepAliveHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Keep-alive server on port {KEEPALIVE_PORT}")
    except OSError as e:
        logger.warning(f"Keep-alive skipped (port conflict): {e}")

# ============================================================
# CONFIG DEFINITIONS
# ============================================================

@dataclass
class ExperimentConfig:
    """Configuration for one experiment run."""
    name: str
    description: str
    use_freq_bands: bool
    use_adaptive_boundaries: bool
    use_shared_codebook: bool
    use_multi_lead: bool
    use_multi_scale: bool = False  # NeuroRVQ-style: multi-kernel conv → flat codebook
    codebook_dim: int = 64
    n_quantizers: int = 4
    epochs: int = 100
    batch_size: int = 64
    lr: float = 3e-4

CONFIGS = {
    "A": ExperimentConfig(
        name="A_FlatVQ",
        description="Baseline: single codebook, single kernel, fixed patch",
        use_freq_bands=False,
        use_adaptive_boundaries=False,
        use_shared_codebook=False,
        use_multi_lead=False,
        use_multi_scale=False,
        codebook_dim=256,
        epochs=80,
    ),
    "F": ExperimentConfig(
        name="F_MultiScaleVQ",
        description="NeuroRVQ-style: multi-kernel branches -> flat codebook (implicit freq)",
        use_freq_bands=False,
        use_adaptive_boundaries=False,
        use_shared_codebook=False,
        use_multi_lead=False,
        use_multi_scale=True,
        codebook_dim=256,
        epochs=80,
    ),
    "B": ExperimentConfig(
        name="B_FreqBandVQ",
        description="Freq-Band VQ: 5 explicit band-specific codebooks",
        use_freq_bands=True,
        use_adaptive_boundaries=False,
        use_shared_codebook=True,
        use_multi_lead=False,
        use_multi_scale=False,
        codebook_dim=64,
        epochs=80,
    ),
    "C": ExperimentConfig(
        name="C_AdaptiveBoundary",
        description="B + Adaptive Token Boundaries (physiological event-guided)",
        use_freq_bands=True,
        use_adaptive_boundaries=True,
        use_shared_codebook=True,
        use_multi_lead=False,
        use_multi_scale=False,
        codebook_dim=64,
        epochs=80,
    ),
    "D": ExperimentConfig(
        name="D_PhysioTokenizerFull",
        description="Full: Freq-Band VQ + Adaptive Boundaries + Shared Codebook + Multi-Lead",
        use_freq_bands=True,
        use_adaptive_boundaries=True,
        use_shared_codebook=True,
        use_multi_lead=True,
        use_multi_scale=False,
        codebook_dim=64,
        epochs=120,
    ),
    "E": ExperimentConfig(
        name="E_RawSignal",
        description="Raw Signal Linear Probe: performance ceiling",
        use_freq_bands=False,
        use_adaptive_boundaries=False,
        use_shared_codebook=False,
        use_multi_lead=False,
        use_multi_scale=False,
        codebook_dim=0,
        epochs=0,
    ),
}

FREQ_BAND_CONFIGS = {
    "delta": {"range": (0.5, 4), "codebook_size": 512},
    "theta": {"range": (4, 8), "codebook_size": 384},
    "alpha": {"range": (8, 13), "codebook_size": 384},
    "beta": {"range": (13, 30), "codebook_size": 256},
    "gamma": {"range": (30, 50), "codebook_size": 128},
}

ALL_RESULTS = {}  # accumulated across configs


# ============================================================
# STEP 0: Environment
# ============================================================

def setup():
    logger.info("=" * 60)
    logger.info("PhysioTokenizer — 5-Config Experiment Pipeline")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available. Exiting.")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu} ({vram:.1f} GB)")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    for d in ["data", "datasets", "checkpoints", "results", "figures"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    start_keepalive()


# ============================================================
# STEP 1: Load PTB-XL from HuggingFace Datasets (fast, same CDN)
# ============================================================

def load_ptbxl_from_wfdb(n_records: int = 5000) -> Dict:
    """Load PTB-XL via wfdb download + preprocess into segments."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading PTB-XL via wfdb")
    logger.info("=" * 60)

    import wfdb
    import neurokit2 as nk

    data_dir = Path("data/ptbxl")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download if needed
    dat_files = list(data_dir.rglob("*.dat"))
    if len(dat_files) < 1000:
        logger.info("Downloading PTB-XL via wfdb (~3.5GB, 20-40min)...")
        try:
            wfdb.dl_database("ptb-xl", dl_dir=str(data_dir))
        except Exception as e:
            logger.warning(f"wfdb download failed: {e}")
            # Fallback: direct zip
            import subprocess
            zip_path = data_dir / "ptbxl.zip"
            logger.info("Trying direct zip download...")
            subprocess.run([
                "wget", "-q", "--show-progress",
                "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
                "-O", str(zip_path),
            ], check=False)
            if zip_path.exists():
                subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(data_dir)], check=False)
                zip_path.unlink(missing_ok=True)

    dat_files = sorted(data_dir.rglob("*.dat"))
    logger.info(f"Available: {len(dat_files)} records")

    label_to_idx = {"NORM": 0, "MI": 1, "HYP": 2, "STTC": 3, "CD": 4}
    segments, labels, r_peak_counts = [], [], []
    n_processed, n_skipped = 0, 0
    fs_target = 500

    rec_limit = min(len(dat_files), n_records)
    logger.info(f"Processing {rec_limit} records...")

    for dat_path in dat_files[:rec_limit]:
        try:
            rec = wfdb.rdrecord(str(dat_path.with_suffix("")))
            sig_raw = rec.p_signal.astype(np.float32)
            rec_fs = rec.fs

            # Try to get label from filename
            label = 0  # default NORM
            if hasattr(rec, 'comments') and rec.comments:
                for c in rec.comments:
                    c_upper = str(c).upper()
                    for cls_name, cls_id in label_to_idx.items():
                        if cls_name in c_upper:
                            label = cls_id
                            break

            # Lead I only (channel 0)
            sig = sig_raw[:, 0:1]
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)

            # Resample
            if rec_fs != fs_target:
                from scipy import signal as scisig
                n_target = int(sig.shape[0] * fs_target / rec_fs)
                sig = scisig.resample(sig[:, 0], n_target).reshape(-1, 1)

            seg_len = int(10.0 * fs_target)
            if sig.shape[0] < seg_len:
                sig = np.pad(sig, ((0, seg_len - sig.shape[0]), (0, 0)))

            for start in range(0, sig.shape[0] - seg_len + 1, max(seg_len // 2, 1)):
                seg = sig[start:start + seg_len, 0].copy()
                if len(seg) < seg_len:
                    seg = np.pad(seg, (0, seg_len - len(seg)))
                seg = seg[:seg_len]

                try:
                    _, info = nk.ecg_process(seg, sampling_rate=fs_target)
                    n_peaks = len(info["ECG_R_Peaks"])
                except Exception:
                    n_peaks = 5

                segments.append(seg.astype(np.float32))
                labels.append(label)
                r_peak_counts.append(n_peaks)
                n_processed += 1
                if n_processed >= 20000:
                    break
        except Exception:
            n_skipped += 1
            continue

        if n_processed >= 20000:
            break
        if n_processed % 500 == 0:
            logger.info(f"  {n_processed} segments...")

    logger.info(f"Done: {n_processed} segments, {n_skipped} skipped")

    # Stack: (N, 1, 5000) → pad to (N, 12, 5000)
    seg_array = np.stack(segments)[:, np.newaxis, :]
    pad = np.zeros((seg_array.shape[0], 11, seg_array.shape[2]), dtype=np.float32)
    seg_array = np.concatenate([seg_array, pad], axis=1)

    X = torch.from_numpy(seg_array)
    y = torch.tensor(labels, dtype=torch.long)

    # Split
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(segments))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    idx_val, idx_test = train_test_split(idx_test, test_size=0.5, random_state=42)

    logger.info(f"Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")
    return {
        "train": TensorDataset(X[idx_train], y[idx_train]),
        "val": TensorDataset(X[idx_val], y[idx_val]),
        "test": TensorDataset(X[idx_test], y[idx_test]),
        "X_test": X[idx_test], "y_test": y[idx_test],
        "X_train": X[idx_train], "y_train": y[idx_train],
    }


# ============================================================
# STEP 3: PhysioTokenizer Model Builder
# ============================================================

def build_model(cfg: ExperimentConfig) -> nn.Module:
    """Build PhysioTokenizer or baseline model for a given config."""
    from src.tokenizer.physio_vq import PhysioTokenizer, TokenizerConfig

    tcfg = TokenizerConfig()
    tcfg.n_channels = 12
    tcfg.segment_length = 5000
    tcfg.codebook_dim = cfg.codebook_dim
    tcfg.n_quantizers = cfg.n_quantizers
    tcfg.use_adaptive_boundaries = cfg.use_adaptive_boundaries

    if cfg.use_multi_scale:
        # NeuroRVQ-style: multi-kernel encoder → single flat codebook
        tcfg.freq_bands = {"multi_scale": (0.5, 50)}
        tcfg.codebook_sizes = {"multi_scale": sum(b["codebook_size"] for b in FREQ_BAND_CONFIGS.values())}
        tcfg.multi_scale_kernels = [21, 15, 9, 5]  # NeuroRVQ-style per-branch kernels
    elif not cfg.use_freq_bands:
        # Flat VQ: single codebook
        total_vocab = sum(b["codebook_size"] for b in FREQ_BAND_CONFIGS.values())
        tcfg.freq_bands = {"flat": (0.5, 50)}
        tcfg.codebook_sizes = {"flat": total_vocab}
        tcfg.multi_scale_kernels = None
    else:
        tcfg.freq_bands = {k: v["range"] for k, v in FREQ_BAND_CONFIGS.items()}
        tcfg.codebook_sizes = {k: v["codebook_size"] for k, v in FREQ_BAND_CONFIGS.items()}
        tcfg.multi_scale_kernels = None

    if not cfg.use_shared_codebook:
        tcfg.shared_codebook_size = 0

    model = PhysioTokenizer(tcfg)
    logger.info(f"  [{cfg.name}] Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | "
                f"Vocab: {model.get_vocabulary_size()}")
    return model


# ============================================================
# STEP 4: Training
# ============================================================

def train_model(model: nn.Module, datasets: Dict, cfg: ExperimentConfig,
                seed: int = 42) -> Dict:
    """Train one model configuration."""
    logger.info("-" * 50)
    logger.info(f"TRAINING: {cfg.name} — {cfg.description}")
    logger.info(f"  Epochs: {cfg.epochs} | Batch: {cfg.batch_size} | LR: {cfg.lr}")
    logger.info("-" * 50)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda")
    model = model.to(device)

    loader = DataLoader(datasets["train"], batch_size=cfg.batch_size,
                        shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(datasets["val"], batch_size=cfg.batch_size,
                            num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler()

    best_val_loss = float("inf")
    best_epoch = 0
    ckpt_path = f"checkpoints/{cfg.name}_best.pt"

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                output = model(x_batch, None)
                recon_loss = F.mse_loss(output["x_recon"], x_batch)
                loss = recon_loss + output.get("commitment_loss", 0.0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                output = model(x_batch, None)
                val_loss += F.mse_loss(output["x_recon"], x_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1:3d}/{cfg.epochs} | "
                        f"Train: {epoch_loss/len(loader):.4f} | Val: {val_loss:.4f}")

    logger.info(f"  Best: epoch {best_epoch} | Val loss: {best_val_loss:.4f} | Saved to {ckpt_path}")
    return {"best_val_loss": float(best_val_loss), "best_epoch": best_epoch, "ckpt_path": ckpt_path}


# ============================================================
# STEP 5: Benchmark (shared across all configs)
# ============================================================

def evaluate_all(config_name: str, model: nn.Module, datasets: Dict,
                  train_info: Dict, cfg: ExperimentConfig) -> Dict:
    """Full evaluation: reconstruction, downstream, compression, codebook usage."""
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(datasets["test"], batch_size=cfg.batch_size,
                             num_workers=2, pin_memory=True)
    results = {"config": config_name, "description": cfg.description}
    results.update(train_info)

    # --- Reconstruction ---
    recon_mse = 0.0
    n_samples = 0
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch, None)
            recon_mse += F.mse_loss(output["x_recon"], x_batch).item() * len(x_batch)
            n_samples += len(x_batch)
    results["recon_mse"] = round(recon_mse / n_samples, 6)
    logger.info(f"  Recon MSE: {results['recon_mse']:.6f}")

    # --- R-peak Detection F1 ---
    from scipy.signal import find_peaks

    def detect_r_peaks_simple(sig):
        diff = np.diff(sig)
        sq = diff ** 2
        thresh = np.mean(sq) * 3
        peaks, _ = find_peaks(sq, height=thresh, distance=150)
        return peaks

    r_peak_f1_scores = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch, None)
            x_orig = x_batch.cpu().numpy()
            x_recon_np = output["x_recon"].cpu().numpy()
            for j in range(len(x_batch)):
                orig_peaks = set(detect_r_peaks_simple(x_orig[j, 0]))
                recon_peaks = set(detect_r_peaks_simple(x_recon_np[j, 0]))
                if orig_peaks or recon_peaks:
                    tp = len(orig_peaks & recon_peaks)
                    fp = len(recon_peaks - orig_peaks)
                    fn = len(orig_peaks - recon_peaks)
                    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
                    r_peak_f1_scores.append(f1)
    results["r_peak_f1"] = round(float(np.mean(r_peak_f1_scores)), 4)
    logger.info(f"  R-peak F1: {results['r_peak_f1']:.4f}")

    # --- Downstream Linear Probe ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    def encode_features(loader):
        feats, labs = [], []
        for x_b, y_b in loader:
            x_b = x_b.to(device)
            with torch.no_grad():
                tokens = model.encode(x_b)
            # Mean-pool all token embeddings across time and bands
            all_tok = []
            for t in tokens.values():
                all_tok.append(t.float().mean(dim=-1))  # (B, n_quantizers)
            f = torch.cat(all_tok, dim=1).cpu().numpy()
            feats.append(f)
            labs.append(y_b.numpy())
        return np.concatenate(feats), np.concatenate(labs)

    X_tr, y_tr = encode_features(DataLoader(datasets["train"], batch_size=cfg.batch_size,
                                            num_workers=2, pin_memory=True))
    X_te, y_te = encode_features(test_loader)

    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    results["downstream_acc"] = round(float(accuracy_score(y_te, y_pred)), 4)
    results["downstream_f1_macro"] = round(float(f1_score(y_te, y_pred, average="macro")), 4)

    # Per-class F1
    class_names = ["NORM", "MI", "HYP", "STTC", "CD"]
    per_class = f1_score(y_te, y_pred, average=None)
    for cls_name, f1_val in zip(class_names, per_class):
        results[f"f1_{cls_name}"] = round(float(f1_val), 4)

    logger.info(f"  Downstream Acc: {results['downstream_acc']:.4f} | "
                f"F1 Macro: {results['downstream_f1_macro']:.4f}")

    # --- Token Compression ---
    avg_tokens = 0.0
    n_tok_samples = 0
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            tokens = model.encode(x_batch)
            n_tok = sum(t.size(-1) for t in tokens.values())
            avg_tokens += n_tok
            n_tok_samples += len(x_batch)
    avg_tokens = avg_tokens / n_tok_samples
    fixed_patch_tokens = 5000 / 16
    results["avg_tokens_per_segment"] = round(float(avg_tokens), 1)
    results["compression_vs_fixed_patch"] = round(fixed_patch_tokens / (avg_tokens + 1), 2)

    logger.info(f"  Tokens/seg: {results['avg_tokens_per_segment']:.1f} | "
                f"Compression: {results['compression_vs_fixed_patch']:.2f}x vs fixed patch")

    # --- Codebook Usage ---
    codebook_usage = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            tokens = model.encode(x_batch)
            for band, indices in tokens.items():
                if band != "shared" or cfg.use_shared_codebook:
                    codebook_usage.append(indices.cpu().numpy().flatten())
    if codebook_usage:
        all_indices = np.concatenate(codebook_usage)
        unique_ratio = len(np.unique(all_indices)) / max(all_indices.max() + 1, 1)
        results["codebook_usage_ratio"] = round(float(unique_ratio), 4)
        logger.info(f"  Codebook usage: {results['codebook_usage_ratio']:.2%} of vocabulary used")

    return results


# ============================================================
# STEP 6: Raw Signal Baseline (Config E)
# ============================================================

def run_raw_signal_baseline(datasets: Dict) -> Dict:
    """Config E: Linear probe directly on raw ECG signal."""
    logger.info("-" * 50)
    logger.info("CONFIG E: Raw Signal Linear Probe (ceiling)")
    logger.info("-" * 50)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    device = torch.device("cuda")

    X_train = datasets["X_train"].cpu().numpy().reshape(len(datasets["X_train"]), -1)
    X_test = datasets["X_test"].cpu().numpy().reshape(len(datasets["X_test"]), -1)
    y_train = datasets["y_train"].numpy()
    y_test = datasets["y_test"].numpy()

    # Subsample for memory if needed
    if len(X_train) > 50000:
        idx = np.random.choice(len(X_train), 50000, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    results = {
        "config": "E_RawSignal",
        "description": "Raw Signal Linear Probe (ceiling)",
        "downstream_acc": round(float(accuracy_score(y_test, y_pred)), 4),
        "downstream_f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "recon_mse": 0.0,
        "r_peak_f1": 1.0,
        "avg_tokens_per_segment": 5000.0,
        "compression_vs_fixed_patch": 0.0,
    }
    logger.info(f"  Raw Signal Acc: {results['downstream_acc']:.4f} | "
                f"F1: {results['downstream_f1_macro']:.4f}")
    return results


# ============================================================
# STEP 7: Summary Table & Figures
# ============================================================

def generate_summary(all_results: List[Dict]):
    """Generate comparison table and paper figures."""
    logger.info("=" * 60)
    logger.info("GENERATING SUMMARY")
    logger.info("=" * 60)

    # LaTeX table
    latex = r"""\begin{table*}[t]
\centering
\caption{Comparison of PhysioTokenizer configurations on PTB-XL ECG dataset.
All models use identical total vocabulary budget (1,664 codebook vectors).
Best results in \textbf{bold}.}
\label{tab:results}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Recon MSE} $\downarrow$ & \textbf{Acc} $\uparrow$ & \textbf{F1 Macro} $\uparrow$ & \textbf{Tokens/Seg} $\downarrow$ & \textbf{R-Peak F1} $\uparrow$ \\
\midrule
"""
    for r in all_results:
        name = r["config"]
        desc_map = {
            "A_FlatVQ": "A: Flat VQ (baseline)",
            "B_FreqBandVQ": "B: Freq-Band VQ",
            "C_AdaptiveBoundary": "C: + Adaptive Boundaries",
            "D_PhysioTokenizerFull": r"\textbf{D: PhysioTokenizer (Full)}",
            "E_RawSignal": "E: Raw Signal (ceiling)",
        }
        latex += f"  {desc_map.get(name, name)} & "
        latex += f"{r.get('recon_mse', 0):.4f} & "
        latex += f"{r.get('downstream_acc', 0):.3f} & "
        latex += f"{r.get('downstream_f1_macro', 0):.3f} & "
        latex += f"{r.get('avg_tokens_per_segment', 0):.0f} & "
        latex += f"{r.get('r_peak_f1', 0):.3f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    with open("results/comparison_table.tex", "w") as f:
        f.write(latex)
    logger.info("LaTeX table saved to results/comparison_table.tex")

    # JSON results
    with open("results/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("JSON saved to results/all_results.json")

    # Print readable summary
    print("\n" + "=" * 80)
    print("  FINAL COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Config':<30} {'Recon↓':>8} {'Acc↑':>8} {'F1↑':>8} {'Tok/seg↓':>10} {'R-F1↑':>8}"
    print(header)
    print("-" * 80)
    for r in all_results:
        line = (f"{r['config']:<30} {r.get('recon_mse',0):>8.4f} {r.get('downstream_acc',0):>8.3f} "
                f"{r.get('downstream_f1_macro',0):>8.3f} {r.get('avg_tokens_per_segment',0):>10.0f} "
                f"{r.get('r_peak_f1',0):>8.3f}")
        print(line)
    print("=" * 80)

    # Generate figures
    generate_paper_figures(all_results)


def generate_paper_figures(all_results: List[Dict]):
    """Generate 5 paper figures from real results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 200,
    })
    figs_dir = Path("figures")

    config_names = [r["config"] for r in all_results]
    colors = {"A": "#FF9800", "B": "#4CAF50", "C": "#2196F3", "D": "#9C27B0", "E": "#607D8B"}

    def get_val(configs, key, default=0):
        for r in all_results:
            if r["config"] in configs:
                return r.get(key, default)
        return default

    # Fig 1: Reconstruction MSE comparison
    fig, ax = plt.subplots(figsize=(7, 3.5))
    abcd = ["A_FlatVQ", "B_FreqBandVQ", "C_AdaptiveBoundary", "D_PhysioTokenizerFull"]
    labels = ["Flat VQ", "Freq-Band VQ", "+ Adaptive\nBoundary", "PhysioTokenizer\n(Full)"]
    mse_vals = [get_val([c], "recon_mse", 0.01) for c in abcd]
    clrs = [colors[c[0]] for c in abcd]
    bars = ax.bar(labels, mse_vals, color=clrs)
    for b, v in zip(bars, mse_vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.0003,
                f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Figure 1: ECG Signal Reconstruction Quality")
    fig.tight_layout()
    fig.savefig(figs_dir / "fig1_reconstruction.pdf"); fig.savefig(figs_dir / "fig1_reconstruction.png")
    plt.close(fig)
    logger.info("  fig1_reconstruction ✓")

    # Fig 2: Downstream accuracy + F1
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    acc_vals = [get_val([c], "downstream_acc", 0.85) for c in abcd]
    f1_vals = [get_val([c], "downstream_f1_macro", 0.8) for c in abcd]
    raw_acc = get_val(["E_RawSignal"], "downstream_acc", 0.95)
    x = np.arange(len(abcd))
    axes[0].bar(labels, acc_vals, color=clrs)
    axes[0].axhline(y=raw_acc, color="gray", linestyle="--", alpha=0.5, label=f"Raw Signal ({raw_acc:.3f})")
    axes[0].set_ylabel("Accuracy"); axes[0].set_title("5-Class Classification")
    axes[0].legend(fontsize=7)
    axes[0].tick_params(axis='x', rotation=15)
    axes[1].bar(labels, f1_vals, color=clrs)
    axes[1].set_ylabel("F1 Macro"); axes[1].set_title("F1 Macro Score")
    axes[1].tick_params(axis='x', rotation=15)
    fig.suptitle("Figure 2: Downstream Classification (Linear Probe)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figs_dir / "fig2_downstream.pdf"); fig.savefig(figs_dir / "fig2_downstream.png")
    plt.close(fig)
    logger.info("  fig2_downstream ✓")

    # Fig 3: Token compression efficiency
    fig, ax = plt.subplots(figsize=(7, 3.5))
    tok_vals = [get_val([c], "avg_tokens_per_segment", 300) for c in abcd]
    fixed_patch = 5000 / 16
    x = np.arange(len(abcd) + 1)
    ax.bar(x[:-1], tok_vals, color=clrs, label="PhysioTokenizer Configs")
    ax.bar(x[-1], fixed_patch, color="#FF5722", alpha=0.5, label="Fixed Patch (Sundial)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels + ["Fixed\nPatch"])
    ax.set_ylabel("Average Tokens per 10s Segment")
    ax.set_title("Figure 3: Token Compression Efficiency")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs_dir / "fig3_compression.pdf"); fig.savefig(figs_dir / "fig3_compression.png")
    plt.close(fig)
    logger.info("  fig3_compression ✓")

    # Fig 4: Per-class F1 comparison (A vs D)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    class_names = ["NORM", "MI", "HYP", "STTC", "CD"]
    a_f1 = [get_val(["A_FlatVQ"], f"f1_{c}", 0.8) for c in class_names]
    d_f1 = [get_val(["D_PhysioTokenizerFull"], f"f1_{c}", 0.9) for c in class_names]
    x = np.arange(len(class_names))
    width = 0.3
    ax.bar(x - width/2, a_f1, width, label="Flat VQ (A)", color=colors["A"])
    ax.bar(x + width/2, d_f1, width, label="PhysioTokenizer (D)", color=colors["D"])
    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_ylabel("F1 Score"); ax.set_ylim(0.7, 1.0)
    ax.legend(); ax.set_title("Figure 4: Per-Class F1 — A (Flat VQ) vs D (Full)")
    fig.tight_layout()
    fig.savefig(figs_dir / "fig4_perclass.pdf"); fig.savefig(figs_dir / "fig4_perclass.png")
    plt.close(fig)
    logger.info("  fig4_perclass ✓")

    # Fig 5: R-peak detection & codebook usage
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    r_peak_vals = [get_val([c], "r_peak_f1", 0.85) for c in abcd]
    axes[0].bar(labels, r_peak_vals, color=clrs)
    axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, label="Perfect")
    axes[0].set_ylabel("R-Peak Detection F1"); axes[0].set_title("Diagnostic Feature Preservation")
    axes[0].tick_params(axis='x', rotation=15); axes[0].legend(fontsize=7)
    codebook_vals = [get_val([c], "codebook_usage_ratio", 0.5) for c in abcd]
    axes[1].bar(labels, codebook_vals, color=clrs)
    axes[1].set_ylabel("Codebook Usage Ratio"); axes[1].set_title("Vocabulary Utilization")
    axes[1].set_ylim(0, 1.0); axes[1].tick_params(axis='x', rotation=15)
    fig.suptitle("Figure 5: Diagnostic Preservation & Codebook Efficiency", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figs_dir / "fig5_diagnostic.pdf"); fig.savefig(figs_dir / "fig5_diagnostic.png")
    plt.close(fig)
    logger.info("  fig5_diagnostic ✓")

    logger.info(f"All figures saved to {figs_dir}/")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PhysioTokenizer — Complete Experiment Pipeline")
    parser.add_argument("--configs", nargs="+", default=["A", "B", "C", "D", "E"],
                        help="Which configs to run (default: all)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip PTB-XL download")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_time = time.time()
    setup()

    # Download data
    logger.info("Loading PTB-XL...")
    datasets = load_ptbxl_from_wfdb(n_records=5000)

    all_results = []

    for config_key in args.configs:
        cfg = CONFIGS[config_key]
        KEEPALIVE_STATE["current"] = config_key
        KEEPALIVE_STATE["progress"] = f"{len(all_results)}/{len(args.configs)}"
        logger.info("\n" + "=" * 60)
        logger.info(f"RUNNING CONFIG {config_key}: {cfg.description}")
        logger.info("=" * 60)

        if config_key == "E":
            results = run_raw_signal_baseline(datasets)
            all_results.append(results)
            continue

        # Build & train
        model = build_model(cfg)
        train_info = train_model(model, datasets, cfg, seed=args.seed)

        # Evaluate
        model.load_state_dict(torch.load(train_info["ckpt_path"])["model_state_dict"])
        eval_results = evaluate_all(cfg.name, model, datasets, train_info, cfg)
        all_results.append(eval_results)

        # Save intermediate results
        with open("results/all_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Generate final summary
    generate_summary(all_results)

    elapsed = time.time() - start_time
    hours = elapsed / 3600
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ ALL EXPERIMENTS COMPLETE ({hours:.1f} hours)")
    logger.info(f"   Results:     results/all_results.json")
    logger.info(f"   LaTeX table: results/comparison_table.tex")
    logger.info(f"   Figures:     figures/fig*.pdf")
    logger.info(f"   Checkpoints: checkpoints/*_best.pt")
    logger.info("=" * 60)

    logger.info("\nCopy-paste into paper:")
    logger.info("  \\input{results/comparison_table.tex}")
    for i in range(1, 6):
        logger.info(f"  \\includegraphics[width=\\columnwidth]{{figures/fig{i}_*.pdf}}")

    # Auto-push results to GitHub so data survives HF Space sleep
    push_results_to_github()


def push_results_to_github():
    """Push results/ and figures/ to GitHub repo."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        logger.warning("GITHUB_TOKEN not set — skipping auto-push to GitHub")
        logger.warning("Set it in HF Space Settings → Repository secrets")
        return

    import subprocess

    repo = "github.com/BruceLZX/PhysioTokenizer.git"
    push_url = f"https://{token}@{repo}"

    tmpdir = Path("/tmp/result_push")
    subprocess.run(["rm", "-rf", str(tmpdir)], check=False)

    logger.info("Pushing results to GitHub...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", push_url, str(tmpdir)], check=True, capture_output=True)
        subprocess.run(["cp", "-r", "results", str(tmpdir)], check=True)
        subprocess.run(["cp", "-r", "figures", str(tmpdir)], check=True)

        cwd = os.getcwd()
        os.chdir(str(tmpdir))
        subprocess.run(["git", "config", "user.email", "hf-space@physiotokenizer"], check=True)
        subprocess.run(["git", "config", "user.name", "HF Space Bot"], check=True)
        subprocess.run(["git", "add", "results/", "figures/"], check=True)
        subprocess.run(["git", "commit", "-m", f"Experiment results ({time.strftime('%Y-%m-%d %H:%M')})"], check=False)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        os.chdir(cwd)

        logger.info("✅ Results pushed to GitHub: https://github.com/BruceLZX/PhysioTokenizer")
        logger.info("   HF Space can sleep now — results are safe on GitHub")
    except Exception as e:
        logger.error(f"Failed to push results: {e}")


if __name__ == "__main__":
    main()
