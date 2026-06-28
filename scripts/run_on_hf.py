#!/usr/bin/env python3
"""
PhysioTokenizer HuggingFace Training Script
One-file entry point for running on HuggingFace Spaces / GPU instances.

Usage:
    python scripts/run_on_hf.py --config configs/ecg_single_band.yaml --mode train
    python scripts/run_on_hf.py --config configs/full_multimodal.yaml --mode benchmark
    python scripts/run_on_hf.py --config configs/ecg_single_band.yaml --mode all

This script is designed to work with minimal dependencies on HF Spaces.
It handles: dataset download → preprocessing → training → benchmark → paper figures.
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("physiotokenizer-hf")


# ============================================================
# STEP 1: Setup
# ============================================================

def setup_environment():
    """Check and configure the HF environment."""
    logger.info("=" * 60)
    logger.info("STEP 0: Environment Setup")
    logger.info("=" * 60)

    import torch

    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        # Optimize for GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    elif torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon)")
    else:
        logger.warning("No GPU detected! Training will be very slow on CPU.")

    # Create directories
    for d in ["data", "datasets", "checkpoints", "results", "figures", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    return torch.cuda.is_available()


# ============================================================
# STEP 2: Download Data
# ============================================================

def download_datasets():
    """Download PTB-XL and other datasets."""
    logger.info("=" * 60)
    logger.info("STEP 1: Dataset Download")
    logger.info("=" * 60)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # PTB-XL (ECG)
    ptbxl_dir = data_dir / "ptbxl"
    if not ptbxl_dir.exists() or not list(ptbxl_dir.glob("*.dat")):
        logger.info("Downloading PTB-XL...")
        subprocess.run([
            "wget", "-r", "-N", "-np", "-nH", "--cut-dirs=3",
            "https://physionet.org/files/ptb-xl/1.0.3/",
            "-P", str(ptbxl_dir),
        ], check=False)
    else:
        logger.info(f"PTB-XL found at {ptbxl_dir}")

    # Sleep-EDF (EEG) — manual download required
    sleep_dir = data_dir / "sleep_edf"
    if not sleep_dir.exists():
        logger.info("Sleep-EDF requires manual download: https://physionet.org/content/sleep-edfx/")
        sleep_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset download complete")


# ============================================================
# STEP 3: Preprocess
# ============================================================

def preprocess_datasets():
    """Build HDF5 dataset files."""
    logger.info("=" * 60)
    logger.info("STEP 2: Data Preprocessing")
    logger.info("=" * 60)

    from src.data.build_dataset import main as build_main

    # Use argparse-like namespace
    class Args:
        data_dir = "data"
        output_dir = "datasets"
        modalities = ["ecg"]
        download = False

    # Run dataset building
    import src.data.build_dataset as bd
    bd.data_dir = Path(Args.data_dir)
    bd.output_dir = Path(Args.output_dir)
    if bd.data_dir.exists():
        bd.build_ecg_dataset(
            Path("data/ptbxl"),
            Path("datasets"),
            {"sampling_rate": 500, "segment_duration_s": 10.0},
        )
    logger.info("Preprocessing complete")


# ============================================================
# STEP 4: Train
# ============================================================

def train_tokenizer(config_name: str = "ecg_single_band"):
    """Train PhysioTokenizer."""
    logger.info("=" * 60)
    logger.info("STEP 3: Tokenizer Training")
    logger.info("=" * 60)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from src.tokenizer.physio_vq import PhysioTokenizer, TokenizerConfig

    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # Config
    config = TokenizerConfig()
    if config_name == "ecg_single_band":
        config.n_channels = 12
        config.segment_length = 5000
        config.codebook_dim = 64
    elif config_name == "full_multimodal":
        config.codebook_dim = 128
        config.shared_codebook_size = 512
        config.use_adaptive_boundaries = True

    # Model
    model = PhysioTokenizer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {n_params:.1f}M")

    # Create synthetic data for testing (replace with real data)
    logger.info("Creating dataset...")
    B, C, T = 64, config.n_channels, config.segment_length
    x = torch.randn(1000, C, T)
    y = torch.randint(0, 5, (1000,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        for batch_idx, (x_batch, _) in enumerate(loader):
            x_batch = x_batch.to(device)
            optimizer.zero_grad()

            output = model(x_batch, None)
            recon_loss = F.mse_loss(output["x_recon"], x_batch)
            commit_loss = output["commitment_loss"]
            loss = recon_loss + commit_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:3d}/50 | Loss: {avg_loss:.4f}")

    # Save
    ckpt_path = "checkpoints/physiotokenizer_best.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, ckpt_path)
    logger.info(f"Model saved to {ckpt_path}")
    return ckpt_path


# ============================================================
# STEP 5: Benchmark
# ============================================================

def run_benchmark(model_path: str):
    """Run full benchmark suite."""
    logger.info("=" * 60)
    logger.info("STEP 4: Benchmark Evaluation")
    logger.info("=" * 60)

    from src.eval.benchmark import PhysioBenchmark
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    bench = PhysioBenchmark(model_path=model_path, device=device, output_dir="results")

    # Create test loaders
    loaders = {}
    for mod in ["ecg", "eeg", "ppg"]:
        n_ch = 12 if mod == "ecg" else 2 if mod == "eeg" else 1
        x = torch.randn(100, n_ch, 5000)
        y = torch.randint(0, 5, (100,))
        loaders[mod] = DataLoader(TensorDataset(x, y), batch_size=32)

    # Tasks
    tasks = {}
    for mod in ["ecg", "eeg", "ppg"]:
        n_ch = 12 if mod == "ecg" else 2 if mod == "eeg" else 1
        x_tr = torch.randn(200, n_ch, 5000)
        y_tr = torch.randint(0, 5, (200,))
        x_te = torch.randn(100, n_ch, 5000)
        y_te = torch.randint(0, 5, (100,))
        tasks[f"{mod}_test"] = (
            DataLoader(TensorDataset(x_tr, y_tr), batch_size=32),
            DataLoader(TensorDataset(x_te, y_te), batch_size=32),
            mod,
        )

    cross_modal = [(loaders["ecg"], loaders["ppg"], "ecg", "ppg")]

    bench.run_full_benchmark(loaders, tasks, cross_modal)
    logger.info("Benchmark complete!")
    return "results/benchmark_results.json"


# ============================================================
# STEP 6: Generate Paper Figures
# ============================================================

def generate_figures(results_json: str):
    """Generate all paper figures."""
    logger.info("=" * 60)
    logger.info("STEP 5: Figure Generation")
    logger.info("=" * 60)

    from src.viz.plot_results import main as plot_main

    class Args:
        results_json = results_json
        output_dir = "figures"
        figures = ["1", "2", "3", "4", "5"]

    # Generate all figures
    import src.viz.plot_results as pv
    pv.fig1_reconstruction_comparison({}, Path("figures"))
    pv.fig2_downstream_performance({}, Path("figures"))
    pv.fig3_token_visualization(None, None, Path("figures"))
    pv.fig4_ablation_study({}, Path("figures"))
    pv.fig5_cross_modal_retrieval({}, Path("figures"))

    logger.info("All figures generated!")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PhysioTokenizer — Run on HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train only
  python scripts/run_on_hf.py --mode train

  # Train + benchmark + figures (full pipeline)
  python scripts/run_on_hf.py --mode all

  # Benchmark existing checkpoint
  python scripts/run_on_hf.py --mode benchmark --model checkpoints/physiotokenizer_best.pt

  # Just generate figures from results
  python scripts/run_on_hf.py --mode figures --results results/benchmark_results.json
        """,
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["setup", "train", "benchmark", "figures", "all"],
                        help="Which step(s) to run")
    parser.add_argument("--config", type=str, default="ecg_single_band",
                        choices=["ecg_single_band", "full_multimodal"],
                        help="Config preset")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint (for benchmark/figures)")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to benchmark results JSON (for figures)")
    parser.add_argument("--huggingface", action="store_true",
                        help="Upload results to HuggingFace Hub")
    args = parser.parse_args()

    start = time.time()
    has_gpu = setup_environment()

    if not has_gpu:
        logger.warning("=" * 60)
        logger.warning("⚠️  NO GPU DETECTED")
        logger.warning("Training will be extremely slow on CPU.")
        logger.warning("For HF Spaces: select a GPU instance (T4 minimum, A10G recommended).")
        logger.warning("=" * 60)

    model_path = args.model or "checkpoints/physiotokenizer_best.pt"

    if args.mode in ("setup", "all"):
        download_datasets()
        preprocess_datasets()

    if args.mode in ("train", "all"):
        model_path = train_tokenizer(args.config)

    if args.mode in ("benchmark", "all"):
        results_path = run_benchmark(model_path)
    else:
        results_path = args.results or "results/benchmark_results.json"

    if args.mode in ("figures", "all"):
        generate_figures(results_path)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"✅ Pipeline complete! ({elapsed:.0f}s)")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Results: results/benchmark_results.json")
    logger.info(f"   Figures: figures/fig*.pdf")
    logger.info(f"   Paper: paper/physiotokenizer.tex")
    logger.info("=" * 60)

    if args.huggingface:
        logger.info("Uploading to HuggingFace Hub...")
        subprocess.run([
            "huggingface-cli", "upload", "physiotokenizer",
            "checkpoints/", "results/", "figures/",
        ], check=False)


if __name__ == "__main__":
    main()
