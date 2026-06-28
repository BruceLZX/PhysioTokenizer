#!/bin/bash
# PhysioTokenizer — End-to-End Pipeline Runner
# Usage: bash scripts/run_pipeline.sh [train|benchmark|figures|all]

set -euo pipefail

MODE="${1:-all}"
CONFIG="${2:-ecg_single_band}"
MODEL="${3:-checkpoints/physiotokenizer_best.pt}"
RESULTS="${4:-results/benchmark_results.json}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BLUE}╔════════════════════════════════════════════════╗${NC}"; echo -e "${BLUE}║  $*${NC}"; echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"; }

# Check GPU
check_gpu() {
    log_info "Checking GPU availability..."
    python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA GPU: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')
elif torch.backends.mps.is_available():
    print('⚠️  Apple MPS GPU (limited VRAM)')
else:
    print('❌ No GPU detected — training will be very slow')
" 2>/dev/null || log_warn "PyTorch not installed"
}

# Step 1: Setup
setup() {
    log_step "STEP 0: Environment Setup"
    pip install -q -r requirements.txt 2>/dev/null || log_warn "Some packages failed to install"
    mkdir -p data datasets checkpoints results figures logs
    log_info "Setup complete"
}

# Step 2: Download Data
download_data() {
    log_step "STEP 1: Data Download"

    # PTB-XL
    if [ ! -d "data/ptbxl" ] || [ -z "$(ls -A data/ptbxl/*.dat 2>/dev/null)" ]; then
        log_info "Downloading PTB-XL (ECG)..."
        wget -q -r -N -np -nH --cut-dirs=3 \
            "https://physionet.org/files/ptb-xl/1.0.3/" \
            -P data/ptbxl/ 2>/dev/null || log_warn "PTB-XL download failed (may need VPN)"
    else
        log_info "PTB-XL already downloaded"
    fi

    log_info "NOTE: Sleep-EDF and PPG-DaLiA require manual download."
    log_info "  Sleep-EDF: https://physionet.org/content/sleep-edfx/"
    log_info "  PPG-DaLiA: https://archive.ics.uci.edu/dataset/495/ppg+dalia"
}

# Step 3: Preprocess
preprocess() {
    log_step "STEP 2: Data Preprocessing"
    python3 src/data/build_dataset.py \
        --data_dir data \
        --output_dir datasets \
        --modalities ecg
}

# Step 4: Train
train() {
    log_step "STEP 3: Training PhysioTokenizer"
    python3 scripts/run_on_hf.py \
        --mode train \
        --config "$CONFIG"
}

# Step 5: Benchmark
benchmark() {
    log_step "STEP 4: Benchmark Evaluation"
    python3 src/eval/benchmark.py \
        --model_path "$MODEL" \
        --output_dir results \
        --batch_size 32
}

# Step 6: Figures
figures() {
    log_step "STEP 5: Generate Paper Figures"
    python3 src/viz/plot_results.py \
        --results_json "$RESULTS" \
        --output_dir figures \
        --figures 1 2 3 4 5
}

# Step 7: LaTeX compilation
compile_paper() {
    log_step "STEP 6: Compile Paper"
    if command -v pdflatex &>/dev/null; then
        cd paper
        cp ../figures/fig*.pdf . 2>/dev/null || true
        pdflatex -interaction=nonstopmode physiotokenizer.tex
        bibtex physiotokenizer
        pdflatex -interaction=nonstopmode physiotokenizer.tex
        pdflatex -interaction=nonstopmode physiotokenizer.tex
        cd ..
        log_info "Paper compiled: paper/physiotokenizer.pdf"
    else
        log_warn "pdflatex not installed. Skip paper compilation."
        log_info "Compile on Overleaf: upload paper/ folder"
    fi
}

# Main
main() {
    echo "PhysioTokenizer Pipeline"
    echo "  Mode:   $MODE"
    echo "  Config: $CONFIG"
    echo "=============================="

    check_gpu
    setup

    case "$MODE" in
        setup)
            download_data
            preprocess
            ;;
        train)
            download_data
            preprocess
            train
            ;;
        benchmark)
            benchmark
            ;;
        figures)
            figures
            ;;
        paper)
            figures
            compile_paper
            ;;
        all)
            download_data
            preprocess
            train
            benchmark
            figures
            compile_paper
            ;;
        *)
            echo "Usage: $0 [setup|train|benchmark|figures|paper|all]"
            exit 1
            ;;
    esac

    log_info "✅ Pipeline complete!"
    log_info "   Checkpoint: checkpoints/physiotokenizer_best.pt"
    log_info "   Results:    results/benchmark_results.json"
    log_info "   Figures:    figures/fig*.pdf"
    log_info "   Paper:      paper/physiotokenizer.pdf"
}

main "$@"
