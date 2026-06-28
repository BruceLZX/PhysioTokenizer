# PhysioTokenizer — Hardware Assessment & Experiment Plan

## Hardware Assessment

### Your Machine: Apple M4 Pro, 24GB Unified Memory

| Capability | Feasible? | Details |
|-----------|-----------|---------|
| **Data preprocessing** | ✅ Yes | CWT, event detection, segment extraction for PTB-XL/Sleep-EDF — all CPU/ANE-bound |
| **Small-scale tokenizer training** | ⚠️ Marginal | A single-band VQ on PTB-XL (21K x 10s ECG) fits in 24GB. Multi-band + multi-modal: likely OOM |
| **Full PhysioTokenizer pretraining** | ❌ No | 500K hours × 3 modalities × multi-band VQ + Transformer decoder > 24GB by wide margin |
| **Linear probe evaluation** | ✅ Yes | Frozen tokenizer + small Transformer probe (2-4 layers) fits easily |
| **Inference / demo** | ✅ Yes | Single-segment tokenization + reconstruction: ~1-2GB |

### Recommended Setup

1. **Local (M4 Pro)**: Data preprocessing, dataset building, evaluation, visualization, paper writing
2. **HuggingFace Spaces**: Tokenizer training + downstream experiments

### HuggingFace GPU Options

| GPU | VRAM | HF Cost | Suitability |
|-----|------|---------|-------------|
| T4 | 16GB | Free/$0 | Too small for multi-band VQ |
| L4 | 24GB | ~$0.80/hr | Marginal; similar to local |
| **A10G** | **24GB** | **~$1.50/hr** | **Good entry point** |
| A100-40G | 40GB | ~$3/hr | Best for full training |
| A100-80G | 80GB | ~$5/hr | Comfortable for multi-modal pretraining |

**Recommendation**: Start with A10G for single-modality VQ training; scale to A100-40G for multi-modal.

## Experiment Plan

### Phase 1: Single-Modality VQ (Week 1-3) — Local + Light GPU

```
Goal: Demonstrate frequency-band VQ works for ECG
Data: PTB-XL (21,837 x 10s @ 500Hz) — fits in 24GB
Model: Single-band RVQ (delta/theta/alpha/beta/gamma codebooks)
Baselines:
  - Fixed-patch VQ (standard time series tokenization)
  - Flat VQ (single codebook, no frequency bands)
Metrics:
  - Reconstruction MSE
  - R-peak detection F1 from tokenized vs raw signal
  - Linear probe diagnostic classification (5 classes)
```

### Phase 2: Adaptive Boundaries (Week 3-5) — A10G/A100

```
Goal: Add continuous-time adaptive token boundaries
Data: PTB-XL + Sleep-EDF (adds EEG with sleep stage labels)
Add: Boundary predictor f_θ + event detector integration
Baselines:
  - Fixed 16-sample patches (Sundial-style)
  - Fixed 1-second windows (medical standard)
Metrics:
  - Token compression ratio (fewer tokens for same information)
  - Sleep stage classification from tokenized EEG
  - Information preservation vs. token count
```

### Phase 3: Cross-Modal Shared Codebook (Week 5-8) — A100-40G

```
Goal: Shared codebook captures ECG-EEG-PPG common patterns
Data: All 3 modalities
Add: Shared codebook + alpha mixing
Baselines:
  - Per-modality separate VQ (no sharing)
  - Concatenation + linear projection (no learned sharing)
Metrics:
  - Cross-modal retrieval precision@k
  - Zero-shot transfer: train probe on ECG, test on PPG
  - Representation similarity (CKA) across modalities
```

### Phase 4: Full System & Paper (Week 8-12)

```
- Scale to full dataset
- Ablation studies
- Token interpretability visualization
- AAAI paper writing (deadline: July 28, 2026 — TIGHT!)
  → Target abstract deadline: July 21, 2026
```

## Why This Won't Work on Your Laptop

Your M4 Pro 24GB is an excellent development machine but **cannot** train PhysioTokenizer:

1. **Memory**: Multi-band VQ with 5 codebooks × 64-dim embeddings × 4 quantizers + Transformer decoder > 24GB even with batch_size=1
2. **Multi-modal**: Adding EEG (8-hour recordings!) and PPG explodes memory requirements
3. **Pretraining**: Even single-modality pretraining on full PTB-XL with gradient accumulation needs ~32GB

## HF Upload Plan

```bash
# 1. Build dataset locally
python src/data/build_dataset.py --output physio_dataset/

# 2. Upload to HuggingFace
huggingface-cli upload <your-username>/physio-dataset physio_dataset/

# 3. Create HF Space with GPU
#    Space type: Docker
#    SDK: Gradio or just training script
#    GPU: A10G (starter) or A100 (full)

# 4. Run training on HF Space
python src/train/run_pretraining.py --dataset physio_dataset/ --gpu a100

# 5. Download results
huggingface-cli download <your-username>/physiotokenizer-model .
```

## Immediate Next Steps

- [ ] Install PyTorch with MPS support: `pip install torch torchvision torchaudio`
- [ ] Download PTB-XL (small subset first): `wget -r https://physionet.org/files/ptb-xl/`
- [ ] Build dataset loader + CWT preprocessing locally
- [ ] Test single-band VQ on local M4 Pro (CPU fallback if MPS OOM)
- [ ] Set up HuggingFace account + create Space
- [ ] Upload project + dataset to HF
- [ ] Run Phase 1 training on HF GPU
