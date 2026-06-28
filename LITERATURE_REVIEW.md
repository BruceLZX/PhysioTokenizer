# PhysioTokenizer: Learning Discrete Physiological Tokens for Multimodal Health Time Series

> **Research Direction**: Health Time-Series Foundation Models — Tokenization & Continuous-Time Representations
> **Date**: 2026-06-28
> **Status**: Literature Review & Gap Analysis

---

## 1. Topic Justification (Why This Topic?)

### 1.1 Research Problem

Current time-series foundation models (Sundial, Time-MoE, MOMENT) are designed for general domains (weather, finance, IoT). Medical physiological signals (ECG, EEG, PPG, sleep, ICU vitals) present fundamentally different challenges:

1. **Irregular sampling**: clinical measurements are event-driven, not uniformly sampled
2. **Heterogeneous frequencies**: EEG (250Hz+) vs. lab tests (daily) vs. diagnoses (sporadic)
3. **Missing-not-at-random**: missingness carries clinical information (doctor decided NOT to order a test)
4. **Cross-modal physiology**: heart, brain, respiration are coupled — but existing tokenizers treat channels independently

**Core hypothesis**: Physiological signals need their own tokenization — not fixed windows, not generic patching, but tokens that respect physiological rhythms, frequency bands, and inter-modal coupling.

### 1.2 Why This Is a Good Research Pitch

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| CS Depth | ⭐⭐⭐⭐⭐ | Tokenization, continuous-time modeling, MoE routing, SSL — all hard ML |
| Medical Relevance | ⭐⭐⭐⭐ | Direct clinical applications without needing clinical annotation |
| Compute Feasibility | ⭐⭐⭐⭐ | Public datasets + single-GPU fine-tuning (Hugging Face L4/A10G) |
| Lab Fit (HAIL) | ⭐⭐⭐⭐⭐ | Directly matches Yuzhe Yang's multisensory foundation models |
| Paper Venues | ⭐⭐⭐⭐ | ICML, NeurIPS, ICLR, MLHC, KDD Health |

### 1.3 Gap Analysis

| Existing Work | What It Does | What's Missing |
|---------------|-------------|----------------|
| MIRA (2025) | CT-RoPE + Freq-MoE + Neural ODE for medical TS | Still uses fixed window patching; no learned discrete tokenizer |
| Sundial (2025) | Flow matching + 1T time points pretraining | General domain; no physiology-specific inductive bias |
| Time-MoE (2025) | Sparse MoE at 2.4B scale | Tokenization is still simple patching; no frequency-aware routing |
| CLMT (2026) | Hierarchical RVQ for PPG↔ECG translation | Only two modalities; small scale (0.09B params) |
| Hypnos (2026) | RVQ tokenization for sleep (8 modalities) | Sleep-specific; not a general physiological tokenizer |
| SPOTR (2026) | Single-token bottleneck for EEG/ECG/PPG | Single token loses temporal structure; reconstruction-only objective |
| NormWear (2026) | CWT-based tokenization for wearables | Only wearables; no clinical EHR integration |

**The gap**: No existing work learns a **unified discrete vocabulary** for multimodal physiological signals that is (a) frequency-aware, (b) continuous-time-native, (c) cross-modal aligned, and (d) useful for both generative and discriminative downstream tasks.

---

## 2. Annotated Bibliography

### 2.1 Core Time-Series Foundation Models

#### [P1] Sundial: A Family of Highly Capable Time Series Foundation Models
- **Authors**: Yong Liu, Guo Qin, Zhiyuan Shi, Zhi Chen, Caiyin Yang, Xiangdong Huang, Jianmin Wang, Mingsheng Long (Tsinghua University)
- **Venue**: ICML 2025 **Oral** (Top 1%)
- **Links**: [arXiv:2502.00816](https://arxiv.org/abs/2502.00816) | [GitHub](https://github.com/thuml/Sundial)
- **Key Contributions**:
  - **TimeFlow Loss**: Replaces MSE/cross-entropy with flow matching objective — enables probabilistic forecasting without assuming a prior distribution
  - **TimeBench Dataset**: First trillion-scale (10^12 time points) pretraining dataset
  - **Mode Collapse Mitigation**: Sampling randomness during training prevents "over-smooth" predictions on heterogeneous data
  - **Architecture**: Decoder-only causal Transformer, 128M params (base), context length 2880, patch length 16
- **SOTA Results**: #1 MASE on GIFT-Eval; matches Chronos accuracy with 1/35th inference time
- **Relevance to PhysioTokenizer**: TimeFlow Loss is a potential training objective for physiological token generation. The decoder-only architecture could be adapted for next-token physiological prediction. However, Sundial's patching is generic — replacing it with physiological tokenization is the key contribution opportunity.
- **Limitation**: General domain only; no medical/physiological evaluation; fixed patching ignores signal structure.

#### [P2] Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts
- **Authors**: Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, Ming Jin
- **Venue**: ICLR 2025 **Spotlight** (Top 5.1%)
- **Links**: [arXiv:2409.16040](https://arxiv.org/abs/2409.16040) | [GitHub](https://github.com/Time-MoE/Time-MoE)
- **Key Contributions**:
  - First time-series FM scaled to **2.4B parameters** with sparse MoE
  - **Time-300B Dataset**: Largest open-access TS dataset (300B+ time points, 9+ domains)
  - Validated **scaling laws** for time series (training tokens + model size)
  - Flexible forecasting with context length up to 4096
- **Relevance**: The sparse MoE architecture is directly applicable to physiological signals — different experts for different frequency bands. But Time-MoE's tokenization is simple patching, and it has no medical domain specialization.
- **Limitation**: No continuous-time modeling; no handling of irregular sampling.

#### [P3] MIRA: Medical Time Series Foundation Model for Real-World Health Data
- **Authors**: Microsoft Research, University of Manchester, Peking University, Tsinghua University, Imperial College London
- **Venue**: NeurIPS 2025
- **Links**: [arXiv:2506.07584](https://arxiv.org/abs/2506.07584) | [GitHub](https://github.com/microsoft/MIRA)
- **Key Contributions**:
  - **CT-RoPE** (Continuous-Time Rotary Positional Encoding): Generalizes RoPE to real-valued irregular timestamps — attention depends on time differences, not absolute positions
  - **Frequency-Specific MoE**: Low-freq signals (weekly) → focused expert subset; high-freq (250Hz ECG) → distributed expert set
  - **Neural ODE Latent Dynamics**: Continuous trajectory modeling → forecast at arbitrary target timestamps
  - **Scale**: 454B medical time points; 455M total params, 200M active (sparse MoE)
  - **Performance**: ~8-10% OOD improvement, SOTA zero-shot on 4/5 OOD settings
- **Relevance**: **This is the closest existing work to PhysioTokenizer.** MIRA addresses irregular sampling, frequency heterogeneity, and missing values — three of the four core challenges. However, it still uses fixed-window patching rather than learned discrete tokens. PhysioTokenizer can build directly on MIRA's CT-RoPE and Freq-MoE while replacing the tokenization layer.
- **Limitation**: Tokenization is still window-based patching; no discrete vocabulary for physiological events; the MoE routing is frequency-based but not learned from signal structure.

### 2.2 Physiological Signal Tokenization (2025-2026)

#### [P4] CLMT: Compact Latent Manifold Translation — Cross-Modal Physiological Signal Synthesis
- **Authors**: Bo Cui et al. (University of Twente)
- **Venue**: Preprint, May 2026
- **Links**: [arXiv:2605.13248](https://arxiv.org/abs/2605.13248)
- **Key Contributions**:
  - **Universal Tokenizer**: Hierarchical Residual Vector Quantization (RVQ) decouples heterogeneous signals (ECG, PPG) into structured discrete latent manifolds
  - **Context-Prompted Latent Translator**: Maps tokens across modalities (PPG→ECG synthesis)
  - **Efficiency**: Only 0.09B parameters — suitable for edge deployment
  - **Results**: R-peak detection F1 0.83 (vs. 0.37 baseline); Pearson 0.9956 for cross-frequency super-resolution (25Hz→100Hz)
- **Relevance**: Directly demonstrates that RVQ-based discrete tokenization works for physiological signals. The cross-modal translation capability suggests that physiological tokens capture semantically meaningful latent structure. Key limitation: only 2 modalities (ECG, PPG); no clinical integration.
- **Limitation**: Small scale; only PPG and ECG; no temporal dynamics modeling beyond the tokenizer.

#### [P5] SPOTR: Spatio-temporal Pooling One-Token Reconstruction for Universal Physiological SSL
- **Authors**: 5GYYYYY et al.
- **Venue**: IJCAI-ECAI 2026
- **Links**: [arXiv:2606.21973](https://arxiv.org/abs/2606.21973) | [GitHub](https://github.com/5GYYYYY/SPOTR)
- **Key Contributions**:
  - Compresses each waveform into a **single global token** and reconstructs from this bottleneck
  - Pretrained on 20 datasets across EEG, iEEG, ECG, PPG
  - Improves AUC by **18.49% (EEG), 21.71% (iEEG), 17.86% (ECG), 4.64% (PPG)** under linear probing
  - ~78% lower latency, ~52% lower peak GPU memory
- **Relevance**: Shows that extreme compression (single token) can retain diagnostically useful information. The PhysioTokenizer could use a hierarchical approach: SPOTR-like global tokens + fine-grained local tokens. However, single-token compression loses temporal structure — a hybrid approach is needed.
- **Limitation**: Single token per waveform loses all temporal structure within the signal; purely reconstruction-based; no cross-modal alignment.

#### [P6] Hypnos: Next-Token Prediction Learns Generalisable Representations of Sleep Physiology
- **Authors**: (Multi-institutional)
- **Venue**: Preprint, June 2026
- **Links**: [arXiv:2606.09605](https://arxiv.org/abs/2606.09605)
- **Key Contributions**:
  - Multi-modal sleep FM using **8 modalities** (EEG, ECG, respiratory, etc.) from 20,000+ PSG recordings
  - Tokenizes each modality via **RVQ** into discrete tokens
  - Trains auto-regressive **RQ-Transformer** for next-token prediction across all modalities in parallel
  - Matches supervised baselines with **100× less labeled data**
- **Relevance**: Strong evidence that next-token prediction with discrete physiological tokens works for representation learning. The multi-modal parallel prediction design could be generalized beyond sleep to all physiological signals. The 100× label efficiency is a key selling point for medical applications.
- **Limitation**: Sleep-specific; the RVQ tokenizer is modality-specific rather than unified.

#### [P7] NormWear: Foundation Model for Multivariate Wearable Sensing of Physiological Signals
- **Authors**: (Multiple institutions)
- **Venue**: ACM Transactions on Computing for Healthcare, May 2026
- **Links**: [DOI:10.1145/3803808](https://dlnext.acm.org/doi/10.1145/3803808)
- **Key Contributions**:
  - **CWT-based tokenization** (Continuous Wavelet Transform scalograms) → modality-agnostic representations
  - Supports PPG, ECG, EEG, GSR, IMU
  - Channel-aware attention with [CLS] liaison tokens for cross-sensor fusion
  - Evaluated on 18 downstream tasks (zero-shot, few-shot, full-shot)
- **Relevance**: CWT-based tokenization is an alternative to RVQ — frequency-domain tokens naturally capture physiological rhythms. The [CLS] liaison token design could be adapted for cross-modal physiological attention. Key limitation: wearables only, no clinical integration.
- **Limitation**: Only consumer wearables; no clinical-grade signals; CWT tokens are continuous (not discrete vocabulary).

#### [P8] PhysioOmni: Towards Robust Multimodal Physiological Foundation Models
- **Authors**: Wei-Bang Jiang et al.
- **Venue**: Preprint, April 2025 (updated March 2026)
- **Links**: [arXiv:2504.19596](https://arxiv.org/abs/2504.19596)
- **Key Contributions**:
  - Foundation model for EEG, ECG, EOG, EMG
  - **Decoupled multimodal tokenizer**: modality-invariant + modality-specific pretraining
  - Handles **arbitrary missing modalities** at inference
- **Relevance**: Missing modality handling is crucial for real-world clinical deployment. The decoupled tokenizer design (shared + private subspaces) is directly applicable to PhysioTokenizer.
- **Limitation**: Still uses continuous embeddings, not discrete tokens; no temporal dynamics modeling.

### 2.3 Methodological Foundations

#### [P9] MOMENT: A Family of Open Time-Series Foundation Models
- **Authors**: Mononito Goswami et al. (CMU, Bosch, etc.)
- **Venue**: ICML 2024
- **Links**: [GitHub](https://github.com/moment-timeseries-foundation-model/moment)
- **Key Contributions**: First large-scale open time-series FM family; established the patch-based pretraining paradigm.
- **Relevance**: Baseline comparison; established the "patch + mask + reconstruct" paradigm that PhysioTokenizer should improve upon.

#### [P10] TimesFM / TimesFM-2.0 (Google)
- **Authors**: Google Research
- **Venue**: ICML 2024 / 2025
- **Key Contributions**: Decoder-only architecture for time-series forecasting at Google scale.
- **Relevance**: Architectural reference for decoder-only time-series models.

---

## 3. Research Gap Summary

| Challenge | Existing Solutions | Gap |
|-----------|-------------------|-----|
| **Physiological tokenization** | Fixed patching (MIRA, Sundial), RVQ (CLMT, Hypnos), CWT (NormWear), Single-token (SPOTR) | No unified discrete vocabulary for multimodal physiological signals that is frequency-aware and physiology-grounded |
| **Continuous-time modeling** | CT-RoPE + Neural ODE (MIRA), continuous age encoding (Delphi-2M) | Tokenization and continuous-time modeling are separate systems — no end-to-end CT-native tokenizer |
| **Cross-modal alignment** | RVQ manifolds (CLMT), decoupled tokenizer (PhysioOmni) | No shared discrete codebook across ECG/EEG/PPG/other modalities |
| **Frequency-aware routing** | Freq-MoE (MIRA), sparse MoE (Time-MoE) | MoE routing is hand-designed by frequency band, not learned from data |
| **Irregular sampling** | CT-RoPE (MIRA), Neural ODE (MIRA) | Tokenization doesn't adapt to sampling density — dense regions and sparse regions get same treatment |

---

## 4. Proposed Entry Points

### Entry Point A: "PhysioVQ" — Frequency-Aware RVQ for ECG + PPG
- **Scope**: Start with 2 modalities (ECG, PPG) from public datasets (MIMIC-IV, PTB-XL, PPG-DaLiA)
- **Method**: Extend CLMT's hierarchical RVQ with frequency-band-conditioned codebooks
- **Evaluation**: Reconstruction fidelity, downstream classification (arrhythmia, stress detection), cross-modal translation
- **Compute**: Single A10G GPU
- **Novelty**: First frequency-conditioned discrete tokenizer for physiological signals

### Entry Point B: "CT-Tokenizer" — Continuous-Time Adaptive Tokenization
- **Scope**: Build on MIRA's CT-RoPE + Neural ODE, replace fixed patching with adaptive tokenization
- **Method**: Token boundaries determined by physiological events (R-peaks, sleep spindles, etc.) AND learned from data, not fixed windows
- **Evaluation**: Zero-shot forecasting on OOD medical datasets, token interpretability
- **Compute**: A100 for pretraining, A10G for fine-tuning
- **Novelty**: First tokenizer whose segmentation is jointly determined by physiology and data-driven learning

### Entry Point C: "PhysioCodex" — Shared Discrete Codebook for Multimodal Physiology
- **Scope**: ECG + EEG + PPG, learning a shared discrete vocabulary
- **Method**: Multi-head RVQ with shared codebook + modality-specific codebooks (inspired by PhysioOmni's decoupled design)
- **Evaluation**: Cross-modal retrieval, zero-shot transfer between modalities, representation probing
- **Compute**: A100 for multi-modal pretraining
- **Novelty**: First demonstration that ECG and EEG share semantically meaningful discrete tokens

---

## 5. Recommended Reading Order

1. **Start here**: MIRA [P3] — understand the medical TS FM landscape
2. **Then**: Sundial [P1] — understand state-of-the-art in general TS FM (flow matching, scaling)
3. **Then**: CLMT [P5] + Hypnos [P6] — understand discrete tokenization for physiology
4. **Then**: SPOTR [P7] + NormWear [P8] — understand alternative tokenization paradigms
5. **Finally**: Time-MoE [P2] — understand MoE scaling for time series

**Total reading time estimate**: ~15-20 hours for deep comprehension of all 10 papers.

---

## 6. Quick-Start: Minimal Viable Experiment

```python
# Pseudo-code for PhysioVQ (Entry Point A)
# 1. Load ECG data from PTB-XL
# 2. Compute CWT → frequency bands: delta (0.5-4Hz), theta (4-8Hz), alpha (8-13Hz), beta (13-30Hz)
# 3. Per-band RVQ with learnable codebooks
# 4. Train with: reconstruction loss + contrastive loss (same patient, different time) + frequency consistency loss
# 5. Evaluate: linear probe on arrhythmia classification
```

**Estimated timeline**: 4-6 weeks for first results (assuming part-time research).

---

*Literature review compiled using ARS academic-paper lit-review mode + targeted web search. Last updated: 2026-06-28.*
