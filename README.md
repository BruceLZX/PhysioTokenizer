# PhysioTokenizer

> **Research Question**: How should multimodal physiological signals (ECG, EEG, PPG, sleep, ICU vitals) be tokenized for foundation model pretraining?

**Status**: 📚 Literature Review Phase | **Target Venues**: ICML / NeurIPS / ICLR / MLHC

## Quick Links
- [📖 Full Literature Review](LITERATURE_REVIEW.md)
- [🎯 Research Proposal (TBD)](PROPOSAL.md)
- [💻 Code (TBD)](src/)

## Core Idea

Current time-series foundation models use fixed-window patching — the same for stock prices and ECG. But physiological signals have structure (heartbeats, sleep spindles, frequency bands) that should inform how we tokenize them.

**PhysioTokenizer** learns a discrete physiological vocabulary:
1. **Frequency-aware**: Different codebooks for delta/theta/alpha/beta/gamma bands
2. **Continuous-time-native**: Token boundaries adapt to physiological events + sampling density
3. **Cross-modal**: Shared codebook across ECG/EEG/PPG capturing common physiological patterns

## Top 5 Papers to Read First

1. **MIRA** (NeurIPS 2025) — Medical TS foundation model with CT-RoPE + Freq-MoE + Neural ODE
2. **Sundial** (ICML 2025 Oral) — Flow matching + 1 trillion time points pretraining
3. **CLMT** (May 2026) — Hierarchical RVQ for PPG↔ECG tokenization
4. **Hypnos** (June 2026) — Next-token prediction with RVQ for sleep (8 modalities)
5. **SPOTR** (IJCAI-ECAI 2026) — Single-token reconstruction for EEG/ECG/PPG

## Three Entry Points

| Entry | Scope | Compute | Timeline | Risk |
|-------|-------|---------|----------|------|
| **PhysioVQ** | ECG + PPG, frequency-aware RVQ | A10G | 4-6 weeks | Low |
| **CT-Tokenizer** | Build on MIRA, adaptive token boundaries | A100 | 8-12 weeks | Medium |
| **PhysioCodex** | ECG + EEG + PPG shared codebook | A100 | 12-16 weeks | Medium-High |

## Lab Pitch

> "I want to study how physiological signals should be represented for foundation models — not just applying existing architectures, but designing tokenization and continuous-time representations that respect the physics of biological signals."

**Target labs**: Yuzhe Yang (HAIL), Corey Arnold, general ML/representation learning labs
