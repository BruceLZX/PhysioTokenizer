# PhysioTokenizer Experiment Runner

5-config comparison on PTB-XL: Flat VQ → Freq-Band VQ → Adaptive Boundary → Full → Raw Signal

## Run

```bash
pip install -r requirements.txt
python scripts/run_on_hf.py
```

## Output

- `results/all_results.json`
- `results/comparison_table.tex`
- `figures/fig*.pdf`
- `checkpoints/*_best.pt`
