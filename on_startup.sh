#!/bin/bash
# Auto-resume experiment on Space restart (runs from persistent /data)
cd /data/PhysioTokenizer 2>/dev/null || {
    git clone https://github.com/BruceLZX/PhysioTokenizer.git /data/PhysioTokenizer
    cd /data/PhysioTokenizer
    pip install -q -r requirements.txt
}
git pull origin main  # update code, keep checkpoints/data/results
pip install -q -r requirements.txt
python scripts/run_on_hf.py
