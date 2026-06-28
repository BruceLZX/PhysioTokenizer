#!/bin/bash
# Push experiment files to HF Space
# Usage: bash scripts/push_to_hf.sh
set -e

HF_SPACE="https://huggingface.co/spaces/RustedFish/PaperSpace"
TMPDIR="/tmp/PaperSpace_push"

echo "Cloning HF Space..."
rm -rf "$TMPDIR"
git clone "$HF_SPACE" "$TMPDIR"

echo "Cleaning old files..."
cd "$TMPDIR"
rm -rf src scripts requirements.txt Dockerfile README.md

echo "Copying experiment files..."
mkdir -p src/tokenizer scripts
touch src/__init__.py src/tokenizer/__init__.py

cp "$HOME/CodingProject/PhysioTokenizer/scripts/run_on_hf.py" scripts/
cp "$HOME/CodingProject/PhysioTokenizer/src/tokenizer/physio_vq.py" src/tokenizer/
cp "$HOME/CodingProject/PhysioTokenizer/requirements.txt" .
cp "$HOME/CodingProject/PhysioTokenizer/Dockerfile.hf" Dockerfile

echo "Committing and pushing..."
git add -A
git commit -m "Update experiment pipeline $(date +%Y-%m-%d_%H:%M)" || echo "No changes"
git push origin main

echo "Done. Space will rebuild at:"
echo "  https://huggingface.co/spaces/RustedFish/PaperSpace"
