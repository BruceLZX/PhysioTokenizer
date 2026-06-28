"""
PhysioTokenizer — HuggingFace Space Gradio App
Provides a web UI to trigger training, view benchmarks, and browse paper figures.
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

STATUS_FILE = Path("logs/status.json")
RESULTS_FILE = Path("results/benchmark_results.json")
FIGURES_DIR = Path("figures")
CKPT_DIR = Path("checkpoints")


def init():
    """Ensure directories exist."""
    for d in [CKPT_DIR, FIGURES_DIR, Path("results"), Path("logs"), Path("data")]:
        d.mkdir(parents=True, exist_ok=True)
    if not STATUS_FILE.exists():
        save_status({"phase": "idle", "progress": 0, "message": "Ready to train", "started_at": None})


def save_status(status: dict):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)


def load_status() -> dict:
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {"phase": "idle", "progress": 0, "message": "Ready", "started_at": None}


def check_gpu() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"✅ {name} ({vram:.1f} GB VRAM)"
        elif torch.backends.mps.is_available():
            return "⚠️ Apple MPS (limited VRAM)"
        else:
            return "❌ No GPU — training will be very slow"
    except ImportError:
        return "⏳ PyTorch not installed yet"


def run_training(config_name: str = "ecg_single_band", progress=gr.Progress()):
    """Run full PhysioTokenizer pipeline: download → train → benchmark → figures."""
    progress(0.05, desc="Checking GPU...")
    save_status({"phase": "starting", "progress": 0.05, "message": f"Starting full pipeline ({config_name})...", "started_at": time.strftime("%H:%M:%S")})

    try:
        # Run the one-click pipeline script (stream output)
        proc = subprocess.Popen(
            ["python", "scripts/run_on_hf.py", "--config", config_name],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        output_lines = []
        for line in proc.stdout:
            output_lines.append(line)
            # Update progress based on log markers
            line_stripped = line.strip()
            if "STEP 1" in line_stripped:
                progress(0.1, desc="Downloading PTB-XL (3.5GB)...")
            elif "STEP 2" in line_stripped:
                progress(0.25, desc="Preprocessing ECG segments...")
            elif "STEP 3" in line_stripped:
                progress(0.35, desc="Training PhysioTokenizer (A100)...")
            elif "Epoch" in line_stripped and "/100" in line_stripped:
                try:
                    epoch = int(line_stripped.split("Epoch")[1].split("/")[0].strip())
                    progress(0.35 + 0.35 * (epoch / 100), desc=f"Training: epoch {epoch}/100")
                except:
                    pass
            elif "STEP 4" in line_stripped:
                progress(0.75, desc="Running benchmarks...")
            elif "STEP 5" in line_stripped:
                progress(0.90, desc="Generating figures...")

        proc.wait(timeout=86400)
        output_text = "".join(output_lines)

        if proc.returncode == 0:
            save_status({"phase": "complete", "progress": 1.0, "message": "✅ Pipeline complete!", "started_at": None})
            return f"✅ Pipeline complete!\n\n```\n{output_text[-3000:]}\n```"
        else:
            save_status({"phase": "error", "progress": 0, "message": "Pipeline failed", "started_at": None})
            return f"❌ Failed:\n\n```\n{output_text[-3000:]}\n```"

    except subprocess.TimeoutExpired:
        save_status({"phase": "error", "progress": 0, "message": "Timeout", "started_at": None})
        return "⏰ Timed out (24h limit)"
    except Exception as e:
        save_status({"phase": "error", "progress": 0, "message": str(e)[:100], "started_at": None})
        return f"❌ Error: {e}"


def run_benchmark(model_path: str = "checkpoints/physiotokenizer_best.pt", progress=gr.Progress()):
    """Run benchmark evaluation."""
    progress(0.1, desc="Loading model...")
    save_status({"phase": "benchmark", "progress": 0.1, "message": "Running benchmarks...", "started_at": time.strftime("%H:%M:%S")})

    result = subprocess.run(
        ["python", "src/eval/benchmark.py", "--model_path", model_path,
         "--output_dir", "results", "--batch_size", "16"],
        capture_output=True, text=True, timeout=7200,
    )

    if result.returncode == 0:
        save_status({"phase": "benchmark_done", "progress": 1.0, "message": "Benchmark complete!", "started_at": None})
        # Load results for display
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        return f"✅ Benchmark done!\n\n```\n{result.stdout[-3000:]}\n```"
    return f"❌ Error:\n\n```\n{result.stderr[-2000:]}\n```"


def run_figures(progress=gr.Progress()):
    """Generate paper figures."""
    progress(0.1, desc="Generating figures...")
    result = subprocess.run(
        ["python", "src/viz/plot_results.py", "--output_dir", "figures",
         "--figures", "1", "2", "3", "4", "5"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode == 0:
        figs = list(FIGURES_DIR.glob("*.png"))
        return f"✅ Generated {len(figs)} figures:\n" + "\n".join(f"  - {f.name}" for f in figs), *[
            str(f) for f in sorted(figs) if f.suffix == ".png"
        ]
    return f"❌ Error: {result.stderr[-1000:]}"


def run_full_pipeline(config_name: str = "ecg_single_band", progress=gr.Progress()):
    """Run full pipeline: train + benchmark + figures."""
    results = []
    results.append(run_training(config_name, progress))
    results.append(run_benchmark(progress=progress))
    figs_result = run_figures(progress)
    results.append(figs_result[0] if isinstance(figs_result, tuple) else figs_result)
    return "\n\n---\n\n".join(results)


def get_status() -> str:
    """Get current status as HTML."""
    s = load_status()
    gpu = check_gpu()

    # Check for checkpoints
    ckpts = list(CKPT_DIR.glob("*.pt"))
    ckpt_info = "\n".join(f"  📦 {c.name} ({c.stat().st_size/1e6:.1f} MB)" for c in ckpts[-3:]) if ckpts else "  (none)"

    # Check for results
    results_exist = RESULTS_FILE.exists()
    if results_exist:
        with open(RESULTS_FILE) as f:
            r = json.load(f)
        rec = r.get("reconstruction", {})
        ds = r.get("downstream", {})
        results_summary = f"""
**Reconstruction:**
  ECG MSE: {rec.get('ecg_reconstruction_mse', 'N/A'):.4f} | Pearson: {rec.get('ecg_reconstruction_pearson', 'N/A'):.4f}

**Downstream (Linear Probe Accuracy):**
  ECG: {ds.get('ecg_diagnostic_accuracy', 'N/A'):.3f} | EEG: {ds.get('eeg_sleep_accuracy', 'N/A'):.3f}"""
    else:
        results_summary = "  (not yet run)"

    # Check for figures
    figs = list(FIGURES_DIR.glob("*.pdf"))
    fig_info = "\n".join(f"  📊 {f.name}" for f in sorted(figs)) if figs else "  (none)"

    return f"""
## 🖥️ GPU Status
{gpu}

## 📊 Current Status
Phase: **{s.get('phase', 'idle')}** | Progress: {s.get('progress', 0)*100:.0f}%
Message: {s.get('message', 'Ready')}

## 💾 Checkpoints
{ckpt_info}

## 📈 Latest Results
{results_summary}

## 🎨 Generated Figures
{fig_info}
"""


def preview_figure(fig_name: str) -> str:
    """Return path to figure for preview."""
    path = FIGURES_DIR / fig_name
    if path.exists():
        return str(path)
    return None


# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(title="PhysioTokenizer") as demo:
    gr.Markdown("""
    # 🫀 PhysioTokenizer
    ### Frequency-Aware Discrete Tokenization for Multimodal Physiological Signals

    Learning discrete physiological tokens from ECG, EEG, and PPG signals.
    AAAI 2027 Submission.
    """)

    with gr.Tabs():
        # Tab 1: Dashboard
        with gr.TabItem("📊 Dashboard"):
            status_md = gr.Markdown(get_status(), every=10)

        # Tab 2: Train
        with gr.TabItem("🏋️ Train"):
            gr.Markdown("### Train PhysioTokenizer")
            config_dropdown = gr.Dropdown(
                choices=["ecg_single_band", "full_multimodal"],
                value="ecg_single_band",
                label="Config",
            )
            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
            train_output = gr.Textbox(label="Training Log", lines=15, max_lines=30)

            train_btn.click(
                fn=run_training,
                inputs=[config_dropdown],
                outputs=[train_output],
            )

        # Tab 3: Benchmark
        with gr.TabItem("📈 Benchmark"):
            gr.Markdown("### Run Benchmark Evaluation")
            model_input = gr.Textbox(
                value="checkpoints/physiotokenizer_best.pt",
                label="Model Path",
            )
            bench_btn = gr.Button("📊 Run Benchmark", variant="primary")
            bench_output = gr.Textbox(label="Results (JSON)", lines=20, max_lines=40)

            bench_btn.click(
                fn=run_benchmark,
                inputs=[model_input],
                outputs=[bench_output],
            )

        # Tab 4: Figures
        with gr.TabItem("🎨 Figures"):
            gr.Markdown("### Generate & View Paper Figures")
            fig_btn = gr.Button("🎨 Generate All Figures", variant="primary")
            fig_output = gr.Textbox(label="Status", lines=5)
            fig_gallery = gr.Gallery(label="Generated Figures", columns=3)

            fig_btn.click(
                fn=lambda: run_figures(),
                outputs=[fig_output, fig_gallery],
            )

        # Tab 5: Full Pipeline
        with gr.TabItem("🚀 Full Pipeline"):
            gr.Markdown("### Run Everything (Train → Benchmark → Figures)")
            gr.Markdown("⚠️ This will take several hours on GPU.")
            full_config = gr.Dropdown(
                choices=["ecg_single_band", "full_multimodal"],
                value="ecg_single_band",
                label="Config",
            )
            full_btn = gr.Button("🚀 Run Full Pipeline", variant="primary", size="lg")
            full_output = gr.Textbox(label="Pipeline Log", lines=25, max_lines=50)

            full_btn.click(
                fn=run_full_pipeline,
                inputs=[full_config],
                outputs=[full_output],
            )

        # Tab 6: Paper
        with gr.TabItem("📄 Paper"):
            gr.Markdown("### AAAI 2027 Paper")
            paper_path = Path("paper/physiotokenizer.tex")
            if paper_path.exists():
                with open(paper_path) as f:
                    tex_content = f.read()
                gr.Code(value=tex_content, language="latex", lines=40)
            else:
                gr.Markdown("Paper file not found.")

    gr.Markdown("---\n*PhysioTokenizer — AAAI 2027 | [GitHub](https://github.com/BruceLZX/PhysioTokenizer)*")


if __name__ == "__main__":
    init()
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
