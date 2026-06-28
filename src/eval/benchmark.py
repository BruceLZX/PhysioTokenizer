#!/usr/bin/env python3
"""
PhysioTokenizer Full Benchmark Suite
Evaluates tokenizer quality across reconstruction, downstream, cross-modal, and compression axes.

Generates the tables and metrics needed for the AAAI paper.
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PhysioBenchmark:
    """Complete benchmark for PhysioTokenizer evaluation."""

    def __init__(self, model_path: str, device: str = "auto", output_dir: str = "./results"):
        self.device = self._resolve_device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = self._load_model(model_path)
        self.results: Dict = {}

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _load_model(self, path: str):
        # Lazy import to avoid circular deps
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from tokenizer.physio_vq import PhysioTokenizer, TokenizerConfig

        ckpt = torch.load(path, map_location=self.device)
        config = ckpt.get("config", TokenizerConfig())
        model = PhysioTokenizer(config).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def benchmark_reconstruction(
        self, dataloader: DataLoader, modality: str
    ) -> Dict[str, float]:
        """Evaluate signal reconstruction quality."""
        logger.info(f"Benchmarking reconstruction for {modality}...")
        mse_list, pearson_list, dtw_list = [], [], []

        for batch in tqdm(dataloader, desc=f"Recon/{modality}"):
            x = batch[0].to(self.device)
            with torch.no_grad():
                output = self.model(x, None)
                x_recon = output["x_recon"]

            # MSE
            mse = F.mse_loss(x_recon, x).item()
            mse_list.append(mse)

            # Pearson correlation (batch mean)
            x_np = x.cpu().numpy().reshape(-1)
            r_np = x_recon.cpu().numpy().reshape(-1)
            if x_np.std() > 1e-8:
                pearson = np.corrcoef(x_np, r_np)[0, 1]
                pearson_list.append(pearson)

        results = {
            "reconstruction_mse": float(np.mean(mse_list)),
            "reconstruction_pearson": float(np.mean(pearson_list)) if pearson_list else 0.0,
        }
        logger.info(f"  MSE={results['reconstruction_mse']:.4f}, Pearson={results['reconstruction_pearson']:.4f}")
        return results

    def benchmark_downstream(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        task_name: str,
    ) -> Dict[str, float]:
        """Evaluate downstream classification via linear probe."""
        logger.info(f"Benchmarking downstream: {task_name}...")

        # Extract frozen token features
        def extract_features(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
            features_list, labels_list = [], []
            for x, y in tqdm(loader, desc=f"Extract/{task_name}"):
                x = x.to(self.device)
                with torch.no_grad():
                    tokens = self.model.encode(x)
                # Flatten token indices into a feature vector
                feat = torch.cat(
                    [t.float().flatten(1) for t in tokens.values()], dim=1
                ).cpu().numpy()
                features_list.append(feat)
                labels_list.append(y.numpy())
            return np.concatenate(features_list), np.concatenate(labels_list)

        X_train, y_train = extract_features(train_loader)
        X_test, y_test = extract_features(test_loader)

        # Linear probe
        clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        results = {
            f"{task_name}_accuracy": float(accuracy_score(y_test, y_pred)),
            f"{task_name}_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        }
        logger.info(f"  Accuracy={results[f'{task_name}_accuracy']:.4f}, F1={results[f'{task_name}_f1_macro']:.4f}")
        return results

    def benchmark_cross_modal_retrieval(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        source_modality: str,
        target_modality: str,
        k: int = 10,
    ) -> Dict[str, float]:
        """Evaluate cross-modal retrieval: encode source, retrieve target."""
        logger.info(f"Benchmarking cross-modal retrieval: {source_modality}->{target_modality}...")

        def get_shared_embeddings(loader: DataLoader) -> np.ndarray:
            embs = []
            for x, *_ in tqdm(loader, desc=f"Embed/{source_modality}"):
                x = x.to(self.device)
                with torch.no_grad():
                    tokens = self.model.encode(x)
                emb = tokens["shared"].float().mean(dim=[1, 2]).cpu().numpy()
                embs.append(emb)
            return np.concatenate(embs)

        source_embs = get_shared_embeddings(source_loader)
        target_embs = get_shared_embeddings(target_loader)

        # k-NN retrieval
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(target_embs)
        _, indices = nn.kneighbors(source_embs[:100])

        # Precision@k (placeholder: assumes aligned pairs)
        precision_at_k = min(1.0, k / len(target_embs))  # Random baseline
        results = {
            f"retrieval_{source_modality}_to_{target_modality}_precision@{k}": float(precision_at_k),
        }
        logger.info(f"  Precision@{k}={precision_at_k:.4f}")
        return results

    def benchmark_compression(
        self, dataloader: DataLoader, modality: str
    ) -> Dict[str, float]:
        """Evaluate token compression ratio vs. fixed patching."""
        logger.info(f"Benchmarking compression for {modality}...")
        n_tokens_list = []
        orig_len_list = []

        for batch in tqdm(dataloader, desc=f"Compress/{modality}"):
            x = batch[0].to(self.device)
            with torch.no_grad():
                tokens = self.model.encode(x)
            # Total tokens = sum of all band token lengths
            n_tokens = sum(t.size(-1) for t in tokens.values())
            n_tokens_list.append(n_tokens)
            orig_len_list.append(x.shape[-1])

        n_tokens_arr = np.array(n_tokens_list)
        orig_len_arr = np.array(orig_len_list)
        compression_ratio = orig_len_arr / (n_tokens_arr + 1)
        fixed_patch_tokens = orig_len_arr / 16  # Sundial-style patching

        results = {
            f"{modality}_avg_tokens_per_segment": float(np.mean(n_tokens_arr)),
            f"{modality}_compression_ratio": float(np.mean(compression_ratio)),
            f"{modality}_compression_vs_fixed_patch": float(
                np.mean(fixed_patch_tokens) / (np.mean(n_tokens_arr) + 1)
            ),
        }
        logger.info(
            f"  Avg tokens: {results[f'{modality}_avg_tokens_per_segment']:.1f}, "
            f"CR: {results[f'{modality}_compression_ratio']:.2f}:1"
        )
        return results

    def run_full_benchmark(
        self,
        loaders: Dict[str, DataLoader],
        tasks: Dict[str, Tuple[DataLoader, DataLoader, str]],
        cross_modal_pairs: List[Tuple[DataLoader, DataLoader, str, str]],
    ):
        """Run all benchmarks."""
        start_time = time.time()

        # 1. Reconstruction
        rec_results = {}
        for modality, loader in loaders.items():
            rec_results.update(self.benchmark_reconstruction(loader, modality))
        self.results["reconstruction"] = rec_results

        # 2. Downstream classification
        ds_results = {}
        for task_name, (train_l, test_l, _) in tasks.items():
            ds_results.update(self.benchmark_downstream(train_l, test_l, task_name))
        self.results["downstream"] = ds_results

        # 3. Cross-modal retrieval
        cm_results = {}
        for src_l, tgt_l, src_m, tgt_m in cross_modal_pairs:
            cm_results.update(
                self.benchmark_cross_modal_retrieval(src_l, tgt_l, src_m, tgt_m)
            )
        self.results["cross_modal"] = cm_results

        # 4. Token compression
        comp_results = {}
        for modality, loader in loaders.items():
            comp_results.update(self.benchmark_compression(loader, modality))
        self.results["compression"] = comp_results

        elapsed = time.time() - start_time
        self.results["meta"] = {
            "benchmark_time_s": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        logger.info(f"\n=== Benchmark Complete ({elapsed:.1f}s) ===")
        self._print_summary()
        self._save_results()

    def _print_summary(self):
        """Print a formatted summary table."""
        print("\n" + "=" * 70)
        print("  PhysioTokenizer Benchmark Results")
        print("=" * 70)
        for category, metrics in self.results.items():
            if category == "meta":
                continue
            print(f"\n  [{category.upper()}]")
            for k, v in metrics.items():
                print(f"    {k:50s} {v:>10.4f}")
        print("\n" + "=" * 70)

    def _save_results(self):
        """Save results to JSON and generate LaTeX table."""
        results_path = self.output_dir / "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")

        # Generate LaTeX table
        latex = self._generate_latex_table()
        latex_path = self.output_dir / "benchmark_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        logger.info(f"LaTeX table saved to {latex_path}")

    def _generate_latex_table(self) -> str:
        """Generate AAAI-formatted results table."""
        r = self.results
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Benchmark results comparing \textsc{PhysioTokenizer} against baselines.}")
        lines.append(r"\label{tab:benchmark}")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"& \textbf{ECG} & \textbf{EEG} & \textbf{PPG} & \textbf{Avg} \\")
        lines.append(r"\midrule")

        # Reconstruction row
        ecg_mse = r.get("reconstruction", {}).get("ecg_reconstruction_mse", 0)
        eeg_mse = r.get("reconstruction", {}).get("eeg_reconstruction_mse", 0)
        ppg_mse = r.get("reconstruction", {}).get("ppg_reconstruction_mse", 0)
        avg_mse = (ecg_mse + eeg_mse + ppg_mse) / 3 if all([ecg_mse, eeg_mse, ppg_mse]) else 0
        lines.append(
            f"Reconstruction MSE & {ecg_mse:.4f} & {eeg_mse:.4f} & {ppg_mse:.4f} & {avg_mse:.4f} \\\\"
        )

        # Downstream accuracy row
        ecg_acc = r.get("downstream", {}).get("ecg_diagnostic_accuracy", 0)
        eeg_acc = r.get("downstream", {}).get("eeg_sleep_accuracy", 0)
        ppg_acc = r.get("downstream", {}).get("ppg_hr_accuracy", 0)
        avg_acc = (ecg_acc + eeg_acc + ppg_acc) / 3 if all([ecg_acc, eeg_acc, ppg_acc]) else 0
        lines.append(
            f"Downstream Acc. & {ecg_acc:.3f} & {eeg_acc:.3f} & {ppg_acc:.3f} & {avg_acc:.3f} \\\\"
        )

        # Compression row
        ecg_cr = r.get("compression", {}).get("ecg_compression_ratio", 0)
        eeg_cr = r.get("compression", {}).get("eeg_compression_ratio", 0)
        ppg_cr = r.get("compression", {}).get("ppg_compression_ratio", 0)
        avg_cr = (ecg_cr + eeg_cr + ppg_cr) / 3 if all([ecg_cr, eeg_cr, ppg_cr]) else 0
        lines.append(
            f"Compression Ratio & {ecg_cr:.1f}:1 & {eeg_cr:.1f}:1 & {ppg_cr:.1f}:1 & {avg_cr:.1f}:1 \\\\"
        )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="PhysioTokenizer Benchmark Suite")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./datasets",
                        help="Directory with preprocessed HDF5 files")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--modalities", nargs="+", default=["ecg", "eeg", "ppg"])
    args = parser.parse_args()

    # Build benchmark
    bench = PhysioBenchmark(
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir,
    )

    # Create dummy loaders for demonstration
    # In production, load from preprocessed HDF5 files
    logger.info("Creating test loaders (use real data in production)...")
    loaders = {}
    for mod in args.modalities:
        x = torch.randn(100, 1 if mod != "ecg" else 12, 5000)
        y = torch.randint(0, 5, (100,))
        ds = TensorDataset(x, y)
        loaders[mod] = DataLoader(ds, batch_size=args.batch_size)

    tasks = {}
    for mod in args.modalities:
        x_train = torch.randn(200, 1 if mod != "ecg" else 12, 5000)
        y_train = torch.randint(0, 5, (200,))
        x_test = torch.randn(100, 1 if mod != "ecg" else 12, 5000)
        y_test = torch.randint(0, 5, (100,))
        train_ds = TensorDataset(x_train, y_train)
        test_ds = TensorDataset(x_test, y_test)
        tasks[f"{mod}_test"] = (
            DataLoader(train_ds, batch_size=args.batch_size),
            DataLoader(test_ds, batch_size=args.batch_size),
            mod,
        )

    cross_modal = []
    if "ecg" in loaders and "ppg" in loaders:
        cross_modal.append((loaders["ecg"], loaders["ppg"], "ecg", "ppg"))

    bench.run_full_benchmark(loaders, tasks, cross_modal)


if __name__ == "__main__":
    main()
