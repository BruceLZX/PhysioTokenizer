"""
PhysioTokenizer Training Script
Supports both local (MPS) and cloud (CUDA) execution.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
import argparse
import logging

from tokenizer.physio_vq import PhysioTokenizer, TokenizerConfig
from data.dataset import PhysioDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysioTokenizerTrainer:
    """Trainer for PhysioTokenizer pretraining."""

    def __init__(
        self,
        config: TokenizerConfig,
        device: str = "auto",
        use_amp: bool = True,
    ):
        self.config = config
        self.device = self._resolve_device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.model = PhysioTokenizer(config).to(self.device)
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        logger.info(f"Using device: {self.device}")

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def train_epoch(
        self, dataloader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = total_recon = total_commit = 0.0
        n_batches = 0

        for batch in dataloader:
            x = batch["signal"].to(self.device)  # (B, C, T)
            events = batch.get("events")
            if events is not None:
                events = events.to(self.device)

            optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    output = self.model(x, events)
                    recon_loss = F.mse_loss(output["x_recon"], x)
                    commit_loss = output["commitment_loss"]
                    loss = recon_loss + commit_loss
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = self.model(x, events)
                recon_loss = F.mse_loss(output["x_recon"], x)
                commit_loss = output["commitment_loss"]
                loss = recon_loss + commit_loss
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_commit += commit_loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                logger.info(
                    f"Batch {n_batches}: loss={loss.item():.4f}, "
                    f"recon={recon_loss.item():.4f}, commit={commit_loss.item():.4f}"
                )

        return {
            "loss": total_loss / n_batches,
            "recon_loss": total_recon / n_batches,
            "commit_loss": total_commit / n_batches,
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_recon = total_commit = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                x = batch["signal"].to(self.device)
                events = batch.get("events")
                if events is not None:
                    events = events.to(self.device)

                output = self.model(x, events)
                total_recon += F.mse_loss(output["x_recon"], x).item()
                total_commit += output["commitment_loss"].item()
                n_batches += 1

        return {
            "recon_loss": total_recon / n_batches,
            "commit_loss": total_commit / n_batches,
        }

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info(f"Checkpoint saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--modality", type=str, default="ecg",
                        choices=["ecg", "eeg", "ppg", "all"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Config
    config = TokenizerConfig()
    if args.modality != "all":
        # Single modality: only relevant frequency bands
        if args.modality == "ecg":
            # ECG: heart-focused bands (0.5-40 Hz relevant)
            config.freq_bands = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 50),
            }
        elif args.modality == "eeg":
            # EEG: neural oscillation bands
            config.freq_bands = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 50),
            }
        elif args.modality == "ppg":
            # PPG: low-frequency blood volume changes
            config.freq_bands = {
                "delta": (0.5, 2),
                "theta": (2, 5),
                "alpha": (5, 10),
            }

    # Data (placeholder — implement full dataset)
    # dataset = PhysioDataset(args.data_dir, modality=args.modality)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    logger.warning("Full dataset not yet implemented. Add data loading code here.")

    # Model & trainer
    trainer = PhysioTokenizerTrainer(config, device=args.device)
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=args.lr)

    # Resume from checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Resumed from {args.resume}")

    # Training loop (placeholder)
    logger.info("Training loop ready — implement dataloader to start.")


if __name__ == "__main__":
    main()
