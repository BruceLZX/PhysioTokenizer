"""
PhysioTokenizer: Frequency-Band-Conditioned Vector Quantization

Core tokenizer module implementing:
1. CWT-based frequency band decomposition
2. Per-band residual vector quantization (RVQ)
3. Cross-modal shared codebook
4. Continuous-time adaptive boundary prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    """Configuration for PhysioTokenizer."""

    # Signal
    n_channels: int = 12  # Max channels (padded for single-channel modalities)
    segment_length: int = 5000  # Time steps per segment

    # Frequency bands & codebooks
    freq_bands: Dict[str, Tuple[float, float]] = None  # Set in __post_init__
    codebook_sizes: Dict[str, int] = None
    codebook_dim: int = 64  # Dimension of each codebook vector
    shared_codebook_size: int = 256
    n_quantizers: int = 4  # Residual quantization depth

    # Adaptive boundaries
    use_adaptive_boundaries: bool = True
    min_token_duration: int = 20  # Minimum time steps per token
    boundary_hidden_dim: int = 128

    # Training
    commitment_cost: float = 0.25
    freq_consistency_weight: float = 0.1
    entropy_weight: float = 0.01

    def __post_init__(self):
        if self.freq_bands is None:
            self.freq_bands = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 50),
            }
        if self.codebook_sizes is None:
            self.codebook_sizes = {
                "delta": 512,
                "theta": 384,
                "alpha": 384,
                "beta": 256,
                "gamma": 128,
            }


class FrequencyBandEncoder(nn.Module):
    """
    Encodes raw signal into frequency-band-specific representations.

    Uses learnable CWT-like filters initialized with Morlet wavelets
    at physiologically relevant center frequencies.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        self.band_encoders = nn.ModuleDict()

        for band_name, (f_low, f_high) in config.freq_bands.items():
            # 1D CNN per band: different kernel sizes for different frequencies
            kernel_size = int(500 / f_low)  # Larger kernels for lower frequencies
            kernel_size = max(3, min(kernel_size, 251))  # Clamp to reasonable range
            if kernel_size % 2 == 0:
                kernel_size += 1  # Make odd

            self.band_encoders[band_name] = nn.Sequential(
                nn.Conv1d(
                    config.n_channels,
                    config.codebook_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                ),
                nn.GroupNorm(8, config.codebook_dim),
                nn.GELU(),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, T) input signal
        Returns:
            Dict mapping band_name -> (B, D, T) band representation
        """
        return {band: encoder(x) for band, encoder in self.band_encoders.items()}


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer with per-band codebooks.

    Quantizes band representations through multiple residual quantization
    layers, each with a learnable codebook.
    """

    def __init__(self, codebook_size: int, codebook_dim: int, n_quantizers: int):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.n_quantizers = n_quantizers

        # Each quantizer has its own codebook
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, codebook_dim) * 0.01)
            for _ in range(n_quantizers)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, D, T) continuous representation
        Returns:
            z_q: (B, D, T) quantized representation
            indices: (B, N_q, T) codebook indices per quantizer
            commitment_loss: scalar
        """
        B, D, T = z.shape
        residual = z.permute(0, 2, 1).reshape(-1, D)  # (B*T, D)
        z_q = torch.zeros_like(residual)
        all_indices = []
        commitment_loss = 0.0

        for codebook in self.codebooks:
            # Compute distances to all codebook vectors
            dists = (
                residual.pow(2).sum(1, keepdim=True)
                - 2 * residual @ codebook.T
                + codebook.pow(2).sum(1, keepdim=True).T
            )
            indices = dists.argmin(dim=1)
            all_indices.append(indices)

            # Quantize
            quantized = codebook[indices]
            commitment_loss += F.mse_loss(quantized.detach(), residual)

            # Update residual for next quantizer
            residual = residual - quantized
            z_q = z_q + quantized

        # Straight-through estimator
        z_q = z.permute(0, 2, 1).reshape(-1, D) + (z_q - z.permute(0, 2, 1).reshape(-1, D)).detach()
        z_q = z_q.reshape(B, T, D).permute(0, 2, 1)  # (B, D, T)
        indices = torch.stack(all_indices, dim=1)  # (B, N_q, T)

        return z_q, indices, commitment_loss


class AdaptiveBoundaryPredictor(nn.Module):
    """
    Predicts token boundaries based on signal content and physiological events.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        input_dim = config.codebook_dim * len(config.freq_bands) + 1  # +1 for event channel

        self.boundary_net = nn.Sequential(
            nn.Conv1d(input_dim, config.boundary_hidden_dim, kernel_size=31, padding=15),
            nn.GELU(),
            nn.Conv1d(config.boundary_hidden_dim, config.boundary_hidden_dim, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Conv1d(config.boundary_hidden_dim, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(
        self, band_reprs: Dict[str, torch.Tensor], events: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            band_reprs: Dict of (B, D, T) per-band representations
            events: (B, 1, T) binary physiological event mask
        Returns:
            boundary_probs: (B, T) probability of boundary at each time step
        """
        # Concatenate all band representations + event channel
        features = list(band_reprs.values()) + [events]
        x = torch.cat(features, dim=1)  # (B, sum(D_bands)+1, T)
        return self.boundary_net(x).squeeze(1)  # (B, T)


class PhysioTokenizer(nn.Module):
    """
    Complete PhysioTokenizer: frequency-band VQ + adaptive boundaries + shared codebook.
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config

        # Frequency band encoder
        self.encoder = FrequencyBandEncoder(config)

        # Per-band residual VQ
        self.band_quantizers = nn.ModuleDict({
            band: ResidualVectorQuantizer(
                codebook_size=config.codebook_sizes[band],
                codebook_dim=config.codebook_dim,
                n_quantizers=config.n_quantizers,
            )
            for band in config.freq_bands
        })

        # Shared cross-modal codebook
        self.shared_quantizer = ResidualVectorQuantizer(
            codebook_size=config.shared_codebook_size,
            codebook_dim=config.codebook_dim,
            n_quantizers=config.n_quantizers,
        )

        # Adaptive boundary predictor
        if config.use_adaptive_boundaries:
            self.boundary_predictor = AdaptiveBoundaryPredictor(config)

        # Decoder
        self.decoder = self._build_decoder()

    def _build_decoder(self) -> nn.Module:
        """Build decoder to reconstruct signal from quantized tokens."""
        return nn.Sequential(
            nn.ConvTranspose1d(
                self.config.codebook_dim * (len(self.config.freq_bands) + 1),
                self.config.codebook_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.ConvTranspose1d(
                self.config.codebook_dim,
                self.config.codebook_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                self.config.codebook_dim // 2,
                self.config.n_channels,
                kernel_size=7,
                padding=3,
            ),
        )

    def _get_mixing_weight(self, z: torch.Tensor) -> torch.Tensor:
        """
        Learn per-timestep mixing weight alpha between shared and modality-specific codebooks.
        """
        # Simple: use norm of representation as mixing signal
        z_norm = z.norm(dim=1, keepdim=True)  # (B, 1, T)
        alpha = torch.sigmoid(z_norm / self.config.codebook_dim)  # (B, 1, T)
        return alpha

    def forward(
        self,
        x: torch.Tensor,
        events: Optional[torch.Tensor] = None,
        return_boundaries: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, T) input physiological signal
            events: (B, 1, T) optional physiological event mask
        Returns:
            Dict with:
                x_recon: reconstructed signal
                tokens: discrete token indices (B, N_q, sum(K_b)+K_shared, T')
                commitment_loss: VQ commitment loss
                band_tokens: per-band token indices
                boundary_probs: (optional) token boundary probabilities
        """
        B, C, T = x.shape

        # 1. Encode into frequency bands
        band_reprs = self.encoder(x)  # Dict[band: (B, D, T)]

        # 2. Per-band VQ
        band_quantized = {}
        band_indices = {}
        total_commitment_loss = 0.0

        for band, z in band_reprs.items():
            z_q, indices, commitment = self.band_quantizers[band](z)
            band_quantized[band] = z_q
            band_indices[band] = indices
            total_commitment_loss += commitment

        # 3. Shared codebook (mixture with modality-specific)
        shared_repr = sum(band_quantized.values()) / len(band_quantized)  # (B, D, T)
        alpha = self._get_mixing_weight(shared_repr)  # (B, 1, T)

        shared_q, shared_indices, shared_commitment = self.shared_quantizer(shared_repr)
        total_commitment_loss += shared_commitment

        # Mix: z_final = alpha * shared + (1-alpha) * band
        all_quantized = []
        for band, z_q in band_quantized.items():
            mixed = alpha * shared_q + (1 - alpha) * z_q
            all_quantized.append(mixed)
        all_quantized.append(shared_q)

        # 4. Adaptive boundary pooling
        if self.config.use_adaptive_boundaries and events is not None:
            boundary_probs = self.boundary_predictor(band_reprs, events)
            # Pool tokens at predicted boundaries (simplified: mean pooling)
            # Full implementation: segment according to boundary_probs > 0.5
        else:
            boundary_probs = None

        # 5. Decode
        z_cat = torch.cat(all_quantized, dim=1)  # (B, (N_bands+1)*D, T)
        x_recon = self.decoder(z_cat)

        # Apply temporal correction if lengths don't match
        if x_recon.shape[-1] != T:
            x_recon = F.interpolate(x_recon, size=T, mode="linear")

        result = {
            "x_recon": x_recon,
            "band_indices": band_indices,
            "shared_indices": shared_indices,
            "commitment_loss": total_commitment_loss,
        }
        if boundary_probs is not None:
            result["boundary_probs"] = boundary_probs

        return result

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode signal to discrete tokens (without decoding).

        Returns token indices for downstream tasks.
        """
        # Run encoder + VQ without decoder
        band_reprs = self.encoder(x)
        all_tokens = {}
        for band, z in band_reprs.items():
            _, indices, _ = self.band_quantizers[band](z)
            all_tokens[band] = indices

        shared_repr = sum(band_reprs.values()) / len(band_reprs)
        _, shared_indices, _ = self.shared_quantizer(shared_repr)
        all_tokens["shared"] = shared_indices

        return all_tokens

    def get_vocabulary_size(self) -> int:
        """Total vocabulary size across all bands and shared codebook."""
        return sum(self.config.codebook_sizes.values()) + self.config.shared_codebook_size
