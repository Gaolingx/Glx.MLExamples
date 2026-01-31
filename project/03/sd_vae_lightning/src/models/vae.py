"""
Core VAE components and utilities.
This module provides helper classes for VAE operations.
The main AutoencoderKL uses diffusers backend for compatibility.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class EncoderOutput:
    """
    Output of encoding method.

    Args:
        latent: The encoded latent tensor of shape (batch_size, num_channels, latent_height, latent_width).
    """
    latent: torch.Tensor


@dataclass
class DecoderOutput:
    """
    Output of decoding method.

    Args:
        sample: The decoded output sample from the last layer of the model.
    """
    sample: torch.Tensor


class DiagonalGaussianDistribution:
    """
    Diagonal Gaussian distribution for VAE latent space.
    Supports sampling with reparameterization trick and KL divergence computation.
    
    This class provides the same interface as diffusers' DiagonalGaussianDistribution
    for compatibility when not using the diffusers backend.

    Args:
        parameters: Tensor containing concatenated mean and log variance.
        deterministic: If True, use mode instead of sampling.
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean,
                device=self.parameters.device,
                dtype=self.parameters.dtype,
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Sample from the distribution using reparameterization trick.

        Args:
            generator: Optional random generator for reproducibility.

        Returns:
            Sampled latent tensor.
        """
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * noise

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> torch.Tensor:
        """
        Compute KL divergence against a prior (standard normal if other is None).

        Args:
            other: Optional other distribution for KL divergence.

        Returns:
            KL divergence tensor.
        """
        if self.deterministic:
            return torch.tensor([0.0], device=self.parameters.device)

        if other is None:
            # KL divergence against standard normal N(0, 1)
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        else:
            # KL divergence against other Gaussian
            return 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=[1, 2, 3],
            )

    def nll(
        self,
        sample: torch.Tensor,
        dims: Tuple[int, ...] = (1, 2, 3),
    ) -> torch.Tensor:
        """
        Compute negative log likelihood of a sample.

        Args:
            sample: Sample tensor.
            dims: Dimensions to sum over.

        Returns:
            Negative log likelihood tensor.
        """
        if self.deterministic:
            return torch.tensor([0.0], device=self.parameters.device)

        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        """Return the mode (mean) of the distribution."""
        return self.mean
