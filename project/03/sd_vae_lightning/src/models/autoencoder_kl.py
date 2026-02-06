"""
AutoencoderKL - Variational Autoencoder with KL divergence loss.
Compatible with Stable Diffusion 1.5 weights using diffusers backend.
"""

from typing import Optional, Tuple, Union, Dict, Any
import json
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DecoderOutput:
    """Output of decoding method."""
    sample: torch.Tensor


@dataclass
class AutoencoderKLOutput:
    """Output of AutoencoderKL encode method."""
    latent_dist: Any  # DiagonalGaussianDistribution


class AutoencoderKL(nn.Module):
    """
    Variational Autoencoder (VAE) with KL divergence loss.

    This is a wrapper around diffusers AutoencoderKL for compatibility
    with Stable Diffusion 1.5 weights while providing a clean interface.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output image channels.
        down_block_types: Tuple of downsampling block types.
        up_block_types: Tuple of upsampling block types.
        block_out_channels: Tuple of output channels for each block.
        layers_per_block: Number of layers per block.
        act_fn: Activation function name.
        latent_channels: Number of latent channels.
        norm_num_groups: Number of groups for GroupNorm.
        sample_size: Sample size for the model.
        scaling_factor: Scaling factor for latent space.
        use_quant_conv: Whether to use quantization convolution.
        use_post_quant_conv: Whether to use post-quantization convolution.
        mid_block_add_attention: Whether to add attention in mid blocks.
    """

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
            up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
            block_out_channels: Tuple[int, ...] = (64,),
            layers_per_block: int = 1,
            act_fn: str = "silu",
            latent_channels: int = 4,
            norm_num_groups: int = 32,
            sample_size: int = 512,
            scaling_factor: float = 0.18215,
            use_quant_conv: bool = True,
            use_post_quant_conv: bool = True,
            mid_block_add_attention: bool = True,
    ):
        super().__init__()

        # Store config for saving/loading
        self.config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "down_block_types": list(down_block_types),
            "up_block_types": list(up_block_types),
            "block_out_channels": list(block_out_channels),
            "layers_per_block": layers_per_block,
            "act_fn": act_fn,
            "latent_channels": latent_channels,
            "norm_num_groups": norm_num_groups,
            "sample_size": sample_size,
            "scaling_factor": scaling_factor,
            "use_quant_conv": use_quant_conv,
            "use_post_quant_conv": use_post_quant_conv,
            "mid_block_add_attention": mid_block_add_attention,
        }

        self.scaling_factor = scaling_factor
        self.latent_channels = latent_channels

        # Use diffusers AutoencoderKL as backend
        from diffusers import AutoencoderKL as DiffusersAutoencoderKL

        self._vae = DiffusersAutoencoderKL(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
        )

    @property
    def encoder(self):
        """Access encoder for gradient computation."""
        return self._vae.encoder

    @property
    def decoder(self):
        """Access decoder for gradient computation."""
        return self._vae.decoder

    @property
    def quant_conv(self):
        """Access quant_conv."""
        return self._vae.quant_conv

    @property
    def post_quant_conv(self):
        """Access post_quant_conv."""
        return self._vae.post_quant_conv

    def encode(
            self,
            x: torch.Tensor,
            return_dict: bool = True,
    ) -> Union[AutoencoderKLOutput, Tuple]:
        """
        Encode images into latent distributions.

        Args:
            x: Input image tensor of shape (B, C, H, W).
            return_dict: Whether to return as AutoencoderKLOutput.

        Returns:
            Latent distribution or tuple containing distribution.
        """
        result = self._vae.encode(x, return_dict=True)
        if not return_dict:
            return (result.latent_dist,)
        return AutoencoderKLOutput(latent_dist=result.latent_dist)

    def decode(
            self,
            z: torch.Tensor,
            return_dict: bool = True,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        """
        Decode latent representations into images.

        Args:
            z: Latent tensor of shape (B, latent_channels, H, W).
            return_dict: Whether to return as DecoderOutput.

        Returns:
            Decoded image or tuple containing image.
        """
        result = self._vae.decode(z, return_dict=True)
        if not return_dict:
            return (result.sample,)
        return DecoderOutput(sample=result.sample)

    def forward(
            self,
            sample: torch.Tensor,
            sample_posterior: bool = False,
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Args:
            sample: Input image tensor.
            sample_posterior: Whether to sample from posterior or use mode.
            return_dict: Whether to return as DecoderOutput.
            generator: Optional random generator.

        Returns:
            Reconstructed image or DecoderOutput.
        """
        posterior = self.encode(sample).latent_dist

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def encode_to_latent(
            self,
            x: torch.Tensor,
            sample_posterior: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Encode image to scaled latent representation.

        Args:
            x: Input image tensor (expected range [-1, 1]).
            sample_posterior: Whether to sample from posterior.
            generator: Optional random generator.

        Returns:
            Scaled latent tensor.
        """
        posterior = self.encode(x).latent_dist

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        return z * self.scaling_factor

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from scaled latent to image.

        Args:
            z: Scaled latent tensor.

        Returns:
            Decoded image tensor (range [-1, 1]).
        """
        z = z / self.scaling_factor
        return self.decode(z).sample

    def enable_tiling(self):
        """Enable tiled VAE processing for memory efficiency."""
        self._vae.enable_tiling()

    def disable_tiling(self):
        """Disable tiled VAE processing."""
        self._vae.disable_tiling()

    def enable_slicing(self):
        """Enable sliced VAE processing for memory efficiency."""
        self._vae.enable_slicing()

    def disable_slicing(self):
        """Disable sliced VAE processing."""
        self._vae.disable_slicing()

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._vae.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self._vae.disable_gradient_checkpointing()

    def save_pretrained(self, save_directory: str):
        """
        Save model weights and config to directory.

        Args:
            save_directory: Path to save directory.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Use diffusers save method
        self._vae.save_pretrained(str(save_path))

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            subfolder: Optional[str] = None,
            torch_dtype: Optional[torch.dtype] = None,
            **kwargs,
    ) -> "AutoencoderKL":
        """
        Load pretrained model from path or HuggingFace Hub.

        Args:
            pretrained_model_name_or_path: Path or HuggingFace model ID.
            subfolder: Optional subfolder within model path.
            torch_dtype: Optional torch dtype.
            **kwargs: Additional config overrides.

        Returns:
            Loaded AutoencoderKL model.
        """
        from diffusers import AutoencoderKL as DiffusersAutoencoderKL

        # Load using diffusers
        diffusers_vae = DiffusersAutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        # Create wrapper
        config = diffusers_vae.config
        model = cls(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            down_block_types=tuple(config.down_block_types),
            up_block_types=tuple(config.up_block_types),
            block_out_channels=tuple(config.block_out_channels),
            layers_per_block=config.layers_per_block,
            act_fn=config.act_fn,
            latent_channels=config.latent_channels,
            norm_num_groups=config.norm_num_groups,
            sample_size=config.sample_size,
            scaling_factor=config.scaling_factor,
        )

        # Replace internal VAE with loaded one
        model._vae = diffusers_vae

        return model

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AutoencoderKL":
        """
        Create model from config dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            AutoencoderKL model.
        """
        return cls(**config)

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load config from path.

        Args:
            config_path: Path to config file or HuggingFace model ID.

        Returns:
            Configuration dictionary.
        """
        from diffusers import AutoencoderKL as DiffusersAutoencoderKL
        config = DiffusersAutoencoderKL.load_config(config_path)
        return dict(config)
