"""
AutoencoderKL - Variational Autoencoder with KL divergence loss.
Compatible with Stable Diffusion 1.5 weights.
"""

from typing import Optional, Tuple, Union, Dict, Any
import json
from pathlib import Path

import torch
import torch.nn as nn

from .vae import (
    Encoder,
    Decoder,
    DiagonalGaussianDistribution,
    DecoderOutput,
)


class AutoencoderKLOutput:
    """Output of AutoencoderKL encode method."""

    def __init__(self, latent_dist: DiagonalGaussianDistribution):
        self.latent_dist = latent_dist


class AutoencoderKL(nn.Module):
    """
    Variational Autoencoder (VAE) with KL divergence loss for encoding images
    into latents and decoding latent representations into images.

    This implementation is compatible with Stable Diffusion 1.5 VAE weights.

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

        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

        # Quantization convolutions
        self.quant_conv = (
            nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            nn.Conv2d(latent_channels, latent_channels, 1)
            if use_post_quant_conv
            else None
        )

        # Tiling settings
        self.use_slicing = False
        self.use_tiling = False
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = int(
            sample_size / (2 ** (len(block_out_channels) - 1))
        )
        self.tile_overlap_factor = 0.25

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode images into latent distributions.

        Args:
            x: Input image tensor of shape (B, C, H, W).
            return_dict: Whether to return as AutoencoderKLOutput.

        Returns:
            Latent distribution or tuple containing distribution.
        """
        if self.use_slicing and x.shape[0] > 1:
            # Process images one by one to save memory
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Internal encoding function."""
        batch_size, num_channels, height, width = x.shape

        if self.use_tiling and (
            width > self.tile_sample_min_size or height > self.tile_sample_min_size
        ):
            return self._tiled_encode(x)

        enc = self.encoder(x)
        if self.quant_conv is not None:
            enc = self.quant_conv(enc)

        return enc

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
        if self.use_slicing and z.shape[0] > 1:
            # Process latents one by one to save memory
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def _decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """Internal decoding function."""
        if self.use_tiling and (
            z.shape[-1] > self.tile_latent_min_size
            or z.shape[-2] > self.tile_latent_min_size
        ):
            return self._tiled_decode(z, return_dict=return_dict)

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

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

    def _tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using tiled approach for large images."""
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[
                    :,
                    :,
                    i : i + self.tile_sample_min_size,
                    j : j + self.tile_sample_min_size,
                ]
                tile = self.encoder(tile)
                if self.quant_conv is not None:
                    tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        return torch.cat(result_rows, dim=2)

    def _tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        """Decode using tiled approach for large latents."""
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                if self.post_quant_conv is not None:
                    tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def _blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        """Blend two tensors vertically."""
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (
                1 - y / blend_extent
            ) + b[:, :, y, :] * (y / blend_extent)
        return b

    def _blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        """Blend two tensors horizontally."""
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (
                1 - x / blend_extent
            ) + b[:, :, :, x] * (x / blend_extent)
        return b

    def enable_tiling(self):
        """Enable tiled VAE processing for memory efficiency."""
        self.use_tiling = True

    def disable_tiling(self):
        """Disable tiled VAE processing."""
        self.use_tiling = False

    def enable_slicing(self):
        """Enable sliced VAE processing for memory efficiency."""
        self.use_slicing = True

    def disable_slicing(self):
        """Disable sliced VAE processing."""
        self.use_slicing = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.encoder.gradient_checkpointing = True
        self.decoder.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.encoder.gradient_checkpointing = False
        self.decoder.gradient_checkpointing = False

    def save_pretrained(self, save_directory: str):
        """
        Save model weights and config to directory.

        Args:
            save_directory: Path to save directory.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = save_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        # Save weights
        weights_path = save_path / "diffusion_pytorch_model.safetensors"
        try:
            from safetensors.torch import save_file
            save_file(self.state_dict(), str(weights_path))
        except ImportError:
            weights_path = save_path / "diffusion_pytorch_model.bin"
            torch.save(self.state_dict(), weights_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = None,
        **kwargs,
    ) -> "AutoencoderKL":
        """
        Load pretrained model from path or HuggingFace Hub.

        Args:
            pretrained_model_name_or_path: Path or HuggingFace model ID.
            subfolder: Optional subfolder within model path.
            **kwargs: Additional config overrides.

        Returns:
            Loaded AutoencoderKL model.
        """
        # Try loading from diffusers
        try:
            from diffusers import AutoencoderKL as DiffusersAutoencoder

            diffusers_vae = DiffusersAutoencoder.from_pretrained(
                pretrained_model_name_or_path,
                subfolder=subfolder,
            )

            # Create our model with same config
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
                use_quant_conv=getattr(config, "use_quant_conv", True),
                use_post_quant_conv=getattr(config, "use_post_quant_conv", True),
                mid_block_add_attention=getattr(
                    config, "mid_block_add_attention", True
                ),
            )

            # Load weights
            model.load_state_dict(diffusers_vae.state_dict())
            return model

        except Exception:
            # Fall back to loading from local path
            load_path = Path(pretrained_model_name_or_path)
            if subfolder:
                load_path = load_path / subfolder

            # Load config
            config_path = load_path / "config.json"
            with open(config_path, "r") as f:
                config = json.load(f)

            config.update(kwargs)
            model = cls(**config)

            # Load weights
            try:
                from safetensors.torch import load_file

                weights_path = load_path / "diffusion_pytorch_model.safetensors"
                state_dict = load_file(str(weights_path))
            except (ImportError, FileNotFoundError):
                weights_path = load_path / "diffusion_pytorch_model.bin"
                state_dict = torch.load(weights_path, map_location="cpu")

            model.load_state_dict(state_dict)
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
        try:
            from diffusers import AutoencoderKL as DiffusersAutoencoder

            config = DiffusersAutoencoder.load_config(config_path)
            return dict(config)
        except Exception:
            with open(config_path, "r") as f:
                return json.load(f)
