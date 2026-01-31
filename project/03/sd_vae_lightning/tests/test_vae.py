"""
Unit tests for VAE implementation.
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vae import (
    Encoder,
    Decoder,
    DiagonalGaussianDistribution,
    ResnetBlock2D,
    AttentionBlock,
    DownEncoderBlock2D,
    UpDecoderBlock2D,
)
from src.models.autoencoder_kl import AutoencoderKL


class TestResnetBlock2D:
    """Tests for ResnetBlock2D."""

    def test_same_channels(self):
        """Test with same input/output channels."""
        block = ResnetBlock2D(in_channels=64, out_channels=64)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_different_channels(self):
        """Test with different input/output channels."""
        block = ResnetBlock2D(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 32, 32)


class TestAttentionBlock:
    """Tests for AttentionBlock."""

    def test_forward(self):
        """Test attention forward pass."""
        block = AttentionBlock(channels=64, num_head_channels=32)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == x.shape


class TestDownEncoderBlock2D:
    """Tests for DownEncoderBlock2D."""

    def test_with_downsample(self):
        """Test with downsampling."""
        block = DownEncoderBlock2D(
            in_channels=64,
            out_channels=128,
            num_layers=2,
            add_downsample=True,
        )
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 16, 16)

    def test_without_downsample(self):
        """Test without downsampling."""
        block = DownEncoderBlock2D(
            in_channels=64,
            out_channels=128,
            num_layers=2,
            add_downsample=False,
        )
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 32, 32)


class TestUpDecoderBlock2D:
    """Tests for UpDecoderBlock2D."""

    def test_with_upsample(self):
        """Test with upsampling."""
        block = UpDecoderBlock2D(
            in_channels=128,
            out_channels=64,
            num_layers=2,
            add_upsample=True,
        )
        x = torch.randn(2, 128, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_without_upsample(self):
        """Test without upsampling."""
        block = UpDecoderBlock2D(
            in_channels=128,
            out_channels=64,
            num_layers=2,
            add_upsample=False,
        )
        x = torch.randn(2, 128, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)


class TestEncoder:
    """Tests for Encoder."""

    def test_forward(self):
        """Test encoder forward pass."""
        encoder = Encoder(
            in_channels=3,
            out_channels=4,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            block_out_channels=(64, 128),
            layers_per_block=2,
            double_z=True,
        )
        x = torch.randn(2, 3, 64, 64)
        out = encoder(x)
        # Output: 2*4 channels (mean + logvar), 64/(2^2) = 16 spatial
        assert out.shape == (2, 8, 16, 16)


class TestDecoder:
    """Tests for Decoder."""

    def test_forward(self):
        """Test decoder forward pass."""
        decoder = Decoder(
            in_channels=4,
            out_channels=3,
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(64, 128),
            layers_per_block=2,
        )
        z = torch.randn(2, 4, 16, 16)
        out = decoder(z)
        # Output: 3 channels, 16 * (2^2) = 64 spatial
        assert out.shape == (2, 3, 64, 64)


class TestDiagonalGaussianDistribution:
    """Tests for DiagonalGaussianDistribution."""

    def test_sample(self):
        """Test sampling from distribution."""
        params = torch.randn(2, 8, 16, 16)  # 4 channels mean + 4 channels logvar
        dist = DiagonalGaussianDistribution(params)

        sample = dist.sample()
        assert sample.shape == (2, 4, 16, 16)

    def test_kl(self):
        """Test KL divergence computation."""
        params = torch.randn(2, 8, 16, 16)
        dist = DiagonalGaussianDistribution(params)

        kl = dist.kl()
        assert kl.shape == (2,)
        assert (kl >= 0).all()

    def test_mode(self):
        """Test mode computation."""
        params = torch.randn(2, 8, 16, 16)
        dist = DiagonalGaussianDistribution(params)

        mode = dist.mode()
        assert mode.shape == (2, 4, 16, 16)
        assert torch.allclose(mode, dist.mean)


class TestAutoencoderKL:
    """Tests for AutoencoderKL."""

    @pytest.fixture
    def model(self):
        """Create a small VAE model for testing."""
        return AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(32, 64),
            layers_per_block=1,
            latent_channels=4,
            sample_size=64,
        )

    def test_encode(self, model):
        """Test encoding."""
        x = torch.randn(2, 3, 64, 64)
        output = model.encode(x)

        assert hasattr(output, "latent_dist")
        assert output.latent_dist.mean.shape == (2, 4, 16, 16)

    def test_decode(self, model):
        """Test decoding."""
        z = torch.randn(2, 4, 16, 16)
        output = model.decode(z)

        assert output.sample.shape == (2, 3, 64, 64)

    def test_forward(self, model):
        """Test full forward pass."""
        x = torch.randn(2, 3, 64, 64)
        output = model(x)

        assert output.sample.shape == x.shape

    def test_encode_to_latent(self, model):
        """Test encode_to_latent method."""
        x = torch.randn(2, 3, 64, 64)
        latent = model.encode_to_latent(x)

        # Should be scaled by scaling_factor
        assert latent.shape == (2, 4, 16, 16)

    def test_decode_from_latent(self, model):
        """Test decode_from_latent method."""
        z = torch.randn(2, 4, 16, 16) * model.scaling_factor
        image = model.decode_from_latent(z)

        assert image.shape == (2, 3, 64, 64)

    def test_tiling(self, model):
        """Test tiled encoding/decoding."""
        model.enable_tiling()
        model.tile_sample_min_size = 32
        model.tile_latent_min_size = 8

        x = torch.randn(1, 3, 64, 64)
        output = model(x)

        assert output.sample.shape == x.shape
        model.disable_tiling()

    def test_slicing(self, model):
        """Test sliced encoding/decoding."""
        model.enable_slicing()

        x = torch.randn(4, 3, 64, 64)
        output = model(x)

        assert output.sample.shape == x.shape
        model.disable_slicing()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
