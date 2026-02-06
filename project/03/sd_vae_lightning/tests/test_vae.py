"""
Unit tests for VAE implementation.
Tests the AutoencoderKL wrapper around diffusers backend.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.autoencoder_kl import AutoencoderKL


@pytest.fixture(scope="module")
def vae_model():
    """Create a small VAE model for testing (shared across tests in module)."""
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


class TestLatentDistribution:
    """Tests for the latent distribution returned by AutoencoderKL.encode()."""

    @pytest.fixture
    def distribution(self, vae_model):
        """Get a distribution from the model's encoder."""
        x = torch.randn(2, 3, 64, 64)
        return vae_model.encode(x).latent_dist

    def test_sample_shape(self, distribution):
        """Test sampling from distribution."""
        sample = distribution.sample()
        assert sample.shape == (2, 4, 16, 16)

    def test_kl_divergence(self, distribution):
        """Test KL divergence computation."""
        kl = distribution.kl()
        assert kl.shape == (2,)
        assert (kl >= 0).all()

    def test_mode(self, distribution):
        """Test mode computation."""
        mode = distribution.mode()
        assert mode.shape == (2, 4, 16, 16)
        assert torch.allclose(mode, distribution.mean)

    def test_sample_stochasticity(self, distribution):
        """Test that samples are stochastic."""
        sample1 = distribution.sample()
        sample2 = distribution.sample()
        assert not torch.allclose(sample1, sample2)

    def test_distribution_properties(self, distribution):
        """Test that distribution has expected properties."""
        assert hasattr(distribution, 'mean')
        assert hasattr(distribution, 'std')
        assert hasattr(distribution, 'var')
        assert distribution.mean.shape == (2, 4, 16, 16)
        assert (distribution.var > 0).all()


class TestAutoencoderKL:
    """Tests for AutoencoderKL wrapper."""

    def test_init(self, vae_model):
        """Test model initialization."""
        assert vae_model.latent_channels == 4
        assert vae_model.scaling_factor == 0.18215
        assert vae_model._vae is not None

    def test_config(self, vae_model):
        """Test config storage."""
        assert vae_model.config["in_channels"] == 3
        assert vae_model.config["out_channels"] == 3
        assert vae_model.config["latent_channels"] == 4
        assert vae_model.config["block_out_channels"] == [32, 64]

    def test_encoder_property(self, vae_model):
        """Test encoder property access."""
        encoder = vae_model.encoder
        assert encoder is not None
        assert hasattr(encoder, "conv_in")

    def test_decoder_property(self, vae_model):
        """Test decoder property access."""
        decoder = vae_model.decoder
        assert decoder is not None
        assert hasattr(decoder, "conv_out")

    def test_encode(self, vae_model):
        """Test encoding."""
        x = torch.randn(2, 3, 64, 64)
        output = vae_model.encode(x)

        assert hasattr(output, "latent_dist")
        assert output.latent_dist.mean.shape == (2, 4, 16, 16)

    def test_encode_return_tuple(self, vae_model):
        """Test encoding with return_dict=False."""
        x = torch.randn(2, 3, 64, 64)
        output = vae_model.encode(x, return_dict=False)

        assert isinstance(output, tuple)
        assert len(output) == 1
        assert hasattr(output[0], "sample")

    def test_decode(self, vae_model):
        """Test decoding."""
        z = torch.randn(2, 4, 16, 16)
        output = vae_model.decode(z)

        assert hasattr(output, "sample")
        assert output.sample.shape == (2, 3, 64, 64)

    def test_decode_return_tuple(self, vae_model):
        """Test decoding with return_dict=False."""
        z = torch.randn(2, 4, 16, 16)
        output = vae_model.decode(z, return_dict=False)

        assert isinstance(output, tuple)
        assert output[0].shape == (2, 3, 64, 64)

    def test_forward(self, vae_model):
        """Test full forward pass (encode + decode)."""
        x = torch.randn(2, 3, 64, 64)
        output = vae_model(x)

        assert hasattr(output, "sample")
        assert output.sample.shape == x.shape

    def test_forward_sample_posterior(self, vae_model):
        """Test forward with sampling from posterior."""
        x = torch.randn(2, 3, 64, 64)

        output1 = vae_model(x, sample_posterior=True)
        output2 = vae_model(x, sample_posterior=True)

        assert not torch.allclose(output1.sample, output2.sample)

    def test_forward_deterministic(self, vae_model):
        """Test forward without sampling (deterministic mode)."""
        x = torch.randn(2, 3, 64, 64)

        output1 = vae_model(x, sample_posterior=False)
        output2 = vae_model(x, sample_posterior=False)

        assert torch.allclose(output1.sample, output2.sample)

    def test_encode_to_latent(self, vae_model):
        """Test encode_to_latent method."""
        x = torch.randn(2, 3, 64, 64)
        latent = vae_model.encode_to_latent(x)

        assert latent.shape == (2, 4, 16, 16)

    def test_decode_from_latent(self, vae_model):
        """Test decode_from_latent method."""
        z = torch.randn(2, 4, 16, 16) * vae_model.scaling_factor
        image = vae_model.decode_from_latent(z)

        assert image.shape == (2, 3, 64, 64)

    def test_encode_decode_roundtrip(self, vae_model):
        """Test that encode -> decode approximately reconstructs input."""
        x = torch.randn(2, 3, 64, 64)

        latent = vae_model.encode_to_latent(x, sample_posterior=False)
        reconstructed = vae_model.decode_from_latent(latent)

        assert reconstructed.shape == x.shape
        assert reconstructed.abs().max() < 100

    def test_tiling(self, vae_model):
        """Test tiling enable/disable."""
        vae_model.enable_tiling()
        assert vae_model._vae.use_tiling

        vae_model.disable_tiling()
        assert not vae_model._vae.use_tiling

    def test_slicing(self, vae_model):
        """Test slicing enable/disable."""
        vae_model.enable_slicing()
        assert vae_model._vae.use_slicing

        vae_model.disable_slicing()
        assert not vae_model._vae.use_slicing

    def test_gradient_checkpointing(self, vae_model):
        """Test gradient checkpointing enable/disable."""
        vae_model.enable_gradient_checkpointing()
        vae_model.disable_gradient_checkpointing()

    def test_from_config(self):
        """Test creating model from config dict."""
        config = {
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ("DownEncoderBlock2D", "DownEncoderBlock2D"),
            "up_block_types": ("UpDecoderBlock2D", "UpDecoderBlock2D"),
            "block_out_channels": (32, 64),
            "layers_per_block": 1,
            "latent_channels": 4,
            "sample_size": 64,
        }

        model = AutoencoderKL.from_config(config)

        assert model.latent_channels == 4
        assert model.config["block_out_channels"] == [32, 64]

    def test_save_and_load(self, vae_model, tmp_path):
        """Test saving and loading model."""
        save_dir = tmp_path / "vae_test"

        vae_model.save_pretrained(str(save_dir))

        assert (save_dir / "config.json").exists()
        assert (save_dir / "diffusion_pytorch_model.safetensors").exists() or \
               (save_dir / "diffusion_pytorch_model.bin").exists()

        loaded_model = AutoencoderKL.from_pretrained(str(save_dir))

        x = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            original_output = vae_model(x, sample_posterior=False).sample
            loaded_output = loaded_model(x, sample_posterior=False).sample

        assert torch.allclose(original_output, loaded_output, atol=1e-5)

    def test_parameters_trainable(self, vae_model):
        """Test that model parameters are trainable."""
        params = list(vae_model.parameters())
        assert len(params) > 0

        for param in params:
            assert param.requires_grad

    def test_gradient_flow(self, vae_model):
        """Test that gradients flow through the model."""
        x = torch.randn(1, 3, 64, 64, requires_grad=True)

        output = vae_model(x, sample_posterior=False)
        loss = output.sample.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestAutoencoderKLTraining:
    """Tests related to training scenarios."""

    def test_decoder_conv_out_access(self, vae_model):
        """Test accessing decoder.conv_out for adaptive weight calculation."""
        last_layer = vae_model.decoder.conv_out.weight

        assert last_layer is not None
        assert last_layer.requires_grad

    def test_kl_loss_computation(self, vae_model):
        """Test KL loss computation through latent_dist."""
        x = torch.randn(2, 3, 64, 64)
        output = vae_model.encode(x)

        kl = output.latent_dist.kl()

        assert kl.shape == (2,)
        assert (kl >= 0).all()
        assert kl.mean().requires_grad

    def test_reconstruction_gradient_flow(self, vae_model):
        """Test gradient flow through reconstruction."""
        x = torch.randn(2, 3, 64, 64, requires_grad=True)

        reconstruction = vae_model(x, sample_posterior=False).sample
        rec_loss = torch.nn.functional.mse_loss(reconstruction, x)
        rec_loss.backward()

        # Verify gradients at key layers
        assert x.grad is not None
        assert vae_model.encoder.conv_in.weight.grad is not None
        assert vae_model.decoder.conv_out.weight.grad is not None
        assert vae_model.quant_conv.weight.grad is not None


class TestAutoencoderKLPretrained:
    """Tests for loading pretrained models. 
    These tests require network access and are marked as slow."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Pretrained model tests require GPU for reasonable speed"
    )
    def test_load_sd_vae(self):
        """Test loading Stable Diffusion VAE from HuggingFace Hub."""
        model = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16,
        )

        assert model is not None
        assert model.latent_channels == 4

        # Test inference
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        x = torch.randn(1, 3, 512, 512, dtype=torch.float16, device=device)

        with torch.no_grad():
            output = model(x, sample_posterior=False)

        assert output.sample.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
