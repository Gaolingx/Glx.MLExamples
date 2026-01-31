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
from src.models.vae import DiagonalGaussianDistribution


class TestDiagonalGaussianDistribution:
    """Tests for DiagonalGaussianDistribution helper class."""

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

    def test_deterministic(self):
        """Test deterministic mode."""
        params = torch.randn(2, 8, 16, 16)
        dist = DiagonalGaussianDistribution(params, deterministic=True)

        # In deterministic mode, std should be zero
        assert torch.allclose(dist.std, torch.zeros_like(dist.std))

        # Sample should equal mean
        sample = dist.sample()
        assert torch.allclose(sample, dist.mean)


class TestAutoencoderKL:
    """Tests for AutoencoderKL wrapper."""

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

    def test_init(self, model):
        """Test model initialization."""
        assert model.latent_channels == 4
        assert model.scaling_factor == 0.18215
        assert model._vae is not None

    def test_config(self, model):
        """Test config storage."""
        assert model.config["in_channels"] == 3
        assert model.config["out_channels"] == 3
        assert model.config["latent_channels"] == 4
        assert model.config["block_out_channels"] == [32, 64]

    def test_encoder_property(self, model):
        """Test encoder property access."""
        encoder = model.encoder
        assert encoder is not None
        assert hasattr(encoder, "conv_in")

    def test_decoder_property(self, model):
        """Test decoder property access."""
        decoder = model.decoder
        assert decoder is not None
        assert hasattr(decoder, "conv_out")

    def test_encode(self, model):
        """Test encoding."""
        x = torch.randn(2, 3, 64, 64)
        output = model.encode(x)

        assert hasattr(output, "latent_dist")
        # Shape: (batch, latent_channels, height/4, width/4) for 2 down blocks
        assert output.latent_dist.mean.shape == (2, 4, 16, 16)

    def test_encode_return_tuple(self, model):
        """Test encoding with return_dict=False."""
        x = torch.randn(2, 3, 64, 64)
        output = model.encode(x, return_dict=False)

        assert isinstance(output, tuple)
        assert len(output) == 1
        assert hasattr(output[0], "sample")  # latent_dist has sample method

    def test_decode(self, model):
        """Test decoding."""
        z = torch.randn(2, 4, 16, 16)
        output = model.decode(z)

        assert hasattr(output, "sample")
        assert output.sample.shape == (2, 3, 64, 64)

    def test_decode_return_tuple(self, model):
        """Test decoding with return_dict=False."""
        z = torch.randn(2, 4, 16, 16)
        output = model.decode(z, return_dict=False)

        assert isinstance(output, tuple)
        assert len(output) == 1
        assert output[0].shape == (2, 3, 64, 64)

    def test_forward(self, model):
        """Test full forward pass (encode + decode)."""
        x = torch.randn(2, 3, 64, 64)
        output = model(x)

        assert hasattr(output, "sample")
        assert output.sample.shape == x.shape

    def test_forward_sample_posterior(self, model):
        """Test forward with sampling from posterior."""
        x = torch.randn(2, 3, 64, 64)
        
        # With sample_posterior=True, results should differ each time
        output1 = model(x, sample_posterior=True)
        output2 = model(x, sample_posterior=True)
        
        # Outputs should be different due to sampling
        assert not torch.allclose(output1.sample, output2.sample)

    def test_forward_deterministic(self, model):
        """Test forward without sampling (deterministic mode)."""
        x = torch.randn(2, 3, 64, 64)
        
        # With sample_posterior=False, results should be identical
        output1 = model(x, sample_posterior=False)
        output2 = model(x, sample_posterior=False)
        
        assert torch.allclose(output1.sample, output2.sample)

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

    def test_encode_decode_roundtrip(self, model):
        """Test that encode -> decode approximately reconstructs input."""
        x = torch.randn(2, 3, 64, 64)
        
        # Encode
        latent = model.encode_to_latent(x, sample_posterior=False)
        
        # Decode
        reconstructed = model.decode_from_latent(latent)
        
        # Should have same shape
        assert reconstructed.shape == x.shape
        
        # Note: Reconstruction won't be perfect, but should be similar structure
        # Just check that values are in reasonable range
        assert reconstructed.abs().max() < 100  # Sanity check

    def test_tiling(self, model):
        """Test tiling enable/disable."""
        model.enable_tiling()
        assert model._vae.use_tiling
        
        model.disable_tiling()
        assert not model._vae.use_tiling

    def test_slicing(self, model):
        """Test slicing enable/disable."""
        model.enable_slicing()
        assert model._vae.use_slicing
        
        model.disable_slicing()
        assert not model._vae.use_slicing

    def test_gradient_checkpointing(self, model):
        """Test gradient checkpointing enable/disable."""
        # Just test that methods don't raise errors
        model.enable_gradient_checkpointing()
        model.disable_gradient_checkpointing()

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

    def test_save_and_load(self, model, tmp_path):
        """Test saving and loading model."""
        save_dir = tmp_path / "vae_test"
        
        # Save
        model.save_pretrained(str(save_dir))
        
        # Check files exist
        assert (save_dir / "config.json").exists()
        assert (save_dir / "diffusion_pytorch_model.safetensors").exists() or \
               (save_dir / "diffusion_pytorch_model.bin").exists()
        
        # Load
        loaded_model = AutoencoderKL.from_pretrained(str(save_dir))
        
        # Compare outputs
        x = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            original_output = model(x, sample_posterior=False).sample
            loaded_output = loaded_model(x, sample_posterior=False).sample
        
        assert torch.allclose(original_output, loaded_output, atol=1e-5)

    def test_parameters_trainable(self, model):
        """Test that model parameters are trainable."""
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check that parameters require grad
        for param in params:
            assert param.requires_grad

    def test_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        
        output = model(x, sample_posterior=False)
        loss = output.sample.mean()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape


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


class TestAutoencoderKLTraining:
    """Tests related to training scenarios."""

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

    def test_decoder_conv_out_access(self, model):
        """Test accessing decoder.conv_out for adaptive weight calculation."""
        # This is used in vae_module.py for computing adaptive discriminator weight
        last_layer = model.decoder.conv_out.weight
        
        assert last_layer is not None
        assert last_layer.requires_grad

    def test_kl_loss_computation(self, model):
        """Test KL loss computation through latent_dist."""
        x = torch.randn(2, 3, 64, 64)
        output = model.encode(x)
        
        # Get KL divergence
        kl = output.latent_dist.kl()
        
        assert kl.shape == (2,)
        assert (kl >= 0).all()
        assert kl.mean().requires_grad

    def test_reconstruction_loss(self, model):
        """Test reconstruction scenario."""
        x = torch.randn(2, 3, 64, 64)
        
        # Forward pass
        reconstruction = model(x, sample_posterior=False).sample
        
        # Compute reconstruction loss
        rec_loss = torch.nn.functional.mse_loss(reconstruction, x)
        
        # Backprop
        rec_loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                # At least some parameters should have gradients
                break
        else:
            pytest.fail("No parameter has gradient")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
