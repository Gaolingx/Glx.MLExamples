#!/usr/bin/env python
"""
Inference script for AutoencoderKL.
Supports image encoding/decoding and reconstruction.

Usage:
    # Reconstruct an image
    python scripts/inference.py --input image.jpg --output reconstructed.jpg

    # Encode to latent
    python scripts/inference.py --input image.jpg --encode --output latent.pt

    # Decode from latent
    python scripts/inference.py --input latent.pt --decode --output decoded.jpg
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.models.autoencoder_kl import AutoencoderKL
from src.lightning.vae_module import VAELightningModule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VAE Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="Path to model or HuggingFace model ID",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to Lightning checkpoint (optional)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image path or latent tensor path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path",
    )
    parser.add_argument(
        "--encode",
        action="store_true",
        help="Encode image to latent",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="Decode latent to image",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype",
    )
    return parser.parse_args()


def load_model(args) -> AutoencoderKL:
    """Load VAE model."""
    if args.checkpoint_path:
        # Load from Lightning checkpoint
        module = VAELightningModule.load_from_checkpoint(args.checkpoint_path)
        model = module.vae
    else:
        # Load from pretrained
        model = AutoencoderKL.from_pretrained(args.model_path)

    # Set dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    model = model.to(args.device, dtype=dtype_map[args.dtype])
    model.eval()

    return model


def load_image(path: str, resolution: int) -> torch.Tensor:
    """Load and preprocess image."""
    image = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
    ])

    return transform(image).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save tensor as image."""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy
    tensor = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)

    # Save
    Image.fromarray(tensor).save(path)


def main():
    """Main inference function."""
    args = parse_args()

    # Validate arguments
    if args.encode and args.decode:
        raise ValueError("Cannot specify both --encode and --decode")

    # Load model
    print(f"Loading model from: {args.model_path or args.checkpoint_path}")
    model = load_model(args)

    if args.encode:
        # Encode image to latent
        print(f"Encoding image: {args.input}")
        image = load_image(args.input, args.resolution)
        image = image.to(args.device, dtype=model.encoder.conv_in.weight.dtype)

        with torch.no_grad():
            latent = model.encode_to_latent(image, sample_posterior=True)

        print(f"Latent shape: {latent.shape}")
        torch.save(latent.cpu(), args.output)
        print(f"Saved latent to: {args.output}")

    elif args.decode:
        # Decode latent to image
        print(f"Decoding latent: {args.input}")
        latent = torch.load(args.input, map_location=args.device)
        latent = latent.to(model.decoder.conv_in.weight.dtype)

        with torch.no_grad():
            image = model.decode_from_latent(latent)

        save_image(image, args.output)
        print(f"Saved image to: {args.output}")

    else:
        # Reconstruct image (encode + decode)
        print(f"Reconstructing image: {args.input}")
        image = load_image(args.input, args.resolution)
        image = image.to(args.device, dtype=model.encoder.conv_in.weight.dtype)

        with torch.no_grad():
            # Encode
            latent = model.encode_to_latent(image, sample_posterior=True)
            print(f"Latent shape: {latent.shape}")

            # Decode
            reconstruction = model.decode_from_latent(latent)

        save_image(reconstruction, args.output)
        print(f"Saved reconstruction to: {args.output}")

        # Compute metrics
        from src.utils.metrics import VAEMetrics
        metrics = VAEMetrics(data_range=2.0, use_lpips=True)
        image = image.cpu()
        reconstruction = reconstruction.cpu()
        psnr, ssim, lpips_val = metrics.compute_reconstruction_metrics(
            reconstruction, image
        )
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim:.4f}")
        if lpips_val is not None:
            print(f"LPIPS: {lpips_val:.4f}")


if __name__ == "__main__":
    main()
