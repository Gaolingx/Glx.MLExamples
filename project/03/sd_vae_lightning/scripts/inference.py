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

from diffusers import AutoencoderKL
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
        "--use_ema",
        action="store_true",
        help="Use EMA weights for inference (only works with --checkpoint_path)",
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
        "--width",
        type=int,
        default=None,
        help="Center-crop width (default: 512)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Center-crop height (default: 512)",
    )
    parser.add_argument(
        "--no_metrics",
        action="store_true",
        help="Skip printing reconstruction metrics (PSNR/SSIM/rFID/PSIM)",
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
    ckpt_exists = args.checkpoint_path is not None and Path(args.checkpoint_path).exists()

    if args.checkpoint_path and not ckpt_exists:
        print(f"Warning: checkpoint not found: {args.checkpoint_path}. Falling back to --model_path.")

    if ckpt_exists:
        # Load from Lightning checkpoint
        module = VAELightningModule.load_from_checkpoint(
            args.checkpoint_path,
            map_location=args.device,
        )
        model = module.vae

        if args.use_ema:
            ema = getattr(module, "ema", None)
            if ema is not None:
                ema.copy_to(model.parameters())
                print("Using EMA weights from checkpoint for inference.")
            else:
                print(
                    "Warning: `--use_ema` is ignored because EMA weights are not available in the checkpoint."
                )
    else:
        # Load from pretrained
        if args.use_ema:
            print(
                "Warning: `--use_ema` is ignored because `--checkpoint_path` is not provided."
            )
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


def load_image(path: str, width: int, height: int) -> torch.Tensor:
    """Load and preprocess image with center crop only (no resize)."""
    image = Image.open(path).convert("RGB")

    if image.width < width or image.height < height:
        raise ValueError(
            f"Input image is too small for requested crop: "
            f"image=({image.width}x{image.height}), crop=({width}x{height}). "
            f"This script does center-crop only and does not resize."
        )

    resolution = [height, width]
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Scale to [-1, 1]
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


def encode_to_latent(
        model: AutoencoderKL,
        x: torch.Tensor,
        sample_posterior: bool = True,
) -> torch.Tensor:
    """
    Encode image tensor to scaled latents.
    """
    posterior = model.encode(x).latent_dist
    z = posterior.sample() if sample_posterior else posterior.mode()
    scaling_factor = getattr(model.config, "scaling_factor", 0.18215)
    return z * scaling_factor


def decode_from_latent(
        model: AutoencoderKL,
        z: torch.Tensor,
) -> torch.Tensor:
    """
    Decode scaled latents to image tensor.
    """
    scaling_factor = getattr(model.config, "scaling_factor", 0.18215)
    z = z / scaling_factor
    return model.decode(z).sample


def _get_target_size(args) -> tuple[int, int]:
    """Resolve target crop size."""
    width = 512 if args.width is None else args.width
    height = 512 if args.height is None else args.height

    if width <= 0 or height <= 0:
        raise ValueError("--width and --height must be positive integers")

    return width, height


def compute_and_print_metrics(target: torch.Tensor, reconstruction: torch.Tensor) -> None:
    """Compute and print PSNR/SSIM/rFID/PSIM for reconstructed image."""
    try:
        from src.utils.metrics import PSNR, SSIM, rFID, PSIM
    except ImportError as e:
        print(f"Warning: metrics are unavailable ({e}).")
        return

    target = target.float()
    reconstruction = reconstruction.float()

    with torch.no_grad():
        psnr_metric = PSNR(data_range=2.0)
        ssim_metric = SSIM(data_range=2.0)
        rfid_metric = rFID(feature_dim=2048, reset_real_features=True)
        psim_metric = PSIM(net="vgg")

        psnr = psnr_metric(reconstruction, target)
        ssim = ssim_metric(reconstruction, target)
        rfid_metric.update(target, reconstruction)
        rfid = rfid_metric.compute()
        psim = psim_metric(reconstruction, target)

    print("Reconstruction metrics:")
    print(f"  PSNR: {float(psnr.item()):.6f}")
    print(f"  SSIM: {float(ssim.item()):.6f}")
    print(f"  rFID: {float(rfid.item()):.6f}")
    print(f"  PSIM: {float(psim.item()):.6f}")


def main():
    """Main inference function."""
    args = parse_args()

    # Validate arguments
    if args.encode and args.decode:
        raise ValueError("Cannot specify both --encode and --decode")

    target_width, target_height = _get_target_size(args)

    # Load model
    source = args.checkpoint_path if args.checkpoint_path else args.model_path
    print(f"Loading model from: {source}")
    model = load_model(args)

    if args.encode:
        # Encode image to latent
        print(f"Encoding image: {args.input}")
        image = load_image(args.input, target_width, target_height)
        image = image.to(args.device, dtype=model.encoder.conv_in.weight.dtype)

        with torch.no_grad():
            latent = encode_to_latent(model, image, sample_posterior=True)

        print(f"Latent shape: {latent.shape}")
        torch.save(latent.cpu(), args.output)
        print(f"Saved latent to: {args.output}")

    elif args.decode:
        # Decode latent to image
        print(f"Decoding latent: {args.input}")
        latent = torch.load(args.input, map_location=args.device)
        latent = latent.to(args.device, dtype=model.decoder.conv_in.weight.dtype)

        with torch.no_grad():
            image = decode_from_latent(model, latent)

        save_image(image, args.output)
        print(f"Saved image to: {args.output}")

    else:
        # Reconstruct image (encode + decode)
        print(f"Reconstructing image: {args.input}")
        image = load_image(args.input, target_width, target_height)
        image = image.to(args.device, dtype=model.encoder.conv_in.weight.dtype)

        with torch.no_grad():
            latent = encode_to_latent(model, image, sample_posterior=True)
            print(f"Latent shape: {latent.shape}")
            reconstruction = decode_from_latent(model, latent)

        save_image(reconstruction, args.output)
        print(f"Saved reconstruction to: {args.output}")

        if not args.no_metrics:
            compute_and_print_metrics(image, reconstruction)


if __name__ == "__main__":
    main()
