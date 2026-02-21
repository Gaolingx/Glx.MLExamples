#!/usr/bin/env python
"""
Inference script for AutoencoderKL.
Supports image encoding/decoding and reconstruction.

Usage:
    # Reconstruct an image
    python scripts/inference.py --input image.jpg --output reconstructed.jpg

    # Reconstruct multiple images (batch)
    python scripts/inference.py --input images_dir/ --output output_dir/ --batch_size 4

    # Encode to latent
    python scripts/inference.py --input image.jpg --encode --output latent.pt

    # Decode from latent (batch)
    python scripts/inference.py --input latents_dir/ --decode --output decoded_dir/

    # Custom resolution (non-square)
    python scripts/inference.py --input image.jpg --output out.jpg --width 768 --height 512

    # Use EMA weights from a Lightning checkpoint for inference
    python scripts/inference.py --checkpoint_path path/to/model.ckpt --use_ema \
        --input image.jpg --output out.jpg
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Iterator

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers import AutoencoderKL

from src.lightning.vae_module import VAELightningModule

# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
LATENT_EXTENSIONS = {".pt", ".pth"}


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
        help="Use EMA weights for inference when loading from --checkpoint_path",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image/latent path or directory for batch processing",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path or directory",
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
    # Resolution options
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution (square, used if --width/--height not set)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Image width (must be divisible by 8)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Image height (must be divisible by 8)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing multiple inputs",
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
    parser.add_argument(
        "--no_metrics",
        action="store_true",
        help="Skip metrics computation for reconstruction",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-file output, show only progress bar",
    )
    return parser.parse_args()


def get_resolution(args) -> Tuple[int, int]:
    """
    Get target resolution (width, height) from arguments.
    Ensures dimensions are divisible by 8 (VAE requirement).

    Returns:
        Tuple of (width, height).
    """
    if args.width is not None and args.height is not None:
        width, height = args.width, args.height
    elif args.width is not None:
        width, height = args.width, args.width
    elif args.height is not None:
        width, height = args.height, args.height
    else:
        width, height = args.resolution, args.resolution

    # Ensure divisible by 8
    orig_width, orig_height = width, height
    width = (width // 8) * 8
    height = (height // 8) * 8

    if width != orig_width or height != orig_height:
        print(f"Adjusted resolution: {orig_width}x{orig_height} -> {width}x{height} (must be divisible by 8)")

    return width, height


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


def load_model(args) -> AutoencoderKL:
    """Load VAE model."""
    if args.checkpoint_path:
        # Load from Lightning checkpoint
        module = VAELightningModule.load_from_checkpoint(
            args.checkpoint_path,
            map_location="cpu",
        )
        if args.use_ema:
            if getattr(module, "ema", None) is None:
                raise ValueError(
                    "`--use_ema` is set, but EMA weights were not found in checkpoint. "
                    "Please ensure this checkpoint was trained/saved with EMA enabled."
                )
            module.ema.copy_to(module.vae.parameters())
            print("Using EMA weights from checkpoint for inference.")
        model = module.vae
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
    """Load and preprocess image with custom resolution."""
    image = Image.open(path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
    ])

    return transform(image).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save tensor as image."""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)

    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Convert to numpy
    tensor = tensor.cpu().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)

    # Ensure output directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Save
    Image.fromarray(tensor).save(path)


def get_input_paths(input_path: str, mode: str = "image") -> List[Path]:
    """
    Get list of input paths from file or directory.

    Args:
        input_path: Single file or directory.
        mode: "image" or "latent" to filter extensions.

    Returns:
        Sorted list of Path objects.
    """
    path = Path(input_path)

    if path.is_file():
        return [path]
    elif path.is_dir():
        extensions = IMAGE_EXTENSIONS if mode == "image" else LATENT_EXTENSIONS
        files = [f for f in path.iterdir() if f.suffix.lower() in extensions]
        return sorted(files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def get_output_path(input_path: Path, output_arg: str, new_suffix: Optional[str] = None) -> Path:
    """
    Determine output path based on input and output arguments.

    Args:
        input_path: Original input path.
        output_arg: Output argument (file or directory).
        new_suffix: New file extension (e.g., ".pt", ".png").

    Returns:
        Output Path object.
    """
    output = Path(output_arg)

    # If output has a file extension, treat as file
    if output.suffix and output.suffix.lower() in IMAGE_EXTENSIONS | LATENT_EXTENSIONS:
        return output

    # Treat as directory
    output.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    suffix = new_suffix if new_suffix else input_path.suffix
    return output / f"{stem}{suffix}"


def batch_iterator(items: List, batch_size: int) -> Iterator[List]:
    """Yield successive batches from items list."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def encode_images(
        model: AutoencoderKL,
        input_paths: List[Path],
        output_arg: str,
        width: int,
        height: int,
        batch_size: int,
        device: str,
        quiet: bool = False,
) -> None:
    """Encode images to latents with batch support."""
    model_dtype = model.encoder.conv_in.weight.dtype

    num_batches = (len(input_paths) + batch_size - 1) // batch_size
    pbar = tqdm(
        batch_iterator(input_paths, batch_size),
        total=num_batches,
        desc="Encoding",
        unit="batch",
    )

    for batch_paths in pbar:
        # Update progress bar with current batch info
        pbar.set_postfix({"files": f"{batch_paths[0].name}..."})

        # Load batch
        images = [load_image(str(p), width, height) for p in batch_paths]
        batch_tensor = torch.cat(images, dim=0).to(device, dtype=model_dtype)

        with torch.no_grad():
            latents = encode_to_latent(model, batch_tensor, sample_posterior=True)

        # Save each latent
        for i, path in enumerate(batch_paths):
            output_path = get_output_path(path, output_arg, ".pt")
            torch.save(latents[i:i + 1].cpu(), output_path)
            if not quiet:
                tqdm.write(f"  {path.name} -> {output_path.name} (shape: {tuple(latents[i].shape)})")


def decode_latents(
        model: AutoencoderKL,
        input_paths: List[Path],
        output_arg: str,
        batch_size: int,
        device: str,
        quiet: bool = False,
) -> None:
    """Decode latents to images with batch support."""
    model_dtype = model.decoder.conv_in.weight.dtype

    num_batches = (len(input_paths) + batch_size - 1) // batch_size
    pbar = tqdm(
        batch_iterator(input_paths, batch_size),
        total=num_batches,
        desc="Decoding",
        unit="batch",
    )

    for batch_paths in pbar:
        # Update progress bar with current batch info
        pbar.set_postfix({"files": f"{batch_paths[0].name}..."})

        # Load batch of latents
        latents = [torch.load(p, map_location=device) for p in batch_paths]
        batch_latent = torch.cat(latents, dim=0).to(dtype=model_dtype)

        with torch.no_grad():
            images = decode_from_latent(model, batch_latent)

        # Save each image
        for i, path in enumerate(batch_paths):
            output_path = get_output_path(path, output_arg, ".png")
            save_image(images[i:i + 1], str(output_path))
            if not quiet:
                tqdm.write(f"  {path.name} -> {output_path.name}")


def reconstruct_images(
        model: AutoencoderKL,
        input_paths: List[Path],
        output_arg: str,
        width: int,
        height: int,
        batch_size: int,
        device: str,
        compute_metrics: bool = True,
        quiet: bool = False,
) -> None:
    """Reconstruct images with batch support and optional metrics."""
    model_dtype = model.encoder.conv_in.weight.dtype

    # Lazy load metrics
    psnr_metric = None
    ssim_metric = None
    all_psnr, all_ssim = [], []
    if compute_metrics:
        from src.utils.metrics import PSNR, SSIM
        psnr_metric = PSNR(data_range=2.0)
        ssim_metric = SSIM(data_range=2.0)

    num_batches = (len(input_paths) + batch_size - 1) // batch_size
    pbar = tqdm(
        batch_iterator(input_paths, batch_size),
        total=num_batches,
        desc="Reconstructing",
        unit="batch",
    )

    latent_shape_logged = False

    for batch_paths in pbar:
        # Load batch
        images = [load_image(str(p), width, height) for p in batch_paths]
        batch_tensor = torch.cat(images, dim=0).to(device, dtype=model_dtype)

        with torch.no_grad():
            # Encode
            latents = encode_to_latent(model, batch_tensor, sample_posterior=True)

            # Decode
            reconstructions = decode_from_latent(model, latents)

        # Log latent shape once
        if not latent_shape_logged:
            tqdm.write(f"Latent shape: {tuple(latents.shape)}")
            latent_shape_logged = True

        # Save reconstructions and compute metrics
        batch_cpu = batch_tensor.cpu()
        recon_cpu = reconstructions.cpu()

        batch_psnr = []
        for i, path in enumerate(batch_paths):
            output_path = get_output_path(path, output_arg, ".png")
            save_image(recon_cpu[i:i + 1], str(output_path))

            if psnr_metric is not None and ssim_metric is not None:
                pred = recon_cpu[i:i + 1]
                target = batch_cpu[i:i + 1]

                psnr_metric._metric.reset()
                ssim_metric._metric.reset()

                psnr = float(psnr_metric(pred, target).item())
                ssim = float(ssim_metric(pred, target).item())

                all_psnr.append(psnr)
                all_ssim.append(ssim)
                batch_psnr.append(psnr)

                if not quiet:
                    tqdm.write(
                        f"  {path.name} -> {output_path.name} | PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}"
                    )

        # Update progress bar with batch metrics
        if batch_psnr:
            pbar.set_postfix({"avg_psnr": f"{np.mean(batch_psnr):.2f} dB"})

    # Print summary metrics
    if psnr_metric is not None and len(all_psnr) > 0:
        print(f"\n{'=' * 50}")
        print(f"Summary ({len(all_psnr)} images)")
        print(f"{'=' * 50}")
        print(f"  PSNR:  {np.mean(all_psnr):.2f} dB (±{np.std(all_psnr):.2f})")
        print(f"  SSIM:  {np.mean(all_ssim):.4f} (±{np.std(all_ssim):.4f})")


def main():
    """Main inference function."""
    args = parse_args()

    # Validate arguments
    if args.encode and args.decode:
        raise ValueError("Cannot specify both --encode and --decode")

    # Get resolution
    width, height = get_resolution(args)
    print(f"Resolution: {width}x{height}")

    # Load model
    print(f"Loading model from: {args.checkpoint_path or args.model_path}")
    model = load_model(args)
    print(f"Device: {args.device}, dtype: {args.dtype}")

    # Get input paths and show count
    if args.decode:
        input_paths = get_input_paths(args.input, mode="latent")
    else:
        input_paths = get_input_paths(args.input, mode="image")
    print(f"Found {len(input_paths)} input file(s)\n")

    if args.encode:
        encode_images(
            model=model,
            input_paths=input_paths,
            output_arg=args.output,
            width=width,
            height=height,
            batch_size=args.batch_size,
            device=args.device,
            quiet=args.quiet,
        )

    elif args.decode:
        decode_latents(
            model=model,
            input_paths=input_paths,
            output_arg=args.output,
            batch_size=args.batch_size,
            device=args.device,
            quiet=args.quiet,
        )

    else:
        # Reconstruct image (encode + decode)
        reconstruct_images(
            model=model,
            input_paths=input_paths,
            output_arg=args.output,
            width=width,
            height=height,
            batch_size=args.batch_size,
            device=args.device,
            compute_metrics=not args.no_metrics,
            quiet=args.quiet,
        )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
