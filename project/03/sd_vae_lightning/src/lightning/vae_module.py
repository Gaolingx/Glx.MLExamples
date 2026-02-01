"""
PyTorch Lightning module for AutoencoderKL training and inference.
Supports TensorBoard logging, checkpoint management, and various loss functions.
"""

from typing import Optional, Dict, Any, Tuple, List
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import lpips
import torchvision

from ..models.autoencoder_kl import AutoencoderKL
from ..models.vae import DiagonalGaussianDistribution


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator for adversarial training.

    Args:
        input_nc: Number of input channels.
        ndf: Base number of discriminator filters.
        n_layers: Number of layers in discriminator.
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        layers += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights with normal distribution."""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator."""
        return self.model(x)


class VAELightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training AutoencoderKL.

    Supports:
    - Reconstruction loss (L1/L2)
    - Perceptual loss (LPIPS)
    - KL divergence loss
    - Adversarial loss with PatchGAN discriminator
    - Alternating training between VAE and Discriminator (official diffusers style)
    - Manual gradient accumulation

    Args:
        config: Configuration dictionary containing model and training parameters.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Disable automatic optimization for GAN training with manual gradient accumulation
        self.automatic_optimization = False

        # Build VAE model
        model_config = config.get("model", {})
        pretrained_path = model_config.get("pretrained_model_name_or_path")

        if pretrained_path:
            self.vae = AutoencoderKL.from_pretrained(pretrained_path)
        else:
            self.vae = AutoencoderKL(
                in_channels=model_config.get("in_channels", 3),
                out_channels=model_config.get("out_channels", 3),
                down_block_types=tuple(
                    model_config.get(
                        "down_block_types",
                        [
                            "DownEncoderBlock2D",
                            "DownEncoderBlock2D",
                            "DownEncoderBlock2D",
                            "DownEncoderBlock2D",
                        ],
                    )
                ),
                up_block_types=tuple(
                    model_config.get(
                        "up_block_types",
                        [
                            "UpDecoderBlock2D",
                            "UpDecoderBlock2D",
                            "UpDecoderBlock2D",
                            "UpDecoderBlock2D",
                        ],
                    )
                ),
                block_out_channels=tuple(
                    model_config.get("block_out_channels", [128, 256, 512, 512])
                ),
                layers_per_block=model_config.get("layers_per_block", 2),
                act_fn=model_config.get("act_fn", "silu"),
                latent_channels=model_config.get("latent_channels", 4),
                norm_num_groups=model_config.get("norm_num_groups", 32),
                sample_size=model_config.get("sample_size", 512),
                scaling_factor=model_config.get("scaling_factor", 0.18215),
                use_quant_conv=model_config.get("use_quant_conv", True),
                use_post_quant_conv=model_config.get("use_post_quant_conv", True),
                mid_block_add_attention=model_config.get(
                    "mid_block_add_attention", True
                ),
            )

        # Build discriminator
        loss_config = config.get("loss", {})
        self.use_discriminator = loss_config.get("disc_weight", 0) > 0
        if self.use_discriminator:
            self.discriminator = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3)
            self.disc_start_step = loss_config.get("disc_start_step", 50001)
        else:
            self.discriminator = None
            self.disc_start_step = float("inf")

        # Perceptual loss
        self.perceptual_weight = loss_config.get("perceptual_weight", 0.5)
        if self.perceptual_weight > 0:
            self.perceptual_loss = lpips.LPIPS(net="vgg")
            self.perceptual_loss.eval()
            for param in self.perceptual_loss.parameters():
                param.requires_grad = False
        else:
            self.perceptual_loss = None

        # Loss weights
        self.rec_loss_type = loss_config.get("rec_loss_type", "l2")
        self.kl_weight = loss_config.get("kl_weight", 1e-6)
        self.disc_weight = loss_config.get("disc_weight", 1.0)
        self.disc_loss_type = loss_config.get("disc_loss_type", "hinge")

        # Training config
        train_config = config.get("training", {})
        self.learning_rate = train_config.get("learning_rate", 4.5e-6)
        self.disc_learning_rate = train_config.get("disc_learning_rate", 4.5e-6)
        self.accumulate_grad_batches = train_config.get("accumulate_grad_batches", 1)
        self.gradient_clip_val = train_config.get("gradient_clip_val", 1.0)

        # Logging config
        log_config = config.get("logging", {})
        self.log_images_every_n_steps = log_config.get("log_images_every_n_steps", 500)
        self.num_val_images = log_config.get("num_val_images", 4)

        # Validation images storage
        self.validation_step_outputs: List[Dict[str, torch.Tensor]] = []
        
        # Cache for G/D loss ratio computation across alternating steps
        self._cached_g_loss: Optional[torch.Tensor] = None
        self._cached_vae_loss_dict: Optional[Dict[str, torch.Tensor]] = None

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """
        Forward pass: encode and decode.

        Args:
            x: Input image tensor.
            sample_posterior: Whether to sample from posterior.

        Returns:
            Tuple of (reconstructed image, posterior distribution).
        """
        posterior = self.vae.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.vae.decode(z).sample
        return dec, posterior

    def forward_with_latent(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, DiagonalGaussianDistribution]:
        """
        Forward pass: encode and decode, also return latent.

        Args:
            x: Input image tensor.
            sample_posterior: Whether to sample from posterior.

        Returns:
            Tuple of (reconstructed image, latent tensor, posterior distribution).
        """
        posterior = self.vae.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.vae.decode(z).sample
        return dec, z, posterior

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent.

        Args:
            x: Input image tensor (range [-1, 1]).

        Returns:
            Scaled latent tensor.
        """
        return self.vae.encode_to_latent(x, sample_posterior=True)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.

        Args:
            z: Scaled latent tensor.

        Returns:
            Decoded image tensor (range [-1, 1]).
        """
        return self.vae.decode_from_latent(z)

    def _compute_discriminator_win_rate(
        self,
        logits_real: torch.Tensor,
        logits_fake: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute discriminator win rate metrics.

        For hinge loss:
        - Discriminator "wins" on real if logits_real > 0
        - Discriminator "wins" on fake if logits_fake < 0

        Args:
            logits_real: Discriminator output for real images.
            logits_fake: Discriminator output for fake/reconstructed images.

        Returns:
            Dictionary containing win rate metrics.
        """
        # Flatten logits for computation
        logits_real_flat = logits_real.view(-1)
        logits_fake_flat = logits_fake.view(-1)

        # Compute accuracy for real samples (D should output > 0 for real)
        real_correct = (logits_real_flat > 0).float()
        win_rate_real = real_correct.mean()

        # Compute accuracy for fake samples (D should output < 0 for fake)
        fake_correct = (logits_fake_flat < 0).float()
        win_rate_fake = fake_correct.mean()

        # Overall win rate (average of real and fake accuracy)
        win_rate = 0.5 * (win_rate_real + win_rate_fake)

        return {
            "win_rate": win_rate,
            "win_rate_real": win_rate_real,
            "win_rate_fake": win_rate_fake,
        }

    def _compute_loss_ratio(
        self,
        g_loss: torch.Tensor,
        d_loss: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute the ratio between Generator and Discriminator losses.

        Args:
            g_loss: Generator loss.
            d_loss: Discriminator loss.
            eps: Small epsilon to avoid division by zero.

        Returns:
            G/D loss ratio.
        """
        return g_loss / (d_loss + eps)

    def compute_loss(
        self,
        targets: torch.Tensor,
        reconstructions: torch.Tensor,
        posterior: DiagonalGaussianDistribution,
        optimizer_idx: int,
        global_step: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute training loss.

        Args:
            targets: Original images.
            reconstructions: Reconstructed images.
            posterior: Latent distribution.
            optimizer_idx: 0 for VAE, 1 for discriminator.
            global_step: Current training step.

        Returns:
            Tuple of (loss, loss_dict).
        """
        loss_dict = {}

        if optimizer_idx == 0:
            # VAE losses
            # Reconstruction loss
            if self.rec_loss_type == "l2":
                rec_loss = F.mse_loss(reconstructions, targets, reduction="none")
            else:
                rec_loss = F.l1_loss(reconstructions, targets, reduction="none")
            rec_loss = rec_loss.mean()
            loss_dict["rec_loss"] = rec_loss

            # Perceptual loss
            if self.perceptual_loss is not None and self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(reconstructions, targets).mean()
                loss_dict["p_loss"] = p_loss
            else:
                p_loss = torch.tensor(0.0, device=targets.device)
                loss_dict["p_loss"] = p_loss

            # KL loss
            kl_loss = posterior.kl().mean()
            loss_dict["kl_loss"] = kl_loss

            # Total NLL loss (reconstruction + perceptual)
            nll_loss = rec_loss + self.perceptual_weight * p_loss
            loss_dict["nll_loss"] = nll_loss

            # Generator loss from discriminator
            if (
                self.discriminator is not None
                and global_step >= self.disc_start_step
            ):
                logits_fake = self.discriminator(reconstructions)
                g_loss = -logits_fake.mean()
                loss_dict["g_loss"] = g_loss

                # Adaptive weight based on gradient norms
                last_layer = self.vae.decoder.conv_out.weight
                nll_grads = torch.autograd.grad(
                    nll_loss, last_layer, retain_graph=True
                )[0]
                g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

                d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
                d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
                d_weight = d_weight * self.disc_weight
                loss_dict["disc_weight"] = d_weight

                loss = nll_loss + self.kl_weight * kl_loss + d_weight * g_loss
            else:
                loss = nll_loss + self.kl_weight * kl_loss
                loss_dict["g_loss"] = torch.tensor(0.0, device=targets.device)
                loss_dict["disc_weight"] = torch.tensor(0.0, device=targets.device)

            loss_dict["loss"] = loss
            return loss, loss_dict

        else:
            # Discriminator loss
            logits_real = self.discriminator(targets.detach())
            logits_fake = self.discriminator(reconstructions.detach())

            if self.disc_loss_type == "hinge":
                d_loss_real = torch.mean(F.relu(1.0 - logits_real))
                d_loss_fake = torch.mean(F.relu(1.0 + logits_fake))
            else:  # vanilla
                d_loss_real = F.binary_cross_entropy_with_logits(
                    logits_real, torch.ones_like(logits_real)
                )
                d_loss_fake = F.binary_cross_entropy_with_logits(
                    logits_fake, torch.zeros_like(logits_fake)
                )

            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            loss_dict["d_loss"] = d_loss
            loss_dict["d_loss_real"] = d_loss_real
            loss_dict["d_loss_fake"] = d_loss_fake
            loss_dict["logits_real"] = logits_real.mean()
            loss_dict["logits_fake"] = logits_fake.mean()

            # Compute discriminator win rate
            win_rate_dict = self._compute_discriminator_win_rate(logits_real, logits_fake)
            loss_dict.update(win_rate_dict)

            return d_loss, loss_dict

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Training step with alternating VAE/Discriminator training.
        
        Following the official diffusers implementation:
        - Even accumulated steps: train VAE (generator)
        - Odd accumulated steps: train Discriminator
        - Before disc_start_step: always train VAE only
        
        Manual gradient accumulation is used to match the official behavior.

        Args:
            batch: Batch dictionary containing 'pixel_values'.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        targets = batch["pixel_values"]
        optimizers = self.optimizers()
        
        # Handle single or multiple optimizers
        if isinstance(optimizers, list):
            opt_vae, opt_disc = optimizers
        else:
            opt_vae = optimizers
            opt_disc = None

        # Determine training phase based on accumulated step index
        # accumulated_step_idx: which "logical" optimization step we're in
        accumulated_step_idx = batch_idx // self.accumulate_grad_batches
        is_last_accumulation_step = (batch_idx + 1) % self.accumulate_grad_batches == 0
        
        # Determine whether to train generator (VAE) or discriminator
        # Even accumulated steps -> train VAE, Odd -> train Discriminator
        # Always train VAE before disc_start_step
        train_generator = (
            (accumulated_step_idx % 2 == 0) or 
            (self.global_step < self.disc_start_step) or
            (self.discriminator is None)
        )

        # Forward pass with latent
        reconstructions, latent, posterior = self.forward_with_latent(targets, sample_posterior=True)

        if train_generator:
            # ========== Train VAE (Generator) ==========
            vae_loss, vae_loss_dict = self.compute_loss(
                targets, reconstructions, posterior, 
                optimizer_idx=0, global_step=self.global_step
            )
            
            # Scale loss for manual gradient accumulation
            scaled_loss = vae_loss / self.accumulate_grad_batches
            self.manual_backward(scaled_loss)
            
            # Step optimizer only at accumulation boundary
            if is_last_accumulation_step:
                self.clip_gradients(
                    opt_vae,
                    gradient_clip_val=self.gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )
                opt_vae.step()
                opt_vae.zero_grad()
            
            # Log VAE losses
            for key, value in vae_loss_dict.items():
                self.log(f"train/{key}", value, on_step=True, on_epoch=True, prog_bar=True)
            
            # Cache g_loss and loss dict for G/D ratio computation in discriminator step
            self._cached_g_loss = vae_loss_dict.get("g_loss", torch.tensor(0.0, device=targets.device))
            self._cached_vae_loss_dict = vae_loss_dict
            
            # Log images periodically (only during VAE training steps)
            if is_last_accumulation_step and self.global_step % self.log_images_every_n_steps == 0:
                self._log_images(targets, reconstructions, latent, "train")
            
            return vae_loss
        
        else:
            # ========== Train Discriminator ==========
            if self.discriminator is not None and opt_disc is not None:
                disc_loss, disc_loss_dict = self.compute_loss(
                    targets, reconstructions, posterior,
                    optimizer_idx=1, global_step=self.global_step,
                )
                
                # Scale loss for manual gradient accumulation
                scaled_loss = disc_loss / self.accumulate_grad_batches
                self.manual_backward(scaled_loss)
                
                # Step optimizer only at accumulation boundary
                if is_last_accumulation_step:
                    self.clip_gradients(
                        opt_disc,
                        gradient_clip_val=self.gradient_clip_val,
                        gradient_clip_algorithm="norm",
                    )
                    opt_disc.step()
                    opt_disc.zero_grad()
                
                # Log discriminator losses
                for key, value in disc_loss_dict.items():
                    self.log(f"train/{key}", value, on_step=True, on_epoch=True, prog_bar=False)
                
                # Compute and log G/D loss ratio using cached g_loss from previous VAE step
                if self._cached_g_loss is not None:
                    g_loss = self._cached_g_loss
                    d_loss = disc_loss_dict.get("d_loss", torch.tensor(1.0, device=targets.device))
                    gd_ratio = self._compute_loss_ratio(g_loss, d_loss)
                    self.log("train/gd_loss_ratio", gd_ratio, on_step=True, on_epoch=True, prog_bar=True)
                
                return disc_loss
            
            return None

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Validation step.

        Args:
            batch: Batch dictionary containing 'pixel_values'.
            batch_idx: Batch index.

        Returns:
            Loss dictionary.
        """
        targets = batch["pixel_values"]
        reconstructions, latent, posterior = self.forward_with_latent(targets, sample_posterior=False)

        loss, loss_dict = self.compute_loss(
            targets, reconstructions, posterior, optimizer_idx=0, global_step=self.global_step
        )

        # Log losses
        for key, value in loss_dict.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=True)

        # Store for epoch end logging
        if batch_idx < self.num_val_images:
            self.validation_step_outputs.append(
                {
                    "targets": targets.detach().cpu(),
                    "reconstructions": reconstructions.detach().cpu(),
                    "latent": latent.detach().cpu(),
                }
            )

        return loss_dict

    def on_validation_epoch_end(self) -> None:
        """Log validation images at epoch end."""
        if len(self.validation_step_outputs) > 0:
            targets = torch.cat(
                [out["targets"] for out in self.validation_step_outputs], dim=0
            )
            reconstructions = torch.cat(
                [out["reconstructions"] for out in self.validation_step_outputs], dim=0
            )
            latent = torch.cat(
                [out["latent"] for out in self.validation_step_outputs], dim=0
            )
            self._log_images(
                targets[: self.num_val_images],
                reconstructions[: self.num_val_images],
                latent[: self.num_val_images],
                "val",
            )
            self.validation_step_outputs.clear()

    def _visualize_latent(
        self,
        latent: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Visualize latent space as an image.

        Args:
            latent: Latent tensor of shape (B, C, H, W).
            target_size: Target size (H, W) to upsample to.

        Returns:
            Visualization tensor of shape (B, 3, H', W') normalized to [0, 1].
        """
        b, c, h, w = latent.shape

        # Min-max normalize per image (across all channels)
        latent_flat = latent.view(b, -1)
        min_vals = latent_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        max_vals = latent_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals < 1e-5, torch.ones_like(range_vals), range_vals)
        
        latent_normalized = (latent - min_vals) / range_vals

        # Use first 3 channels as RGB if available, otherwise use grayscale
        if c >= 3:
            latent_vis = latent_normalized[:, :3]
        else:
            # Repeat single channel to RGB
            latent_vis = latent_normalized[:, :1].repeat(1, 3, 1, 1)

        # Upsample to target size for consistent grid visualization
        latent_vis = F.interpolate(
            latent_vis,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return latent_vis

    def _log_images(
        self,
        targets: torch.Tensor,
        reconstructions: torch.Tensor,
        latent: torch.Tensor,
        prefix: str,
    ) -> None:
        """
        Log images to TensorBoard.

        Args:
            targets: Original images.
            reconstructions: Reconstructed images.
            latent: Latent space tensor.
            prefix: Logging prefix ('train' or 'val').
        """
        if not self.logger:
            return

        # Denormalize images from [-1, 1] to [0, 1]
        targets = (targets + 1) / 2
        reconstructions = (reconstructions + 1) / 2

        # Clamp values
        targets = torch.clamp(targets, 0, 1)
        reconstructions = torch.clamp(reconstructions, 0, 1)

        # Visualize latent space (upsample to match target size)
        target_size = (targets.shape[2], targets.shape[3])
        latent_vis = self._visualize_latent(latent, target_size)

        # Create grid with 3 rows: targets, latent, reconstructions
        n = min(4, targets.shape[0])
        comparison = torch.cat([targets[:n], latent_vis[:n], reconstructions[:n]], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=n, padding=2)

        # Log to tensorboard
        if hasattr(self.logger, "experiment"):
            self.logger.experiment.add_image(
                f"{prefix}/reconstruction",
                grid,
                self.global_step,
            )

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        train_config = self.config.get("training", {})

        # VAE optimizer
        vae_params = list(self.vae.parameters())
        opt_vae = torch.optim.AdamW(
            vae_params,
            lr=self.learning_rate,
            betas=(train_config.get("adam_beta1", 0.9), train_config.get("adam_beta2", 0.999)),
            weight_decay=train_config.get("weight_decay", 0.01),
        )

        if self.discriminator is not None:
            # Discriminator optimizer
            disc_params = list(self.discriminator.parameters())
            opt_disc = torch.optim.AdamW(
                disc_params,
                lr=self.disc_learning_rate,
                betas=(train_config.get("adam_beta1", 0.9), train_config.get("adam_beta2", 0.999)),
                weight_decay=train_config.get("weight_decay", 0.01),
            )
            return [opt_vae, opt_disc]

        return opt_vae

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save model to directory.

        Args:
            save_directory: Path to save directory.
        """
        self.vae.save_pretrained(save_directory)

        # Save discriminator if exists
        if self.discriminator is not None:
            disc_path = Path(save_directory) / "discriminator.pt"
            torch.save(self.discriminator.state_dict(), disc_path)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, config: Optional[Dict[str, Any]] = None, **kwargs
    ) -> "VAELightningModule":
        """
        Load model from Lightning checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            config: Optional config override.
            **kwargs: Additional arguments.

        Returns:
            Loaded VAELightningModule.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if config is None:
            config = checkpoint.get("hyper_parameters", {})

        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])

        return model
