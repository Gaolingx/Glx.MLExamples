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
from diffusers import AutoencoderKL
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import lpips
import math

from ..utils.metrics import PSNR, SSIM, rFID, PSIM
from ..models.discriminator import NLayerDiscriminator


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
        self.config = config
        self.save_hyperparameters({"config": config})

        model_config = config.get("model", {})
        loss_config = config.get("loss", {})
        disc_config = config.get("discriminator", {})
        train_config = config.get("training", {})

        # Disable automatic optimization for GAN training with manual gradient accumulation
        self.automatic_optimization = False

        # Build VAE model
        pretrained_path = model_config.get("pretrained_model_name_or_path")
        model_config_name_or_path = model_config.get("model_config_name_or_path")
        inline_vae_config = model_config.get("vae_config")

        if pretrained_path:
            self.vae = AutoencoderKL.from_pretrained(pretrained_path)
        elif model_config_name_or_path is not None:
            vae_config = AutoencoderKL.load_config(model_config_name_or_path)
            self.vae = AutoencoderKL.from_config(vae_config)
        elif inline_vae_config is not None:
            self.vae = AutoencoderKL.from_config(inline_vae_config)
        else:
            raise ValueError(
                "Must specify one of `pretrained_model_name_or_path`, "
                "`model_config_name_or_path`, or `vae_config` in the config."
            )

        # Enable gradient checkpointing for memory savings
        if model_config.get("gradient_checkpointing", False):
            self.vae.enable_gradient_checkpointing()

        # Build discriminator
        self.use_discriminator = loss_config.get("disc_weight", 1.0) > 0

        if self.use_discriminator:
            self.discriminator = NLayerDiscriminator(
                input_nc=disc_config.get("input_nc", 3),
                ndf=disc_config.get("ndf", 64),
                n_layers=disc_config.get("n_layers", 3),
                norm_type=disc_config.get("norm_type", "spectral_group"),
                num_groups=disc_config.get("num_groups", 32),
            )
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
        self.disc_factor = loss_config.get("disc_factor", 1.0)
        self.use_adaptive_disc_weight = loss_config.get("use_adaptive_disc_weight", True)
        self.adaptive_weight_max = loss_config.get("adaptive_weight_max", 10.0)

        # Generator optimizer config
        gen_opt_cfg = train_config.get("generator_optimizer", {})
        self.gen_opt_config = {
            "type": gen_opt_cfg.get("type", "AdamW"),
            "learning_rate": gen_opt_cfg.get("learning_rate", 4.5e-6),
            "adam_beta1": gen_opt_cfg.get("adam_beta1", 0.9),
            "adam_beta2": gen_opt_cfg.get("adam_beta2", 0.999),
            "weight_decay": gen_opt_cfg.get("weight_decay", 0.01),
            "lr_scheduler": gen_opt_cfg.get("lr_scheduler", "constant_with_warmup"),
            "lr_warmup_steps": gen_opt_cfg.get("lr_warmup_steps", 500),
            "lr_num_cycles": gen_opt_cfg.get("lr_num_cycles", 1),
            "lr_power": gen_opt_cfg.get("lr_power", 1.0),
        }

        # Discriminator optimizer config
        disc_opt_cfg = train_config.get("discriminator_optimizer", {})
        self.disc_opt_config = {
            "type": disc_opt_cfg.get("type", "AdamW"),
            "learning_rate": disc_opt_cfg.get("learning_rate", 4.5e-6),
            "adam_beta1": disc_opt_cfg.get("adam_beta1", 0.9),
            "adam_beta2": disc_opt_cfg.get("adam_beta2", 0.999),
            "weight_decay": disc_opt_cfg.get("weight_decay", 0.01),
            "lr_scheduler": disc_opt_cfg.get("lr_scheduler", "constant_with_warmup"),
            "lr_warmup_steps": disc_opt_cfg.get("lr_warmup_steps", 500),
            "lr_num_cycles": disc_opt_cfg.get("lr_num_cycles", 1),
            "lr_power": disc_opt_cfg.get("lr_power", 1.0),
        }

        # Keep top-level attributes for backward compatibility with training_step logging
        self.learning_rate = self.gen_opt_config["learning_rate"]
        self.disc_learning_rate = self.disc_opt_config["learning_rate"]

        self.accumulate_grad_batches = train_config.get("accumulate_grad_batches", 1)
        self.gradient_clip_val = train_config.get("gradient_clip_val", 1.0)

        # EMA configuration
        self.use_ema = train_config.get("use_ema", True)
        self.ema_decay = train_config.get("ema_decay", 0.9999)
        self.ema_update_interval = train_config.get("ema_update_interval", 1)
        self.ema: Optional[EMAModel] = None

        # Validation metrics
        self.psnr_metric = PSNR(data_range=2.0)
        self.ssim_metric = SSIM(data_range=2.0)
        self.psim_metric = PSIM(net="vgg")

        # rFID metric for reconstruction quality evaluation
        self.rfid_metric = rFID(feature_dim=2048, reset_real_features=True)

        # Cache for G/D loss ratio computation across alternating steps
        self._cached_g_loss: Optional[torch.Tensor] = None

    def on_fit_start(self) -> None:
        """Initialize EMA model at the start of training."""
        if self.use_ema:
            if self.ema is None:
                self.ema = EMAModel(
                    self.vae.parameters(),
                    decay=self.ema_decay,
                    model_cls=AutoencoderKL,
                    model_config=self.vae.config,
                )

            self.ema.to(self.device)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.use_ema and self.ema is not None:
            checkpoint["ema_state"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.use_ema and "ema_state" in checkpoint:
            if self.ema is None:
                self.ema = EMAModel(
                    self.vae.parameters(),
                    decay=self.ema_decay,
                    model_cls=AutoencoderKL,
                    model_config=self.vae.config,
                )

            self.ema.load_state_dict(checkpoint["ema_state"])

    def forward_with_latent(
            self,
            x: torch.Tensor,
            sample_posterior: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
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

    def _calculate_adaptive_weight(
            self,
            nll_loss: torch.Tensor,
            g_loss: torch.Tensor,
            last_layer: nn.Parameter,
    ) -> torch.Tensor:
        """
        Calculate adaptive discriminator weight based on gradient norms.

        This balances the reconstruction loss and adversarial loss by computing
        the ratio of their gradient norms w.r.t. the last decoder layer.
        Following the approach from taming-transformers and diffusers.

        Args:
            nll_loss: Reconstruction + perceptual loss (NLL loss).
            g_loss: Generator adversarial loss.
            last_layer: Last layer weights of decoder for gradient computation.

        Returns:
            Adaptive weight for discriminator loss, scaled by disc_weight.
        """
        try:
            nll_grads = torch.autograd.grad(
                nll_loss, last_layer, retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, last_layer, retain_graph=True
            )[0]
        except RuntimeError:
            # Fallback if gradient computation fails
            return torch.tensor(0.0, device=nll_loss.device)

        nll_grad_norm = torch.norm(nll_grads)
        g_grad_norm = torch.norm(g_grads)

        if g_grad_norm < 1e-6:
            d_weight = torch.tensor(1.0, device=nll_loss.device)
        else:
            d_weight = nll_grad_norm / (g_grad_norm + 1e-4)

        d_weight = torch.clamp(d_weight, 0.0, self.adaptive_weight_max).detach()

        return d_weight * self.disc_weight

    def compute_loss(
            self,
            targets: torch.Tensor,
            reconstructions: torch.Tensor,
            posterior: Any,
            optimizer_idx: int,
            global_step: int,
            training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute training loss.

        Args:
            targets: Original images.
            reconstructions: Reconstructed images.
            posterior: Latent distribution.
            optimizer_idx: 0 for VAE, 1 for discriminator.
            global_step: Current training step.
            training: Whether in training mode (enables gradient-based adaptive weight).

        Returns:
            Tuple of (loss, loss_dict).
        """
        loss_dict = {}

        if optimizer_idx == 0:
            # ========== VAE (Generator) losses ==========

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
            loss_dict["kl_loss"] = kl_loss * self.kl_weight

            # Total NLL loss (reconstruction + perceptual)
            nll_loss = rec_loss + self.perceptual_weight * p_loss
            loss_dict["nll_loss"] = nll_loss

            # Compute disc_factor based on disc_start_step
            disc_factor = self.disc_factor if global_step >= self.disc_start_step else 0.0
            loss_dict["disc_factor"] = torch.tensor(disc_factor, device=targets.device)

            if (
                    self.discriminator is not None
                    and global_step >= self.disc_start_step
                    and training
            ):
                # Generator adversarial loss
                logits_fake = self.discriminator(reconstructions)
                g_loss = -logits_fake.mean()
                loss_dict["g_loss"] = g_loss

                # Calculate discriminator weight
                if self.use_adaptive_disc_weight:
                    last_layer = self.vae.decoder.conv_out.weight
                    d_weight = self._calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer
                    )
                    loss_dict["disc_weight"] = d_weight
                else:
                    d_weight = torch.tensor(self.disc_weight, device=targets.device)
                    loss_dict["disc_weight"] = d_weight

                # Final weighted adversarial loss with warmup factor
                g_loss_weighted = d_weight * disc_factor * g_loss
                loss_dict["g_loss_weighted"] = g_loss_weighted

                loss = nll_loss + self.kl_weight * kl_loss + g_loss_weighted
            else:
                loss = nll_loss + self.kl_weight * kl_loss
                loss_dict["g_loss"] = torch.tensor(0.0, device=targets.device)
                loss_dict["disc_weight"] = torch.tensor(0.0, device=targets.device)
                loss_dict["g_loss_weighted"] = torch.tensor(0.0, device=targets.device)

            loss_dict["loss"] = loss
            return loss, loss_dict

        else:
            # ========== Discriminator losses ==========

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

            # Compute disc_factor based on disc_start_step
            disc_factor = self.disc_factor if global_step >= self.disc_start_step else 0.0

            d_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

            loss_dict["d_loss"] = d_loss
            loss_dict["disc_factor"] = torch.tensor(disc_factor, device=targets.device)
            loss_dict["d_loss_real"] = d_loss_real
            loss_dict["d_loss_fake"] = d_loss_fake
            loss_dict["logits_real"] = logits_real.mean()
            loss_dict["logits_fake"] = logits_fake.mean()

            # Compute discriminator win rate
            win_rate_dict = self._compute_discriminator_win_rate(logits_real, logits_fake)
            loss_dict.update(win_rate_dict)

            return d_loss, loss_dict

    def _compute_global_norm(self, parameters, use_grad: bool = True) -> torch.Tensor:
        """
        Compute total gradient norm for given parameters.

        Args:
            parameters: Iterator of model parameters.
            use_grad: If True, compute the L2 norm of parameter gradients;
                      if False, compute the L2 norm of the parameter values themselves.

        Returns:
            Total L2 norm of gradients.
        """
        reference = None
        total = None

        for param in parameters:
            if not param.requires_grad:
                continue

            tensor = param.grad if use_grad else param.detach()
            if tensor is None:
                continue

            reference = tensor
            part = tensor.detach().float().pow(2).sum()
            total = part if total is None else total + part

        if total is None:
            if reference is not None:
                return torch.tensor(0.0, device=reference.device)
            return torch.tensor(0.0, device=self.device)

        return total.sqrt()

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
        train_metrics: Dict[str, torch.Tensor] = {}
        optimizers = self.optimizers()

        # Handle single or multiple optimizers
        if isinstance(optimizers, list):
            opt_vae, opt_disc = optimizers
        else:
            opt_vae = optimizers
            opt_disc = None

        # Determine training phase based on accumulated step index
        accumulated_step_idx = batch_idx // self.accumulate_grad_batches
        is_last_accumulation_step = (batch_idx + 1) % self.accumulate_grad_batches == 0

        # Determine whether to train generator (VAE) or discriminator
        train_generator = (
                (accumulated_step_idx % 2 == 0) or
                (self.global_step < self.disc_start_step) or
                (self.discriminator is None)
        )

        # Forward pass with latent
        reconstructions, latent, posterior = self.forward_with_latent(targets, sample_posterior=True)

        if train_generator:
            # ========== Train VAE (Generator) ==========
            self.toggle_optimizer(opt_vae)
            vae_loss, vae_loss_dict = self.compute_loss(
                targets, reconstructions, posterior,
                optimizer_idx=0, global_step=self.global_step, training=True
            )

            # Scale loss for manual gradient accumulation
            scaled_loss = vae_loss / self.accumulate_grad_batches
            self.manual_backward(scaled_loss)
            self.untoggle_optimizer(opt_vae)

            # Step optimizer only at accumulation boundary
            if is_last_accumulation_step:
                grad_norm_vae = self._compute_global_norm(self.vae.parameters(), use_grad=True)
                self.clip_gradients(
                    opt_vae,
                    gradient_clip_val=self.gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )
                grad_norm_vae_clip = self._compute_global_norm(self.vae.parameters(), use_grad=True)
                opt_vae.step()
                opt_vae.zero_grad()

                # Update EMA
                if self.use_ema and self.ema is not None:
                    if self.global_step % self.ema_update_interval == 0:
                        self.ema.step(self.vae.parameters())

                # Step the VAE learning rate scheduler
                schedulers = self.lr_schedulers()
                if isinstance(schedulers, list):
                    vae_scheduler = schedulers[0]
                else:
                    vae_scheduler = schedulers
                vae_scheduler.step()

                train_metrics["grad_norm_vae"] = grad_norm_vae
                train_metrics["grad_norm_vae_clip"] = grad_norm_vae_clip

            for key, value in vae_loss_dict.items():
                train_metrics[key] = value
            # Learning rate metric
            schedulers = self.lr_schedulers()
            if isinstance(schedulers, list):
                vae_lr = schedulers[0].get_last_lr()[0]
            else:
                vae_lr = schedulers.get_last_lr()[0]
            train_metrics["lr"] = torch.tensor(vae_lr, device=targets.device)

            # Cache g_loss for G/D ratio computation in discriminator step
            self._cached_g_loss = vae_loss_dict.get("g_loss", torch.tensor(0.0, device=targets.device))

            result = vae_loss

        else:
            # ========== Train Discriminator ==========
            if self.discriminator is not None and opt_disc is not None:
                self.toggle_optimizer(opt_disc)
                disc_loss, disc_loss_dict = self.compute_loss(
                    targets, reconstructions, posterior,
                    optimizer_idx=1, global_step=self.global_step,
                )

                # Scale loss for manual gradient accumulation
                scaled_loss = disc_loss / self.accumulate_grad_batches
                self.manual_backward(scaled_loss)
                self.untoggle_optimizer(opt_disc)

                # Step optimizer only at accumulation boundary
                if is_last_accumulation_step:
                    grad_norm_disc = self._compute_global_norm(self.discriminator.parameters(), use_grad=True)
                    self.clip_gradients(
                        opt_disc,
                        gradient_clip_val=self.gradient_clip_val,
                        gradient_clip_algorithm="norm"
                    )
                    grad_norm_disc_clip = self._compute_global_norm(self.discriminator.parameters(), use_grad=True)
                    opt_disc.step()
                    opt_disc.zero_grad()

                    # Step the discriminator learning rate scheduler
                    schedulers = self.lr_schedulers()
                    if isinstance(schedulers, list) and len(schedulers) > 1:
                        disc_scheduler = schedulers[1]
                        disc_scheduler.step()
                    train_metrics["grad_norm_disc"] = grad_norm_disc
                    train_metrics["grad_norm_disc_clip"] = grad_norm_disc_clip

                if self._cached_g_loss is not None:
                    g_loss = self._cached_g_loss
                    d_loss = disc_loss_dict.get("d_loss", torch.tensor(1.0, device=targets.device))
                    gd_ratio = self._compute_loss_ratio(g_loss, d_loss)
                    disc_loss_dict["gd_loss_ratio"] = gd_ratio

                for key, value in disc_loss_dict.items():
                    train_metrics[key] = value
                schedulers = self.lr_schedulers()
                if isinstance(schedulers, list) and len(schedulers) > 1:
                    disc_lr = schedulers[1].get_last_lr()[0]
                    train_metrics["disc_lr"] = torch.tensor(disc_lr, device=targets.device)
                result = disc_loss
            else:
                result = None

        return {
            "loss": result,
            "train_metrics": train_metrics,
        }

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

        # Use EMA weights for validation if available
        if self.use_ema and self.ema is not None:
            self.ema.store(self.vae.parameters())
            self.ema.copy_to(self.vae.parameters())

        try:
            reconstructions, latent, posterior = self.forward_with_latent(targets, sample_posterior=False)

            loss, loss_dict = self.compute_loss(
                targets, reconstructions, posterior, optimizer_idx=0, global_step=self.global_step, training=False
            )

            # Compute PSNR, SSIM and PSIM
            with torch.no_grad():
                psnr = self.psnr_metric(reconstructions, targets)
                ssim = self.ssim_metric(reconstructions, targets)
                psim = self.psim_metric(reconstructions, targets)

                # Update rFID metric with current batch
                self.rfid_metric.update(targets, reconstructions)

            loss_dict["psnr"] = psnr
            loss_dict["ssim"] = ssim
            loss_dict["psim"] = psim

            val_metrics = {key: value for key, value in loss_dict.items()}
            val_visuals = {
                "targets": targets.detach().cpu(),
                "reconstructions": reconstructions.detach().cpu(),
                "latent": latent.detach().cpu(),
            }
        finally:
            # Restore original weights
            if self.use_ema and self.ema is not None:
                self.ema.restore(self.vae.parameters())

        return {
            "val_metrics": val_metrics,
            "val_visuals": val_visuals,
        }

    def _build_optimizer(
            self,
            params,
            opt_config: Dict[str, Any],
    ) -> torch.optim.Optimizer:
        """
        Build an optimizer from a per-role config dict.

        Args:
            params: Iterable of parameters to optimise.
            opt_config: Dictionary with keys:
                - type: Optimizer class name ('AdamW', 'Adam', 'SGD', 'RMSprop').
                - learning_rate: Base learning rate.
                - adam_beta1: Beta1 for Adam-family optimizers.
                - adam_beta2: Beta2 for Adam-family optimizers.
                - weight_decay: Weight decay coefficient.
                - momentum: Momentum for SGD (default 0.9).

        Returns:
            Configured optimizer instance.

        Raises:
            ValueError: If optimizer type is not supported.
        """
        opt_type = opt_config.get("type", "AdamW").lower()
        lr = opt_config["learning_rate"]
        betas = (opt_config.get("adam_beta1", 0.9), opt_config.get("adam_beta2", 0.999))
        weight_decay = opt_config.get("weight_decay", 0.01)

        if opt_type == "adamw":
            return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
        elif opt_type == "adam":
            return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
        elif opt_type == "sgd":
            momentum = opt_config.get("momentum", 0.9)
            return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif opt_type == "rmsprop":
            return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(
                f"Unknown optimizer type: {opt_type}. "
                f"Supported: 'AdamW', 'Adam', 'SGD', 'RMSprop'."
            )

    def _build_scheduler(
            self,
            optimizer: torch.optim.Optimizer,
            opt_config: Dict[str, Any],
            num_training_steps: int,
            last_epoch: int = -1,
    ):
        """
        Build a learning rate scheduler from a per-role config dict.

        Delegates entirely to ``diffusers.optimization.get_scheduler``, which
        internally validates required arguments per scheduler type.

        Args:
            optimizer: The optimizer to schedule.
            opt_config: Dictionary with keys:
                - lr_scheduler: Scheduler type. Supported:
                    'constant', 'constant_with_warmup', 'linear',
                    'cosine', 'cosine_with_restarts', 'polynomial',
                    'piecewise_constant'.
                - lr_warmup_steps: Number of warmup steps (default 500,
                    clamped to ``num_training_steps``).
                - lr_num_cycles: Cycles for cosine-family schedulers.
                    For 'cosine' this controls the fraction of a cosine
                    period (default 0.5 = half-cosine decay).
                    For 'cosine_with_restarts' this is the number of
                    hard restarts (default 1).
                - lr_power: Exponent for 'polynomial' decay (default 1.0).
                - lr_end: Final learning rate for 'polynomial' (default 1e-7).
                - lr_step_rules: Step rules string for 'piecewise_constant'
                    (e.g. "1:10,0.1:20,0.01:30,0.005").
            num_training_steps: Total training steps for this scheduler
                (already accounts for alternating G/D splits).
            last_epoch: Index of the last epoch for resuming training
                (default -1, meaning fresh start).

        Returns:
            ``LambdaLR`` scheduler from ``diffusers.optimization.get_scheduler``.

        Raises:
            ValueError: If scheduler type is not recognised by diffusers.
        """
        scheduler_type = opt_config.get("lr_scheduler", "constant_with_warmup")
        num_training_steps = max(1, num_training_steps)

        num_warmup_steps = opt_config.get("lr_warmup_steps", 500)
        num_warmup_steps = min(num_warmup_steps, num_training_steps)

        num_cycles = opt_config.get("lr_num_cycles", 1)
        power = opt_config.get("lr_power", 1.0)
        step_rules = opt_config.get("lr_step_rules", None)

        return get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            step_rules=step_rules,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            power=power,
            last_epoch=last_epoch,
        )

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Uses trainer.estimated_stepping_batches for accurate step count estimation.
        Handles alternating G/D training by computing separate step counts for
        each optimizer.

        Important considerations:
            - VAE scheduler steps on even accumulated steps + all steps before disc_start.
            - Discriminator scheduler steps on odd accumulated steps after disc_start.
            - Warmup steps are clamped to each scheduler's total training steps.
        """
        # ---- total training steps estimation ----
        estimated = self.trainer.estimated_stepping_batches
        if estimated is None or not math.isfinite(estimated):
            raise ValueError(
                "Cannot infer total training steps from Trainer. "
                "Please set `max_steps` in `pl.Trainer(...)`, or use finite "
                "`max_epochs` with a finite-length dataloader."
            )
        total_steps = max(1, int(math.ceil(estimated / self.accumulate_grad_batches)))
        # ---- Generator (VAE) ----
        opt_vae = self._build_optimizer(
            list(self.vae.parameters()),
            self.gen_opt_config,
        )

        disc_start_step = self.disc_start_step if self.use_discriminator else float("inf")

        if disc_start_step < total_steps:
            steps_before_disc = min(int(disc_start_step), total_steps)
            steps_after_disc = total_steps - steps_before_disc
            vae_scheduler_steps = steps_before_disc + (steps_after_disc + 1) // 2
        else:
            vae_scheduler_steps = total_steps

        vae_scheduler = self._build_scheduler(
            opt_vae,
            self.gen_opt_config,
            num_training_steps=max(1, vae_scheduler_steps),
        )

        # ---- Discriminator (if needed) ----
        if self.discriminator is not None:
            opt_disc = self._build_optimizer(
                list(self.discriminator.parameters()),
                self.disc_opt_config,
            )

            if disc_start_step < total_steps:
                disc_scheduler_steps = (total_steps - int(disc_start_step)) // 2
            else:
                disc_scheduler_steps = 0

            disc_scheduler = self._build_scheduler(
                opt_disc,
                self.disc_opt_config,
                num_training_steps=max(1, disc_scheduler_steps),
            )

            return (
                [opt_vae, opt_disc],
                [
                    {"scheduler": vae_scheduler, "interval": "step", "frequency": 1},
                    {"scheduler": disc_scheduler, "interval": "step", "frequency": 1},
                ],
            )

        return {
            "optimizer": opt_vae,
            "lr_scheduler": {
                "scheduler": vae_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def save_hf_checkpoint(self, checkpoint_filepath: str) -> None:
        """
        Save HuggingFace format model to directory.

        Args:
            checkpoint_filepath: Path to save directory.
        """
        save_path = Path(checkpoint_filepath)
        save_path.mkdir(parents=True, exist_ok=True)

        vae_save_dir = save_path / "vae"
        self.vae.save_pretrained(vae_save_dir)

        # Save VAE (optionally use EMA weights)
        if self.use_ema and self.ema is not None:
            ema_dir = save_path / "vae_ema"
            self.ema.save_pretrained(str(ema_dir))
            print(f"Saving EMA weights to {ema_dir}...")

        # Save training config
        config_path = save_path / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save model to directory.

        Args:
            save_directory: Path to save directory.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        vae_save_dir = save_path / "vae"
        self.vae.save_pretrained(vae_save_dir)

        # Save VAE (optionally use EMA weights)
        if self.use_ema and self.ema is not None:
            ema_dir = save_path / "vae_ema"
            self.ema.save_pretrained(str(ema_dir))
            print(f"Saving EMA weights to {ema_dir}...")

        # Save discriminator if exists
        if self.discriminator is not None:
            disc_path = save_path / "discriminator.pt"
            torch.save(self.discriminator.state_dict(), disc_path)

        # Save training config
        config_path = save_path / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
