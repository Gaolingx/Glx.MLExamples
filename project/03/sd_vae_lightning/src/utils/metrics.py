"""
Metrics for evaluating VAE reconstruction quality.
Includes PSNR, SSIM, LPIPS, and rFID computation.
"""

from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips

    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from torchmetrics.image.fid import FrechetInceptionDistance

    HAS_FID = True
except ImportError:
    HAS_FID = False

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    HAS_TORCHMETRICS_SSIM = True
except ImportError:
    HAS_TORCHMETRICS_SSIM = False


class PSNR:
    """
    Peak Signal-to-Noise Ratio metric.

    Args:
        data_range: Range of input data (e.g., 1.0 for [0,1] or 2.0 for [-1,1]).
    """

    def __init__(self, data_range: float = 1.0):
        self.data_range = data_range

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PSNR between prediction and target.

        Args:
            pred: Predicted image tensor.
            target: Target image tensor.

        Returns:
            PSNR value in dB.
        """
        mse = F.mse_loss(pred, target, reduction="mean")
        if mse == 0:
            return torch.tensor(float("inf"), device=pred.device)
        psnr = 20 * torch.log10(self.data_range / torch.sqrt(mse))
        return psnr


class SSIM:
    """
    Structural Similarity Index Measure.

    Args:
        data_range: Range of input data.
        window_size: Size of the sliding window (kernel_size).
        channel: Number of channels (kept for backward compatibility).
    """

    def __init__(
        self,
        data_range: float = 1.0,
        window_size: int = 11,
        channel: int = 3,
    ):
        if not HAS_TORCHMETRICS_SSIM:
            raise ImportError(
                "torchmetrics is required for SSIM computation. "
                "Install with: pip install torchmetrics"
            )
        self.data_range = data_range
        self.window_size = window_size
        self.channel = channel  # Kept for backward compatibility
        self._ssim_metric: Optional[StructuralSimilarityIndexMeasure] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

    def _get_metric(
        self, device: torch.device, dtype: torch.dtype
    ) -> StructuralSimilarityIndexMeasure:
        """Get cached metric or create new one if needed."""
        if (
            self._ssim_metric is None
            or self._device != device
            or self._dtype != dtype
        ):
            self._ssim_metric = StructuralSimilarityIndexMeasure(
                data_range=self.data_range,
                kernel_size=self.window_size,
            ).to(device=device, dtype=dtype)
            self._device = device
            self._dtype = dtype
        return self._ssim_metric

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM between prediction and target.

        Args:
            pred: Predicted image tensor.
            target: Target image tensor.

        Returns:
            SSIM value.
        """
        metric = self._get_metric(pred.device, pred.dtype)
        return metric(pred, target)


class rFID:
    """
    Reconstruction FID (rFID) metric.
    
    Computes Frechet Inception Distance between original and reconstructed
    images to measure how well the VAE preserves the image distribution.
    
    Args:
        feature_dim: Feature dimension for Inception network (64, 192, 768, or 2048).
        reset_real_features: Whether to reset real features after each compute.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        reset_real_features: bool = True,
    ):
        self.feature_dim = feature_dim
        self.reset_real_features = reset_real_features
        self.fid = None
        
        if HAS_FID:
            try:
                self.fid = FrechetInceptionDistance(
                    feature=feature_dim,
                    reset_real_features=reset_real_features,
                    normalize=True,
                )
            except ModuleNotFoundError:
                # torch-fidelity not installed
                print(
                    "Warning: torch-fidelity not installed, rFID metric disabled. "
                    "Install with: pip install torch-fidelity"
                )
                self.fid = None
            
    def update(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> None:
        """
        Update FID statistics with a batch of real and reconstructed images.
        
        Args:
            real: Original images in range [-1, 1].
            fake: Reconstructed images in range [-1, 1].
        """
        if self.fid is None:
            return
        
        # Move FID to correct device if needed
        if next(self.fid.parameters()).device != real.device:
            self.fid = self.fid.to(real.device)
            
        # Normalize from [-1, 1] to [0, 1] and clamp
        real = torch.clamp((real + 1) / 2, 0, 1)
        fake = torch.clamp((fake + 1) / 2, 0, 1)
        
        # Update FID metric
        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)
        
    def compute(self) -> torch.Tensor:
        """
        Compute rFID score.
        
        Returns:
            rFID score (lower is better).
        """
        if self.fid is None:
            return torch.tensor(float("nan"))
        return self.fid.compute()
        
    def reset(self) -> None:
        """Reset accumulated statistics."""
        if self.fid is not None:
            self.fid.reset()


class VAEMetrics(nn.Module):
    """
    Collection of metrics for VAE evaluation.

    Args:
        data_range: Range of input data.
        use_lpips: Whether to compute LPIPS.
        lpips_net: Network for LPIPS ('alex', 'vgg', 'squeeze').
    """

    def __init__(
        self,
        data_range: float = 2.0,  # [-1, 1] range
        use_lpips: bool = True,
        lpips_net: str = "vgg",
    ):
        super().__init__()
        self.data_range = data_range

        # PSNR
        self.psnr = PSNR(data_range=data_range)

        # SSIM
        self.ssim = SSIM(data_range=data_range)

        # LPIPS
        self.use_lpips = use_lpips and HAS_LPIPS
        if self.use_lpips:
            self.lpips = lpips.LPIPS(net=lpips_net)
            self.lpips.eval()
            for param in self.lpips.parameters():
                param.requires_grad = False

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all metrics.

        Args:
            pred: Predicted image tensor.
            target: Target image tensor.

        Returns:
            Dictionary of metric values.
        """
        metrics = {}

        # PSNR
        with torch.no_grad():
            metrics["psnr"] = self.psnr(pred, target)

            # SSIM
            metrics["ssim"] = self.ssim(pred, target)

            # LPIPS
            if self.use_lpips:
                # LPIPS expects values in [-1, 1]
                if self.data_range != 2.0:
                    pred_normalized = pred * 2 - 1
                    target_normalized = target * 2 - 1
                else:
                    pred_normalized = pred
                    target_normalized = target
                metrics["lpips"] = self.lpips(pred_normalized, target_normalized).mean()

        return metrics

    def compute_reconstruction_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[float, float, Optional[float]]:
        """
        Compute reconstruction metrics and return as Python floats.

        Args:
            pred: Predicted image tensor.
            target: Target image tensor.

        Returns:
            Tuple of (PSNR, SSIM, LPIPS).
        """
        metrics = self(pred, target)
        psnr = metrics["psnr"].item()
        ssim = metrics["ssim"].item()
        lpips_val = metrics.get("lpips")
        if lpips_val is not None:
            lpips_val = lpips_val.item()
        return psnr, ssim, lpips_val
