"""
Metrics for evaluating VAE reconstruction quality.
Includes PSNR, SSIM, LPIPS, and FID computation.
"""

from typing import Optional, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips

    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image import PeakSignalNoiseRatio

    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False


class PSNR(nn.Module):
    """
    Peak Signal-to-Noise Ratio metric.

    Args:
        data_range: Range of input data (e.g., 1.0 for [0,1] or 2.0 for [-1,1]).
    """

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range

    def forward(
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
            return torch.tensor(float("inf"))
        psnr = 20 * torch.log10(self.data_range / torch.sqrt(mse))
        return psnr


class SSIM(nn.Module):
    """
    Structural Similarity Index Measure.

    Args:
        data_range: Range of input data.
        window_size: Size of the sliding window.
        channel: Number of channels.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        window_size: int = 11,
        channel: int = 3,
    ):
        super().__init__()
        self.data_range = data_range
        self.window_size = window_size
        self.channel = channel

        # Create Gaussian window
        self.window = self._create_window(window_size, channel)

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM."""
        sigma = 1.5
        gauss = torch.Tensor(
            [
                math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        gauss = gauss / gauss.sum()

        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(
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
        channel = pred.size(1)
        window = self.window.to(pred.device).type_as(pred)

        if channel != self.channel:
            window = self._create_window(self.window_size, channel).to(pred.device).type_as(pred)

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(target * target, window, padding=self.window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim_map.mean()


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
        if HAS_TORCHMETRICS:
            self.psnr = PeakSignalNoiseRatio(data_range=data_range)
        else:
            self.psnr = PSNR(data_range=data_range)

        # SSIM
        if HAS_TORCHMETRICS:
            self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
        else:
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
