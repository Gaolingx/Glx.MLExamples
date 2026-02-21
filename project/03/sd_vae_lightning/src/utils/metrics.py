"""
Metrics for evaluating VAE reconstruction quality.
Includes PSNR, SSIM, PSIM, and rFID computation.
"""

import torch

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
    from torchmetrics.image import PeakSignalNoiseRatio

    HAS_TORCHMETRICS_PSNR = True
except ImportError:
    HAS_TORCHMETRICS_PSNR = False

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
        if not HAS_TORCHMETRICS_PSNR:
            raise ImportError(
                "torchmetrics is required for PSNR computation. "
                "Install with: pip install torchmetrics"
            )
        self.data_range = data_range
        self._metric = PeakSignalNoiseRatio(data_range=data_range)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute PSNR between prediction and target.

        Args:
            pred: Predicted image tensor.
            target: Target image tensor.

        Returns:
            PSNR value in dB.
        """
        self._metric = self._metric.to(pred.device)
        return self._metric(pred, target)


class SSIM:
    """
    Structural Similarity Index Measure.

    Args:
        data_range: Range of input data.
        window_size: Size of the sliding window (kernel_size).
    """

    def __init__(self, data_range: float = 1.0, window_size: int = 11):
        if not HAS_TORCHMETRICS_SSIM:
            raise ImportError(
                "torchmetrics is required for SSIM computation. "
                "Install with: pip install torchmetrics"
            )
        self.data_range = data_range
        self.window_size = window_size
        self._metric = StructuralSimilarityIndexMeasure(
            data_range=data_range,
            kernel_size=window_size,
        )

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM between prediction and target.

        Args:
            pred: Predicted image tensor.
            target: Target image tensor.

        Returns:
            SSIM value.
        """
        self._metric = self._metric.to(pred.device)
        return self._metric(pred, target)


class rFID:
    """
    Reconstruction FID (rFID) metric.

    Computes Frechet Inception Distance between original and reconstructed
    images to measure how well the VAE preserves the image distribution.

    Args:
        feature_dim: Feature dimension for Inception network (64, 192, 768, or 2048).
        reset_real_features: Whether to reset real features after each compute.
    """

    def __init__(self, feature_dim: int = 2048, reset_real_features: bool = True):
        if not HAS_FID:
            raise ImportError(
                "torch-fidelity is required for rFID computation. "
                "Install with: pip install torch-fidelity"
            )
        self.feature_dim = feature_dim
        self.reset_real_features = reset_real_features
        self.fid = FrechetInceptionDistance(
            feature=feature_dim,
            reset_real_features=reset_real_features,
            normalize=True,
        )

    def update(self, real: torch.Tensor, fake: torch.Tensor) -> None:
        """
        Update FID statistics with a batch of real and reconstructed images.

        Args:
            real: Original images in range [-1, 1].
            fake: Reconstructed images in range [-1, 1].
        """
        if next(self.fid.parameters()).device != real.device:
            self.fid = self.fid.to(real.device)

        real = torch.clamp((real + 1) / 2, 0, 1)
        fake = torch.clamp((fake + 1) / 2, 0, 1)

        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)

    def compute(self) -> torch.Tensor:
        """
        Compute rFID score.

        Returns:
            rFID score (lower is better).
        """
        return self.fid.compute()

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.fid.reset()


class PSIM:
    """
    Perceptual Similarity Metric (PSIM).
    Here we use LPIPS distance and convert to similarity: sim = 1 / (1 + d).
    Higher is better, range (0, 1].

    Args:
        net: base/trunk networks
    """

    def __init__(self, net: str = "vgg"):
        if not HAS_LPIPS:
            raise ImportError(
                "lpips is required for PSIM computation. "
                "Install with: pip install lpips"
            )
        self.net = net
        self.metric = lpips.LPIPS(net=net)
        self.metric.eval()
        for p in self.metric.parameters():
            p.requires_grad = False

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Reconstructed images in [-1, 1], shape [B, C, H, W]
            target: Ground-truth images in [-1, 1], shape [B, C, H, W]

        Returns:
            PSIM similarity scalar (higher is better)
        """
        if next(self.metric.parameters()).device != pred.device:
            self.metric = self.metric.to(pred.device)

        d = self.metric(pred, target).mean()
        sim = 1.0 / (1.0 + d)
        return sim
