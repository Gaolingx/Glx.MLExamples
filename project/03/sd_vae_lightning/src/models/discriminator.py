import torch
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator for adversarial training.

    Args:
        input_nc: Number of input channels.
        ndf: Base number of discriminator filters.
        n_layers: Number of layers in discriminator.
        norm_type: Normalization type. 'spectral', 'spectral_group', 'batch', or 'none'.
        num_groups: Number of groups for GroupNorm (default: 32).
    """

    def __init__(
            self,
            input_nc: int = 3,
            ndf: int = 64,
            n_layers: int = 3,
            norm_type: str = "spectral_group",
            num_groups: int = 32,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.num_groups = num_groups

        use_spectral = norm_type in ("spectral", "spectral_group")
        use_group = norm_type == "spectral_group"
        use_batch = norm_type == "batch"

        def maybe_spectral(layer):
            return nn.utils.spectral_norm(layer) if use_spectral else layer

        def get_norm_layer(num_features):
            if use_group:
                # Ensure num_groups doesn't exceed num_features
                groups = min(num_groups, num_features)
                return nn.GroupNorm(groups, num_features)
            elif use_batch:
                return nn.BatchNorm2d(num_features)
            else:
                # 'spectral' alone or 'none' — no extra normalization
                return nn.Identity()

        # --- First layer: no normalization, just conv + activation ---
        layers = [
            maybe_spectral(
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, True),
        ]

        # --- Intermediate layers ---
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                maybe_spectral(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    )
                ),
                get_norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        # --- Penultimate layer (stride=1) ---
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            maybe_spectral(
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            ),
            get_norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # --- Final layer: 1-channel output (patch map) ---
        layers += [
            maybe_spectral(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
            ),
        ]

        self.model = nn.Sequential(*layers)

        # Initialize weights (skip spectral-normed layers — they self-regulate)
        if not use_spectral:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find("GroupNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
