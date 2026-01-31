"""
Core VAE components: Encoder, Decoder, and DiagonalGaussianDistribution.
Based on diffusers implementation for Stable Diffusion 1.5 compatibility.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderOutput:
    """
    Output of encoding method.

    Args:
        latent: The encoded latent tensor of shape (batch_size, num_channels, latent_height, latent_width).
    """
    latent: torch.Tensor


@dataclass
class DecoderOutput:
    """
    Output of decoding method.

    Args:
        sample: The decoded output sample from the last layer of the model.
    """
    sample: torch.Tensor


class Upsample2D(nn.Module):
    """
    2D upsampling layer with optional convolution.

    Args:
        channels: Number of input/output channels.
        use_conv: Whether to use a convolution after upsampling.
        out_channels: Number of output channels (defaults to input channels).
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        
        if use_conv:
            self.conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=3,
                padding=padding,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample input tensor by factor of 2."""
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample2D(nn.Module):
    """
    2D downsampling layer with strided convolution.

    Args:
        channels: Number of input/output channels.
        use_conv: Whether to use a strided convolution for downsampling.
        out_channels: Number of output channels (defaults to input channels).
        padding: Padding for the convolution.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=3,
                stride=2,
                padding=padding,
            )
        else:
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample input tensor by factor of 2."""
        return self.conv(x)


class ResnetBlock2D(nn.Module):
    """
    ResNet block with GroupNorm and SiLU activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        groups: Number of groups for GroupNorm.
        eps: Epsilon for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=groups,
            num_channels=self.out_channels,
            eps=eps,
            affine=True,
        )
        self.conv2 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.nonlinearity = nn.SiLU()

        # Skip connection
        if in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.conv_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x

        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)

        return x + self.conv_shortcut(residual)


class AttentionBlock(nn.Module):
    """
    Self-attention block with GroupNorm.

    Args:
        channels: Number of input/output channels.
        num_head_channels: Number of channels per attention head.
        num_groups: Number of groups for GroupNorm.
        eps: Epsilon for GroupNorm.
    """

    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        num_groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = channels // num_head_channels if num_head_channels else 1
        self.head_dim = channels // self.num_heads

        self.group_norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=channels,
            eps=eps,
            affine=True,
        )

        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj_attn = nn.Linear(channels, channels)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with self-attention."""
        residual = x
        batch, channel, height, width = x.shape

        # Normalize and reshape
        x = self.group_norm(x)
        x = x.view(batch, channel, height * width).transpose(1, 2)  # (B, H*W, C)

        # Compute Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape for multi-head attention
        q = q.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, -1, channel)

        # Project and reshape
        out = self.proj_attn(out)
        out = out.transpose(1, 2).view(batch, channel, height, width)

        return out + residual


class DownEncoderBlock2D(nn.Module):
    """
    Encoder block with ResNet blocks and optional downsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_layers: Number of ResNet blocks.
        add_downsample: Whether to add a downsampling layer.
        resnet_groups: Number of groups for GroupNorm in ResNet blocks.
        resnet_eps: Epsilon for GroupNorm in ResNet blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        add_downsample: bool = True,
        resnet_groups: int = 32,
        resnet_eps: float = 1e-6,
        downsample_padding: int = 0,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                )
            )

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=downsample_padding,
                )
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet blocks and optional downsampling."""
        for resnet in self.resnets:
            x = resnet(x)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = F.pad(x, (0, 1, 0, 1))  # Asymmetric padding
                x = downsampler(x)

        return x


class UpDecoderBlock2D(nn.Module):
    """
    Decoder block with ResNet blocks and optional upsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_layers: Number of ResNet blocks.
        add_upsample: Whether to add an upsampling layer.
        resnet_groups: Number of groups for GroupNorm in ResNet blocks.
        resnet_eps: Epsilon for GroupNorm in ResNet blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        add_upsample: bool = True,
        resnet_groups: int = 32,
        resnet_eps: float = 1e-6,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                )
            )

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode="nearest"),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                )
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet blocks and optional upsampling."""
        for resnet in self.resnets:
            x = resnet(x)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)

        return x


class UNetMidBlock2D(nn.Module):
    """
    Middle block of the UNet with attention and ResNet blocks.

    Args:
        in_channels: Number of input channels.
        resnet_groups: Number of groups for GroupNorm.
        resnet_eps: Epsilon for GroupNorm.
        add_attention: Whether to add attention blocks.
        attention_head_dim: Dimension of attention heads.
    """

    def __init__(
        self,
        in_channels: int,
        resnet_groups: int = 32,
        resnet_eps: float = 1e-6,
        add_attention: bool = True,
        attention_head_dim: Optional[int] = None,
    ):
        super().__init__()
        attention_head_dim = attention_head_dim or in_channels

        # First ResNet block
        self.resnets = nn.ModuleList([
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=resnet_groups,
                eps=resnet_eps,
            ),
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                groups=resnet_groups,
                eps=resnet_eps,
            ),
        ])

        # Attention block
        self.attentions = nn.ModuleList()
        if add_attention:
            self.attentions.append(
                AttentionBlock(
                    channels=in_channels,
                    num_head_channels=attention_head_dim,
                    num_groups=resnet_groups,
                    eps=resnet_eps,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through middle block."""
        x = self.resnets[0](x)

        for attn in self.attentions:
            x = attn(x)

        x = self.resnets[1](x)

        return x


class Encoder(nn.Module):
    """
    VAE Encoder that encodes images into latent representations.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output latent channels.
        down_block_types: Types of down blocks to use.
        block_out_channels: Output channels for each block.
        layers_per_block: Number of ResNet layers per block.
        norm_num_groups: Number of groups for GroupNorm.
        act_fn: Activation function name.
        double_z: Whether to output double channels for mean and variance.
        mid_block_add_attention: Whether to add attention in mid block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        # Input convolution
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Down blocks
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]

        for i, _ in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block,
                add_downsample=not is_final_block,
                resnet_groups=norm_num_groups,
                resnet_eps=1e-6,
                downsample_padding=0,
            )
            self.down_blocks.append(down_block)

        # Mid block
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            resnet_eps=1e-6,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
        )

        # Output layers
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(
            block_out_channels[-1],
            conv_out_channels,
            kernel_size=3,
            padding=1,
        )

        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input image into latent representation.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Latent tensor of shape (B, 2*latent_channels, H/8, W/8) if double_z else (B, latent_channels, H/8, W/8).
        """
        x = self.conv_in(x)

        # Down blocks
        for down_block in self.down_blocks:
            x = down_block(x)

        # Mid block
        x = self.mid_block(x)

        # Output
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    """
    VAE Decoder that decodes latent representations into images.

    Args:
        in_channels: Number of input latent channels.
        out_channels: Number of output image channels.
        up_block_types: Types of up blocks to use.
        block_out_channels: Output channels for each block.
        layers_per_block: Number of ResNet layers per block.
        norm_num_groups: Number of groups for GroupNorm.
        act_fn: Activation function name.
        mid_block_add_attention: Whether to add attention in mid block.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        # Input convolution
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Mid block
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            resnet_eps=1e-6,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
        )

        # Up blocks
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, _ in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            up_block = UpDecoderBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block + 1,
                add_upsample=not is_final_block,
                resnet_groups=norm_num_groups,
                resnet_eps=1e-6,
            )
            self.up_blocks.append(up_block)

        # Output layers
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=1e-6,
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.gradient_checkpointing = False

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation into image.

        Args:
            z: Latent tensor of shape (B, latent_channels, H, W).

        Returns:
            Reconstructed image tensor of shape (B, C, H*8, W*8).
        """
        z = self.conv_in(z)

        # Mid block
        z = self.mid_block(z)

        # Up blocks
        for up_block in self.up_blocks:
            z = up_block(z)

        # Output
        z = self.conv_norm_out(z)
        z = self.conv_act(z)
        z = self.conv_out(z)

        return z


class DiagonalGaussianDistribution:
    """
    Diagonal Gaussian distribution for VAE latent space.
    Supports sampling with reparameterization trick and KL divergence computation.

    Args:
        parameters: Tensor containing concatenated mean and log variance.
        deterministic: If True, use mode instead of sampling.
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean,
                device=self.parameters.device,
                dtype=self.parameters.dtype,
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Sample from the distribution using reparameterization trick.

        Args:
            generator: Optional random generator for reproducibility.

        Returns:
            Sampled latent tensor.
        """
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.mean + self.std * noise

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> torch.Tensor:
        """
        Compute KL divergence against a prior (standard normal if other is None).

        Args:
            other: Optional other distribution for KL divergence.

        Returns:
            KL divergence tensor.
        """
        if self.deterministic:
            return torch.tensor([0.0], device=self.parameters.device)

        if other is None:
            # KL divergence against standard normal N(0, 1)
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        else:
            # KL divergence against other Gaussian
            return 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=[1, 2, 3],
            )

    def nll(
        self,
        sample: torch.Tensor,
        dims: Tuple[int, ...] = (1, 2, 3),
    ) -> torch.Tensor:
        """
        Compute negative log likelihood of a sample.

        Args:
            sample: Sample tensor.
            dims: Dimensions to sum over.

        Returns:
            Negative log likelihood tensor.
        """
        if self.deterministic:
            return torch.tensor([0.0], device=self.parameters.device)

        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        """Return the mode (mean) of the distribution."""
        return self.mean
