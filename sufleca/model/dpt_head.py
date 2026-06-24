"""
DPT (Dense Prediction Transformer) Head.

Takes multi-layer ViT features from the frozen perception encoder and produces dense per-pixel
feature maps via reassembly + progressive fusion.

Architecture overview:
    1. Extract patch tokens from 4 intermediate ViT layers
    2. Reshape each to spatial feature maps (B, C, H_p, W_p)
    3. Project each to a common hidden dimension
    4. Progressively upsample and fuse (coarse -> fine)
    5. Final projection to output feature dimension

Reference: Ranftl et al., "Vision Transformers for Dense Prediction", ICCV 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SUFLECAConfig


class Reassemble(nn.Module):
    """
    Takes a sequence of patch tokens from one ViT layer and reshapes + projects
    them into a 2D spatial feature map at a desired scale.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, C) patch tokens from one ViT layer
            h, w: spatial dimensions of the patch grid
        Returns:
            (B, out_dim, H', W') spatial feature map
        """
        B, N, C = tokens.shape
        # Reshape to spatial: (B, C, h, w)
        x = tokens.transpose(1, 2).reshape(B, C, h, w)
        # Project channels
        x = self.proj(x)
        # Scale spatial resolution
        if self.scale_factor != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode="bilinear",
                align_corners=True,
            )
        return x


class FusionBlock(nn.Module):
    """
    Residual convolutional fusion block used in DPT.
    Refines features after merging two scales.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.res_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.res_conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        assert dim % 32 == 0, "GroupNorm num_groups must divide the number of channels"
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=dim)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two norm+conv+ReLU layers and add the input residual."""
        residual = x
        x = F.relu(self.norm1(self.res_conv1(x)))
        x = F.relu(self.norm2(self.res_conv2(x)))
        return x + residual


class DPTHead(nn.Module):
    """
    DPT-style head that fuses multi-layer ViT features into a dense
    per-pixel feature map.

    Uses 4 reassembly stages with progressive upsampling and fusion.
    Outputs a feature map at 4x the patch grid resolution
    (e.g., for 448px input with patch_size=14: 32x32 patches -> 128x128 feature map).
    """

    def __init__(self, config: SUFLECAConfig):
        super().__init__()
        embed_dim = config.encoder_embed_dim
        hidden = config.dpt_hidden_dim

        # Reassemble modules for each of the 4 ViT layers
        # Progressive scale factors: 4x, 2x, 1x, 0.5x relative to patch grid
        # This way the coarsest is 0.5x and finest is 4x the patch grid size
        self.reassemble_layers = nn.ModuleList([
            Reassemble(embed_dim, hidden, scale_factor=4.0),  # finest
            Reassemble(embed_dim, hidden, scale_factor=2.0),
            Reassemble(embed_dim, hidden, scale_factor=1.0),
            Reassemble(embed_dim, hidden, scale_factor=0.5),  # coarsest
        ])

        # Fusion blocks (applied bottom-up: coarse -> fine)
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(hidden) for _ in range(4)
        ])

        # 1x1 projection for combining skip connections
        self.skip_projections = nn.ModuleList([
            nn.Conv2d(hidden, hidden, kernel_size=1) for _ in range(4)
        ])

    def forward(
        self,
        multi_layer_tokens: list[torch.Tensor],
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Args:
            multi_layer_tokens: list of 4 tensors, each (B, N_patches, C),
                                from the 4 selected ViT layers
            h, w: patch grid spatial dimensions (e.g., 28, 28)
        Returns:
            (B, dpt_hidden_dim, H_out, W_out) dense per-pixel features
            where H_out, W_out = 4 * h, 4 * w (e.g., 128x128 for h=w=32)
        """
        assert len(multi_layer_tokens) == 4

        # Reassemble each layer's tokens into spatial feature maps
        reassembled = []
        for i, tokens in enumerate(multi_layer_tokens):
            reassembled.append(self.reassemble_layers[i](tokens, h, w))

        # Progressive fusion: start from coarsest (index 3) and go to finest (index 0)
        # This merges coarse context into finer features
        fused = self.skip_projections[3](reassembled[3])
        fused = self.fusion_blocks[3](fused)

        for i in range(2, -1, -1):  # 2, 1, 0
            # Upsample fused to match the next finer scale
            target_size = reassembled[i].shape[2:]
            fused = F.interpolate(
                fused, size=target_size, mode="bilinear", align_corners=True
            )
            # Add skip connection from reassembled features
            fused = fused + self.skip_projections[i](reassembled[i])
            fused = self.fusion_blocks[i](fused)

        return fused
