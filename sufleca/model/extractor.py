"""The SUFLECA feature extractor: a frozen, hooked DUNE ViT encoder + DPT head."""
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from PIL import Image

from .config import ENCODER_REGISTRY, SUFLECAConfig
from .dpt_head import DPTHead


class TorchvisionImageProcessor:
    """Minimal ImageNet-normalizing processor returning ``{"pixel_values": tensor}``."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self) -> None:
        import torchvision.transforms as T

        self._to_tensor = T.ToTensor()
        self._normalize = T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)

    def __call__(self, images: Image.Image, return_tensors: str = "pt", **kwargs) -> dict:
        """Normalize a single PIL image to a ``(1, 3, H, W)`` pixel-values batch."""
        if isinstance(images, Image.Image):
            tensor = self._normalize(self._to_tensor(images))
            return {"pixel_values": tensor.unsqueeze(0)}
        raise ValueError(f"Expected PIL Image, got {type(images)}")


def _get_nested_attr(obj: Any, dotted_path: str) -> Any:
    """Resolve a dotted attribute path (e.g. ``"backbone.blocks"``) on ``obj``."""
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


class HookedViTEncoderWrapper(nn.Module):
    """Frozen ViT that returns patch tokens captured from selected blocks via hooks."""

    def __init__(
        self,
        vit_model: nn.Module,
        layer_indices: list[int],
        num_special_tokens: int,
        blocks_attr: str = "blocks",
    ):
        super().__init__()
        self.vit = vit_model
        self.layer_indices = layer_indices
        self.num_special_tokens = num_special_tokens
        self._captured: dict[int, torch.Tensor] = {}

        blocks = _get_nested_attr(vit_model, blocks_attr)[0]
        for idx in layer_indices:
            blocks[idx].register_forward_hook(self._make_hook(idx))

        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

    def _make_hook(self, idx: int) -> Callable:
        """Build a forward hook that stashes block ``idx``'s output token tensor."""
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            self._captured[idx] = out

        return hook

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """Run the ViT and return patch tokens (special tokens stripped) from each
        hooked layer."""
        self._captured = {}
        self.vit(pixel_values)

        result = []
        for idx in self.layer_indices:
            tokens = self._captured[idx]
            tokens = tokens[:, self.num_special_tokens :, :]
            result.append(tokens)
        return result

    def train(self, mode: bool = True) -> "HookedViTEncoderWrapper":
        """Keep the wrapped ViT frozen in eval mode regardless of ``mode``."""
        super().train(False)
        self.vit.eval()
        return self


def build_encoder(config: SUFLECAConfig) -> nn.Module:
    """Load the configured DUNE ViT from torch hub and wrap it for hooked extraction."""
    enc_cfg = ENCODER_REGISTRY[config.encoder]
    if enc_cfg["kind"] != "dune":
        raise ValueError("SUFLECA supports DUNE encoders only.")
    dune_encoder = torch.hub.load(enc_cfg["hub_repo"], enc_cfg["hub_model"])
    return HookedViTEncoderWrapper(
        vit_model=dune_encoder,
        layer_indices=config.encoder_layer_indices,
        num_special_tokens=1 + config.num_register_tokens,
        blocks_attr=enc_cfg["blocks_attr"],
    )


class SUFLECAFeatureExtractor(nn.Module):
    """Frozen encoder plus SUFLECA DPT head.

    Forward returns only dense DPT features with shape ``(B, C, H, W)``.
    """

    def __init__(self, config: SUFLECAConfig):
        super().__init__()
        self.config = config
        self.encoder = build_encoder(config)
        self.dpt_head = DPTHead(config)

    @staticmethod
    def get_image_processor(config: SUFLECAConfig) -> TorchvisionImageProcessor:
        """Return the image processor matching this model's preprocessing."""
        return TorchvisionImageProcessor()

    def _extract_multi_layer_patch_tokens(
        self, pixel_values: torch.Tensor
    ) -> tuple[list[torch.Tensor], int, int]:
        """Return the hooked layers' patch tokens plus the patch-grid ``(h, w)``."""
        H, W = pixel_values.shape[2], pixel_values.shape[3]
        h = H // self.config.patch_size
        w = W // self.config.patch_size
        return self.encoder(pixel_values), h, w

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode an image batch and decode dense DPT features ``(B, C, H, W)``."""
        multi_layer_tokens, h, w = self._extract_multi_layer_patch_tokens(pixel_values)
        return self.dpt_head(multi_layer_tokens, h, w)
