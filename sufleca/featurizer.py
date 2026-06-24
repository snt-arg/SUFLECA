"""Load SUFLECA DPT feature extractors and run dense feature extraction."""
from __future__ import annotations

import os
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from .model import SUFLECAFeatureExtractor, sufleca_config_from_checkpoint


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINTS = {
    "sufleca": os.path.join(_ROOT, "checkpoints", "sufleca", "best.pt"),
    "sufleca-wo-scannet": os.path.join(_ROOT, "checkpoints", "sufleca-wo-scannet", "best.pt"),
    "sufleca-small": os.path.join(_ROOT, "checkpoints", "sufleca-small", "best.pt"),
}

PUBLIC_CHECKPOINTS = ("sufleca", "sufleca-small", "sufleca-wo-scannet")


def resolve_checkpoint(name_or_path: str = "sufleca") -> str:
    """Map a known checkpoint alias to its path, or pass an explicit path through."""
    return CHECKPOINTS.get(name_or_path, name_or_path)


def load_sufleca_model(checkpoint: str = "sufleca", device: str = "cuda") -> dict[str, Any]:
    """Load a SUFLECA dense feature extractor from a checkpoint.

    :param checkpoint: a known alias (see :data:`CHECKPOINTS`) or a ``.pt`` path.
    :param device: requested device; falls back to CPU when CUDA is unavailable.
    :returns: a bundle dict with the eval-mode ``model``, its ``image_processor``,
        ``config``, the resolved ``checkpoint`` path, and the ``device``.
    """
    device = device if torch.cuda.is_available() else "cpu"
    checkpoint_path = resolve_checkpoint(checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = sufleca_config_from_checkpoint(ckpt["config"])

    model = SUFLECAFeatureExtractor(config).to(device)
    model.dpt_head.load_state_dict(ckpt["dpt_head"])
    model.eval()

    return {
        "model": model,
        "image_processor": SUFLECAFeatureExtractor.get_image_processor(config),
        "config": config,
        "checkpoint": checkpoint_path,
        "device": device,
    }


@torch.inference_mode()
def extract_sufleca_features(
    pil_images: Image.Image | Sequence[Image.Image],
    model_pack: dict[str, Any],
    resize_to: int | tuple[int, int] | None = (360, 360),
) -> torch.Tensor:
    """Return L2-normalized dense SUFLECA DPT features as ``(B, H, W, C)``.

    :param pil_images: a single PIL image or a sequence of them.
    :param model_pack: bundle from :func:`load_sufleca_model`.
    :param resize_to: bilinearly resize the feature map to this size (and
        re-normalize) before returning; ``None`` keeps the native resolution.
    """
    model = model_pack["model"]
    image_processor = model_pack["image_processor"]
    device = model_pack["device"]

    if not isinstance(pil_images, (list, tuple)):
        pil_images = [pil_images]

    pixel_values = torch.cat(
        [
            image_processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"]
            for img in pil_images
        ],
        dim=0,
    ).to(device)

    feat = F.normalize(model(pixel_values), p=2, dim=1)
    if resize_to is not None:
        feat = F.interpolate(feat, size=resize_to, mode="bilinear", align_corners=False)
        feat = F.normalize(feat, p=2, dim=1)
    return feat.permute(0, 2, 3, 1)
