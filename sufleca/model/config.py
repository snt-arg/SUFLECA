"""Configuration for the SUFLECA feature extractor and its DUNE ViT encoders."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Per-encoder hyperparameters (hub source, patch/embed dims, DPT tap layers).
ENCODER_REGISTRY: dict[str, dict[str, Any]] = {
    "dune_vitb14_448": {
        "kind": "dune",
        "hub_repo": "naver/dune",
        "hub_model": "dune_vitbase_14_448_encoder",
        "patch_size": 14,
        "embed_dim": 768,
        "num_layers": 12,
        "num_register_tokens": 4,
        "layer_indices": [2, 5, 8, 11],
        "default_input_size": 448,
        "blocks_attr": "blocks",
    },
    "dune_vits14_448": {
        "kind": "dune",
        "hub_repo": "naver/dune",
        "hub_model": "dune_vitsmall_14_448_encoder",
        "patch_size": 14,
        "embed_dim": 384,
        "num_layers": 12,
        "num_register_tokens": 4,
        "layer_indices": [2, 5, 8, 11],
        "default_input_size": 448,
        "blocks_attr": "blocks",
    },
}


@dataclass
class SUFLECAConfig:
    """Feature-extractor config. ``encoder`` selects an :data:`ENCODER_REGISTRY`
    entry; the remaining encoder fields are filled from it in ``__post_init__``."""

    encoder: str = "dune_vitb14_448"
    input_size: int = 0
    dpt_hidden_dim: int = 256

    encoder_name: str = field(init=False, default="")
    patch_size: int = field(init=False, default=14)
    num_register_tokens: int = field(init=False, default=4)
    encoder_embed_dim: int = field(init=False, default=768)
    encoder_num_layers: int = field(init=False, default=12)
    encoder_layer_indices: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Populate encoder fields from the registry and validate ``input_size``."""
        if self.encoder not in ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown encoder '{self.encoder}'. "
                f"Valid options: {list(ENCODER_REGISTRY.keys())}"
            )
        enc = ENCODER_REGISTRY[self.encoder]
        self.patch_size = enc["patch_size"]
        self.encoder_embed_dim = enc["embed_dim"]
        self.encoder_num_layers = enc["num_layers"]
        self.num_register_tokens = enc["num_register_tokens"]
        self.encoder_layer_indices = list(enc["layer_indices"])
        self.encoder_name = enc["hub_model"]
        if self.input_size == 0:
            self.input_size = enc["default_input_size"]
        if self.input_size % self.patch_size != 0:
            raise ValueError(
                f"input_size={self.input_size} must be a multiple of "
                f"patch_size={self.patch_size} for encoder '{self.encoder}'"
            )


def sufleca_config_from_checkpoint(raw: dict[str, Any]) -> SUFLECAConfig:
    """Build a :class:`SUFLECAConfig` from a checkpoint's raw config dict,
    keeping only the constructor (``init=True``) fields."""
    fields = {
        name
        for name, field_info in SUFLECAConfig.__dataclass_fields__.items()
        if field_info.init
    }
    return SUFLECAConfig(**{k: v for k, v in raw.items() if k in fields})
