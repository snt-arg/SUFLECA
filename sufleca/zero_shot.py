"""Zero-shot CAD retrieval: DINOv3 coarse-to-fine search over a template index.

The object label gates which ShapeNet synsets are searched (via the SV
vocabulary); a masked DINOv3 embedding then ranks candidate CADs coarsely by a
single pooled descriptor and refines the top-``k`` by dense patch similarity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DENSE_CACHE_CAP = 6000


@dataclass(frozen=True)
class ZeroShotVocabulary:
    """SV zero-shot vocabulary: synset id->name, label->synsets, retrieval defaults."""

    name: str
    synsets: dict[str, str]
    vocabulary: dict[str, list[str]]
    retrieval: dict[str, Any]


@dataclass(frozen=True)
class ZeroShotRetrievalResult:
    """A retrieved CAD: its synset/model, the best-matching view, similarity
    scores, and the render-view paths feeding alignment."""

    label: str
    synset: str
    synset_name: str
    model_id: str
    view_id: str
    score: float
    coarse_score: float
    render_paths: list[str]


def load_zero_shot_vocabulary(path: str | Path) -> ZeroShotVocabulary:
    """Load the SV zero-shot vocabulary and DINOv3 retrieval defaults."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    synsets = {str(k): str(v) for k, v in cfg["synsets"].items()}
    vocabulary = {str(k).lower(): [str(s) for s in v] for k, v in cfg["vocabulary"].items()}
    for label, candidates in vocabulary.items():
        missing = [s for s in candidates if s not in synsets]
        if missing:
            raise ValueError(
                f"{path}: vocabulary label '{label}' maps to undeclared synsets {missing}"
            )
    retrieval = dict(cfg.get("retrieval", {}))
    if retrieval.get("featurizer", "dinov3") != "dinov3":
        raise ValueError("zero-shot retrieval only supports the DINOv3 featurizer")
    return ZeroShotVocabulary(
        name=str(cfg.get("name", "sv")),
        synsets=synsets,
        vocabulary=vocabulary,
        retrieval=retrieval,
    )


def candidate_synsets(label: str, vocabulary: ZeroShotVocabulary) -> list[str]:
    """Map a detected object label to SV vocabulary synsets."""
    return [
        synset
        for synset in vocabulary.vocabulary.get(label.lower(), [])
        if synset in vocabulary.synsets
    ]


def load_dinov3_retriever(
    device: str = "cuda",
    model_id: str = DINOV3_MODEL_ID,
) -> dict[str, Any]:
    """Load the DINOv3 dense patch featurizer used by zero-shot CAD retrieval."""
    from transformers import AutoImageProcessor, AutoModel

    device = device if torch.cuda.is_available() else "cpu"
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
    except Exception as exc:
        raise RuntimeError(
            f"could not load DINOv3 model {model_id!r}. The official Meta "
            "DINOv3 repositories are gated: accept the model terms and run "
            "`hf auth login`, or pre-populate the Hugging Face cache"
        ) from exc
    n_prefix = 1 + int(getattr(model.config, "num_register_tokens", 0))
    mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
    return {
        "model": model,
        "mean": mean.to(device),
        "std": std.to(device),
        "n_prefix": n_prefix,
        "patch_size": int(getattr(model.config, "patch_size", 16)),
        "device": device,
        "model_id": model_id,
    }


@torch.inference_mode()
def featurize_dinov3_dense(
    pil_images: list[Image.Image],
    retriever: dict[str, Any],
    size: int = 512,
    batch_size: int = 8,
) -> torch.Tensor:
    """Return L2-normalized DINOv3 patch features as ``(N, g, g, D)``."""
    patch_size = int(retriever["patch_size"])
    if size % patch_size != 0:
        raise ValueError(f"image size {size} is not a multiple of patch size {patch_size}")
    g = size // patch_size
    chunks = []
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start : start + batch_size]
        arr = np.stack(
            [
                np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR))
                for image in batch
            ]
        )
        tensor = (
            torch.from_numpy(arr)
            .to(retriever["device"])
            .float()
            .permute(0, 3, 1, 2)
            / 255.0
        )
        tensor = (tensor - retriever["mean"]) / retriever["std"]
        out = retriever["model"](tensor).last_hidden_state
        patches = out[:, retriever["n_prefix"] :, :].reshape(len(batch), g, g, -1)
        chunks.append(F.normalize(patches, p=2, dim=-1))
    return torch.cat(chunks, dim=0)


def mask_to_grid(mask: np.ndarray, grid_size: int, frac: float = 0.5) -> np.ndarray:
    """Downsample an image-space mask to a DINOv3 patch occupancy grid."""
    pooled = cv2.resize(mask.astype(np.float32), (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    grid = pooled > frac
    if not grid.any():
        grid = pooled > 0.0
    return grid


def masked_mean_embedding(patches: torch.Tensor, grid: np.ndarray) -> np.ndarray:
    """Mean-pool DINOv3 patch embeddings under a patch mask."""
    selected = patches[torch.from_numpy(grid).to(patches.device)]
    if selected.shape[0] == 0:
        selected = patches.reshape(-1, patches.shape[-1])
    embedding = selected.mean(dim=0)
    embedding = F.normalize(embedding, p=2, dim=0)
    return embedding.detach().cpu().numpy().astype(np.float32)


def _load_template_synset(template_root: Path, synset: str) -> dict[str, Any] | None:
    """Load one synset's template index (coarse ``singles`` descriptors, the
    ``index`` of (model, view) rows, and the dense-feature dir), or ``None`` if
    its cache is absent."""
    cat_dir = template_root / synset
    singles_path = cat_dir / "singles.npy"
    index_path = cat_dir / "index.json"
    dense_dir = cat_dir / "dense"
    if not (singles_path.exists() and index_path.exists() and dense_dir.is_dir()):
        return None
    meta_path = cat_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("featurizer", "dinov3") != "dinov3":
        raise ValueError(f"{cat_dir} was not built with DINOv3 features")
    return {
        "singles": np.load(singles_path).astype(np.float32),
        "index": json.loads(index_path.read_text()),
        "dense_dir": dense_dir,
        "meta": meta,
    }


def _template_state(
    template_root: str | Path,
    synsets: list[str],
    device: str,
) -> dict[str, Any]:
    """Build retrieval state for the requested synsets: per-synset coarse
    descriptors (on ``device``) and index, plus a shared dense-feature cache."""
    template_root = Path(template_root)
    cats = {}
    for synset in synsets:
        templates = _load_template_synset(template_root, synset)
        if templates is None:
            continue
        singles = templates["singles"]
        index = templates["index"]
        if singles.ndim != 2 or singles.shape[0] == 0:
            raise ValueError(f"{template_root / synset / 'singles.npy'} must be a non-empty 2D array")
        if len(index) != singles.shape[0]:
            raise ValueError(
                f"template index length mismatch for {synset}: "
                f"{len(index)} rows in index.json vs {singles.shape[0]} descriptors"
            )
        cats[synset] = {
            "singles": torch.from_numpy(singles).to(device),
            "index": index,
            "dense_dir": templates["dense_dir"],
            "meta": templates["meta"],
        }
    return {"cats": cats, "cache": {}}


def _retrieve_from_templates(
    src_patches: torch.Tensor,
    src_single: torch.Tensor,
    synsets: list[str],
    state: dict[str, Any],
    top_k: int,
    fine_agg: str,
    device: str,
) -> tuple[str, str, str, float, float] | None:
    """Coarse-to-fine search across synsets: rank templates by the coarse
    descriptor, refine the top-``k`` by dense patch similarity (``fine_agg``),
    and return the best ``(synset, model_id, view_id, fine, coarse)`` or ``None``."""
    best = None
    cache = state["cache"]
    for synset in synsets:
        templates = state["cats"].get(synset)
        if templates is None:
            continue
        singles = templates["singles"]
        index = templates["index"]
        coarse = singles @ src_single
        top_idx = torch.topk(coarse, min(top_k, singles.shape[0])).indices.tolist()

        per_model: dict[str, tuple[str, float, float]] = {}
        for template_idx in top_idx:
            model_id, view_id = index[template_idx]
            cache_key = (synset, template_idx)
            dense = cache.get(cache_key)
            if dense is None:
                if len(cache) >= DENSE_CACHE_CAP:
                    cache.clear()
                dense_path = templates["dense_dir"] / f"{model_id}__{view_id}.npy"
                dense_arr = np.load(dense_path).astype(np.float32)
                dense = torch.from_numpy(dense_arr).to(device)
                cache[cache_key] = dense

            sim = src_patches @ dense.T
            if fine_agg == "symmetric":
                fine = 0.5 * (
                    sim.max(dim=1).values.mean().item()
                    + sim.max(dim=0).values.mean().item()
                )
            elif fine_agg == "maxmean":
                fine = sim.max(dim=1).values.mean().item()
            else:
                raise ValueError("fine_agg must be 'symmetric' or 'maxmean'")
            coarse_score = float(coarse[template_idx].item())
            if model_id not in per_model or fine > per_model[model_id][1]:
                per_model[model_id] = (view_id, fine, coarse_score)

        if not per_model:
            continue
        model_id = max(per_model, key=lambda mid: per_model[mid][1])
        view_id, fine, coarse_score = per_model[model_id]
        if best is None or fine > best[3]:
            best = (synset, model_id, view_id, fine, coarse_score)
    return best


def retrieve_zero_shot_cad(
    color_path: str | Path,
    mask_path: str | Path,
    inst_id: int,
    object_label: str,
    vocabulary: ZeroShotVocabulary,
    template_root: str | Path | None = None,
    render_pool: str | Path | None = None,
    retriever: dict[str, Any] | None = None,
    device: str = "cuda",
    image_size: int | None = None,
    top_k: int | None = None,
    fine_agg: str | None = None,
    mask_frac: float | None = None,
    min_fg_pixels: int | None = None,
) -> ZeroShotRetrievalResult:
    """Retrieve a CAD model from the SV vocabulary using DINOv3 coarse-to-fine search."""
    device = device if torch.cuda.is_available() else "cpu"
    cfg = vocabulary.retrieval
    template_root = Path(template_root or cfg.get("template_root", "data/zero_templates/dinov3"))
    render_pool = Path(render_pool or cfg.get("render_pool", "render_pool"))
    image_size = int(image_size or cfg.get("image_size", 512))
    top_k = int(top_k or cfg.get("top_k", 64))
    fine_agg = str(fine_agg or cfg.get("fine_agg", "symmetric"))
    mask_frac = float(mask_frac if mask_frac is not None else cfg.get("mask_frac", 0.5))
    min_fg_pixels = int(min_fg_pixels or cfg.get("min_fg_pixels", 32))

    synsets = candidate_synsets(object_label, vocabulary)
    if not synsets:
        raise ValueError(f"'{object_label}' is not in the SV zero-shot vocabulary")

    state = _template_state(template_root, synsets, device)
    synsets = [synset for synset in synsets if synset in state["cats"]]
    if not synsets:
        raise FileNotFoundError(f"no DINOv3 template index found for '{object_label}' under {template_root}")

    if retriever is None:
        retriever = load_dinov3_retriever(device=device, model_id=str(cfg.get("model_id", DINOV3_MODEL_ID)))

    for synset in synsets:
        meta = state["cats"][synset]["meta"]
        if meta.get("size") is not None and int(meta["size"]) != image_size:
            raise ValueError(
                f"template index for {synset} was built at size {meta['size']}, "
                f"but retrieval requested {image_size}; rebuild it or use the matching image_size"
            )
        if meta.get("patch_size") is not None and int(meta["patch_size"]) != int(retriever["patch_size"]):
            raise ValueError(f"template index for {synset} uses an incompatible DINOv3 patch size")
        if meta.get("model_id") is not None and str(meta["model_id"]) != str(retriever["model_id"]):
            raise ValueError(
                f"template index for {synset} was built with {meta['model_id']!r}, "
                f"but retrieval loaded {retriever['model_id']!r}"
            )
        if state["cats"][synset]["singles"].shape[1] != int(getattr(retriever["model"].config, "hidden_size")):
            raise ValueError(f"template descriptors for {synset} have an incompatible feature dimension")

    pil = Image.open(color_path).convert("RGB")
    mask = np.array(Image.open(mask_path).convert("L"))
    if mask.shape != (pil.height, pil.width):
        raise ValueError(
            f"mask shape {mask.shape} does not match image shape {(pil.height, pil.width)}"
        )
    fg = mask == int(inst_id)
    if int(fg.sum()) < min_fg_pixels:
        raise ValueError(f"instance {inst_id} in {mask_path} has fewer than {min_fg_pixels} pixels")

    patches = featurize_dinov3_dense([pil], retriever, size=image_size)[0]
    grid_size = image_size // int(retriever["patch_size"])
    grid = mask_to_grid(fg, grid_size, frac=mask_frac)
    src_patches = patches[torch.from_numpy(grid).to(device)]
    src_single = torch.from_numpy(masked_mean_embedding(patches, grid)).to(device)

    retrieved = _retrieve_from_templates(
        src_patches=src_patches,
        src_single=src_single,
        synsets=synsets,
        state=state,
        top_k=top_k,
        fine_agg=fine_agg,
        device=device,
    )
    if retrieved is None:
        raise RuntimeError(f"no CAD retrieved for '{object_label}'")

    synset, model_id, view_id, score, coarse_score = retrieved
    # Feature-only pool: derive per-view paths from the precomputed clean caches.
    model_dir = render_pool / synset / model_id
    clean_views = sorted((model_dir / "precomputed").glob("clean_*.npz"))
    render_paths = [str(model_dir / "renders" / f"render_{p.stem.split('_')[-1]}.png")
                    for p in clean_views]
    if not render_paths:
        raise FileNotFoundError(
            f"no precomputed view caches for {synset}/{model_id} under {render_pool}")

    return ZeroShotRetrievalResult(
        label=object_label,
        synset=synset,
        synset_name=vocabulary.synsets[synset],
        model_id=model_id,
        view_id=view_id,
        score=float(score),
        coarse_score=float(coarse_score),
        render_paths=render_paths,
    )
