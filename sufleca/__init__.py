"""SUFLECA single-view CAD alignment."""

from .featurizer import (
    CHECKPOINTS,
    PUBLIC_CHECKPOINTS,
    extract_sufleca_features,
    load_sufleca_model,
)
from .sv_align import align_single_view
from .zero_shot import (
    ZeroShotRetrievalResult,
    ZeroShotVocabulary,
    load_dinov3_retriever,
    load_zero_shot_vocabulary,
    retrieve_zero_shot_cad,
)
from .demo_utils import (
    DepthPrediction,
    GroundingDinoPrediction,
    MaskPrediction,
    bbox_from_json,
    detect_bbox_with_grounding_dino,
    load_grounding_dino,
    load_pointmap_model,
    load_sam_predictor,
    make_alignment_3d_figure,
    make_cad_overlay_image,
    make_correspondence_image,
    predict_depth_with_pointmap,
    render_cad_view,
    segment_with_sam_box,
)

__all__ = [
    "CHECKPOINTS",
    "PUBLIC_CHECKPOINTS",
    "DepthPrediction",
    "GroundingDinoPrediction",
    "MaskPrediction",
    "ZeroShotRetrievalResult",
    "ZeroShotVocabulary",
    "align_single_view",
    "bbox_from_json",
    "detect_bbox_with_grounding_dino",
    "extract_sufleca_features",
    "load_dinov3_retriever",
    "load_grounding_dino",
    "load_pointmap_model",
    "load_sam_predictor",
    "load_sufleca_model",
    "load_zero_shot_vocabulary",
    "make_alignment_3d_figure",
    "make_cad_overlay_image",
    "make_correspondence_image",
    "predict_depth_with_pointmap",
    "render_cad_view",
    "retrieve_zero_shot_cad",
    "segment_with_sam_box",
]
