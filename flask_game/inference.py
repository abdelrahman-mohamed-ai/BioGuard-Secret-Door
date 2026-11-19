"""Flask import surface for the shared BioGuard feature pipeline.

The implementation is intentionally kept in the project-root
``bioguard_matching.py`` module so the master notebook and the Flask game use
the same face crop, HOG/LBP extraction, voice extractor, and local matcher.
"""
from bioguard_matching import (  # noqa: F401
    calibration_for,
    crop_single_face,
    extract_face,
    extract_face_live,
    extract_voice,
    extract_voice_live,
    hierarchical_verify,
    rank_gallery,
    score_vectors,
    verify_claim,
)

__all__ = [
    "calibration_for",
    "crop_single_face",
    "extract_face",
    "extract_face_live",
    "extract_voice",
    "extract_voice_live",
    "hierarchical_verify",
    "rank_gallery",
    "score_vectors",
    "verify_claim",
]
