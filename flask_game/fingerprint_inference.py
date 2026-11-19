"""Fingerprint-only inference adapter for the offline lab.

This module deliberately does not participate in face, voice, command, door,
or enrollment flows. The prediction is delegated to the existing
``fingerprint_research.identify_upload`` implementation so the trained SVM,
Sobel/LBP feature extractor, pair-feature construction, score direction, and
threshold semantics remain unchanged. The extra code only validates the
trusted artifact and creates visualizations from the same preprocessing path.
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern

import bioguard_core as core
import fingerprint_research


class FingerprintInferenceError(ValueError):
    """Safe fingerprint-only error with a stable API code."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(detail)


def load_fingerprint_package(model_path: str | Path) -> dict:
    """Load and validate the existing artifact without retraining or mutation."""
    try:
        package = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - exact loader failures vary by environment
        raise FingerprintInferenceError("MODEL_LOAD_FAILED", "The fingerprint model could not be loaded.") from exc
    if not isinstance(package, dict):
        raise FingerprintInferenceError("MODEL_INCOMPATIBLE", "The fingerprint model artifact has an invalid structure.")
    required = {"model", "model_name", "threshold", "demo_gallery"}
    if not required.issubset(package):
        raise FingerprintInferenceError("MODEL_INCOMPATIBLE", "The fingerprint model is missing required metadata.")
    model = package["model"]
    try:
        threshold = float(package["threshold"])
    except (TypeError, ValueError) as exc:
        raise FingerprintInferenceError("MODEL_INCOMPATIBLE", "The fingerprint threshold is invalid.") from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise FingerprintInferenceError("MODEL_INCOMPATIBLE", "The fingerprint threshold is outside the probability range.")
    expected_pair_dim = core.FINGER_DIM + 1
    if int(getattr(model, "n_features_in_", -1)) != expected_pair_dim:
        raise FingerprintInferenceError(
            "MODEL_INCOMPATIBLE",
            f"The model expects {getattr(model, 'n_features_in_', 'unknown')} features; the shared pipeline produces {expected_pair_dim}.",
        )
    gallery = package["demo_gallery"]
    if not isinstance(gallery, dict) or not gallery:
        raise FingerprintInferenceError("MODEL_INCOMPATIBLE", "The fingerprint demo gallery is empty.")
    for reference in gallery.values():
        vector = np.asarray(reference, dtype=np.float32)
        if vector.shape != (core.FINGER_DIM,) or not np.isfinite(vector).all():
            raise FingerprintInferenceError("MODEL_INCOMPATIBLE", "The fingerprint gallery contains an invalid reference vector.")
    return package


def _read_color(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FingerprintInferenceError("IMAGE_DECODE_FAILED", "The uploaded file is not a readable fingerprint image.")
    return image


def _preprocess(path: str | Path) -> dict[str, np.ndarray]:
    """Return real intermediate arrays from the exact fingerprint pipeline."""
    color = _read_color(path)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)
    normalized = equalized.astype(np.float32) / 255.0
    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    lbp = local_binary_pattern(equalized, P=8, R=1, method="uniform").astype(np.float32)
    # Calling the unchanged research extractor is the source of truth for the
    # vector used by the model. The arrays above are display-only intermediates.
    vector = np.asarray(core.extract_fingerprint(path), dtype=np.float32)
    if vector.shape != (core.FINGER_DIM,) or not np.isfinite(vector).all():
        raise FingerprintInferenceError("FEATURE_EXTRACTION_FAILED", "The fingerprint feature vector has an invalid dimension.")
    return {
        "original": color,
        "grayscale": equalized,
        "sobel": magnitude,
        "lbp": lbp,
        "vector": vector,
    }


def _data_url(image: np.ndarray, mode: str = "gray") -> str:
    array = np.asarray(image)
    if array.dtype.kind == "f":
        array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if mode == "gray" and array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise FingerprintInferenceError("VISUALIZATION_FAILED", "A fingerprint visualization could not be generated.")
    value = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{value}"


def verify_upload_with_visuals(image_path: str | Path, model_path: str | Path) -> dict:
    """Run the existing real prediction and return safe UI-ready diagnostics."""
    package = load_fingerprint_package(model_path)
    try:
        visual = _preprocess(image_path)
        research_result = fingerprint_research.identify_upload(image_path, model_path=model_path)
    except FingerprintInferenceError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise FingerprintInferenceError("VERIFICATION_FAILED", "Fingerprint verification could not be completed.") from exc
    match = bool(research_result.get("match"))
    score = research_result.get("score")
    threshold = float(package["threshold"])
    return {
        "ok": True,
        "match": match,
        "decision": "MATCH" if match else "NO MATCH",
        "finger": research_result.get("finger", "Unknown"),
        "reason": research_result.get("reason") or ("Matched a demo gallery reference." if match else "No demo gallery reference reached the trained threshold."),
        "score": float(score) if score is not None else None,
        "threshold": threshold,
        "model_name": str(package["model_name"]),
        "feature_dim": int(core.FINGER_DIM),
        "pair_feature_dim": int(getattr(package["model"], "n_features_in_")),
        "gallery_count": len(package["demo_gallery"]),
        "dataset": package.get("dataset", "unknown"),
        "method": package.get("method", "unknown"),
        "limits": package.get("limits", ""),
        "visualizations": {
            "original": _data_url(visual["original"], "color"),
            "grayscale": _data_url(visual["grayscale"]),
            "sobel": _data_url(visual["sobel"]),
            "lbp": _data_url(visual["lbp"]),
        },
    }
