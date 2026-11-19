"""Traditional-ML matching for the private Flask enrollment gallery.

The public RAVDESS verifier is intentionally kept separate.  Its probability
model was trained on public-actor pair features and is useful for the academic
reports, but it is not a calibrated score for a newly enrolled local person.

This module compares the exact HOG/LBP and MFCC feature vectors directly to
private enrollment samples.  It does not train or modify any public artifact.
Distances are converted to a monotonically increasing similarity score so the
Flask diagnostics can display one consistent ``score >= threshold`` contract.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import cv2
import librosa

import bioguard_core as core


LOCAL_MATCHING_VERSION = "1.0"
MODALITY_DIMS = {"image": core.FACE_DIM, "voice": core.VOICE_DIM}
TARGET_VALIDATION_FAR = 0.05
DEFAULT_METHODS = {"image": "zdist_nn", "voice": "zcos_knn3"}

# These are generated from public_data/feature_cache.joblib validation samples
# (known validation + validation unknowns, target FAR <= 5%).  They are only a
# provisional fallback when a local gallery has fewer than two identities or
# has no individual sample vectors yet.  They are not public SVM thresholds.
FALLBACK_CALIBRATION = {
    "image": {
        "method": "zdist_nn",
        "threshold_raw": 1.1850834457702148,
        "threshold": 1.0 / (1.0 + 1.1850834457702148),
        "higher_is_better": True,
        "source": "public_data_validation_plus_validation_unknown",
        "status": "provisional_public_reference",
        "target_far": TARGET_VALIDATION_FAR,
    },
    "voice": {
        "method": "zcos_knn3",
        "threshold_raw": 0.40812410656005293,
        "threshold": 0.40812410656005293,
        "higher_is_better": True,
        "source": "public_data_validation_plus_validation_unknown",
        "status": "provisional_public_reference",
        "target_far": TARGET_VALIDATION_FAR,
    },
}

METHOD_SPECS = {
    "cosine_nn": ("cosine", "raw", "nn"),
    "cosine_mean": ("cosine", "raw", "mean"),
    "cosine_knn3": ("cosine", "raw", "knn3"),
    "zcos_nn": ("cosine", "zscore", "nn"),
    "zcos_mean": ("cosine", "zscore", "mean"),
    "zcos_knn3": ("cosine", "zscore", "knn3"),
    "zdist_nn": ("distance", "zscore_distance", "nn"),
    "zdist_mean": ("distance", "zscore_distance", "mean"),
    "zdist_knn3": ("distance", "zscore_distance", "knn3"),
    "voice_shape_cos_nn": ("cosine", "voice_shape", "nn"),
    "voice_shape_cos_mean": ("cosine", "voice_shape", "mean"),
    "voice_shape_cos_knn3": ("cosine", "voice_shape", "knn3"),
    "voice_shape_dist_nn": ("distance", "voice_shape_distance", "nn"),
    "voice_shape_dist_mean": ("distance", "voice_shape_distance", "mean"),
    "voice_shape_dist_knn3": ("distance", "voice_shape_distance", "knn3"),
}


def crop_single_face(image_or_path, require_face: bool = True) -> np.ndarray:
    """Use the shared live-capture Haar crop before the core HOG/LBP extractor."""
    image = cv2.imread(str(image_or_path), cv2.IMREAD_GRAYSCALE) if isinstance(image_or_path, (str, Path)) else image_or_path
    if image is None:
        raise ValueError("Image could not be loaded")
    if getattr(image, "ndim", 0) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Image is not a valid grayscale frame")
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV Haar cascade is missing")
    faces = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4, minSize=(45, 45))
    if len(faces) == 0:
        if require_face:
            raise ValueError("No frontal face detected: capture again with good lighting")
        return image
    if require_face and len(faces) != 1:
        raise ValueError("Capture exactly one face")
    x, y, width, height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    return image[int(y):int(y + height), int(x):int(x + width)]


def extract_face_live(image_or_path) -> np.ndarray:
    """Shared live/enrollment face adapter: Haar crop then core HOG/LBP."""
    crop = crop_single_face(image_or_path, require_face=True)
    vector = core.extract_face(crop, require_face=False, already_cropped=True)
    if vector.shape != (core.FACE_DIM,) or not np.isfinite(vector).all():
        raise ValueError("Face feature extraction produced an invalid vector")
    return vector


# Explicit aliases make the research and deployment extractor contract clear.
extract_face = core.extract_face


def extract_voice_live(wav_path) -> np.ndarray:
    """Validate live audio, then call the exact MFCC/ZCR/RMS extractor."""
    try:
        audio, _ = librosa.load(str(wav_path), sr=core.SR, mono=True)
    except Exception as exc:
        raise ValueError("Audio could not be decoded") from exc
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or invalid")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if peak < 1e-4 or rms < 1e-4:
        raise ValueError("Recording is silent or too quiet")
    vector = core.extract_voice(wav_path)
    if vector.shape != (core.VOICE_DIM,) or not np.isfinite(vector).all():
        raise ValueError("Voice feature extraction produced an invalid vector")
    return vector


# The Flask and enrollment paths use the validated wrapper; the wrapper calls
# the unchanged research extractor after rejecting unusable microphone input.
extract_voice = extract_voice_live


def _as_vector(value: Any, modality: str) -> np.ndarray:
    if modality not in MODALITY_DIMS:
        raise ValueError(f"Unsupported local modality: {modality}")
    vector = np.asarray(value, dtype=np.float64)
    expected = MODALITY_DIMS[modality]
    if vector.shape != (expected,) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid {modality} feature vector; expected {expected} finite values")
    return vector


def _l2(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


@lru_cache(maxsize=4)
def public_reference_stats(project_root: str, modality: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load fixed normalization statistics from the existing feature cache.

    The cache is a research feature cache, not a model artifact.  If it is not
    available, the caller falls back to private-gallery statistics.  No raw
    media is read or returned by this function.
    """
    cache_path = Path(project_root) / "public_data" / "feature_cache.joblib"
    if not cache_path.is_file():
        return None
    try:
        cache = joblib.load(cache_path)
        frame = cache.get("frame") if isinstance(cache, dict) else None
        if frame is None or "split" not in frame:
            return None
        column = f"{modality}_vec"
        values = [np.asarray(value, dtype=np.float64) for value in frame.loc[frame.split == "enroll", column]]
        if not values:
            return None
        matrix = np.stack(values)
        center = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale = np.where(scale >= 1e-6, scale, 1.0)
        return center, scale
    except (OSError, KeyError, TypeError, ValueError, EOFError):
        return None


def _gallery_stats(gallery: dict, modality: str) -> tuple[np.ndarray, np.ndarray] | None:
    values: list[np.ndarray] = []
    for identity in gallery.get("templates", {}):
        samples, _ = sample_vectors(gallery, identity, modality)
        values.extend(samples)
    if not values:
        return None
    matrix = np.stack(values)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale >= 1e-6, scale, 1.0)
    return center, scale


def _stats(gallery: dict, modality: str, project_root: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    if project_root is not None:
        fixed = public_reference_stats(str(Path(project_root).resolve()), modality)
        if fixed is not None:
            return fixed
    local = _gallery_stats(gallery, modality)
    if local is not None:
        return local
    return np.zeros(MODALITY_DIMS[modality], dtype=np.float64), np.ones(MODALITY_DIMS[modality], dtype=np.float64)


def _prepare(vector: np.ndarray, modality: str, normalization: str, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    vector = _as_vector(vector, modality)
    if normalization == "raw":
        return _l2(vector)
    if normalization == "zscore":
        center, scale = stats
        return _l2((vector - center) / scale)
    if normalization == "zscore_distance":
        center, scale = stats
        return (vector - center) / scale
    if normalization in {"voice_shape", "voice_shape_distance"}:
        if modality != "voice":
            return _l2(vector) if normalization == "voice_shape" else vector
        shaped = vector.copy()
        # MFCC coefficient 0 and RMS carry much of the recording-level gain.
        # Remove only those offsets; retain the spectral/delta structure.
        shaped[:13] -= shaped[0]
        shaped[65:67] -= shaped[65]
        return _l2(shaped) if normalization == "voice_shape" else shaped
    raise ValueError(f"Unsupported normalization: {normalization}")


def _score_details(
    query: np.ndarray,
    references: np.ndarray,
    modality: str,
    method: str,
    stats: tuple[np.ndarray, np.ndarray],
) -> dict[str, float | str]:
    if method not in METHOD_SPECS:
        raise ValueError(f"Unknown local matching method: {method}")
    kind, normalization, aggregation = METHOD_SPECS[method]
    prepared_query = _prepare(query, modality, normalization, stats)
    prepared_refs = np.stack([_prepare(ref, modality, normalization, stats) for ref in references])
    if kind == "cosine":
        raw_values = prepared_refs @ prepared_query
        if aggregation == "nn":
            raw = float(np.max(raw_values))
        elif aggregation == "mean":
            raw = float(prepared_query @ _l2(prepared_refs.mean(axis=0)))
        else:
            raw = float(np.mean(np.sort(raw_values)[-min(3, len(raw_values)):]))
        return {
            "score": raw,
            "raw_score": raw,
            "distance": 1.0 - raw,
            "method": method,
            "higher_is_better": "true",
        }
    distances = np.linalg.norm(prepared_refs - prepared_query, axis=1) / np.sqrt(len(prepared_query))
    if aggregation == "nn":
        raw_distance = float(np.min(distances))
    elif aggregation == "mean":
        mean_ref = _prepare(np.mean(references, axis=0), modality, normalization, stats)
        raw_distance = float(np.linalg.norm(mean_ref - prepared_query) / np.sqrt(len(prepared_query)))
    else:
        raw_distance = float(np.mean(np.sort(distances)[:min(3, len(distances))]))
    return {
        "score": 1.0 / (1.0 + raw_distance),
        "raw_score": -raw_distance,
        "distance": raw_distance,
        "method": method,
        "higher_is_better": "true",
    }


def score_vectors(
    query: np.ndarray,
    references: np.ndarray | list[np.ndarray],
    modality: str,
    method: str,
    project_root: str | Path | None = None,
) -> dict[str, float | str]:
    """Public score API shared by notebook experiments and Flask diagnostics."""
    matrix = np.asarray(references, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != MODALITY_DIMS[modality] or len(matrix) == 0:
        raise ValueError(f"Invalid {modality} reference matrix")
    stats = _stats({}, modality, project_root)
    return _score_details(query, matrix, modality, method, stats)


def sample_vectors(gallery: dict, identity: str, modality: str) -> tuple[list[np.ndarray], bool]:
    """Return individual vectors when available, otherwise the legacy mean.

    Existing galleries remain readable.  New enrollments write individual
    vectors under ``sample_vectors`` while retaining the legacy mean under
    ``templates`` for compatibility with the research/command code.
    """
    samples = gallery.get("sample_vectors", {}).get(identity, {}).get(modality)
    if samples is not None:
        array = np.asarray(samples, dtype=np.float64)
        if array.ndim == 2 and array.shape[1] == MODALITY_DIMS[modality] and len(array) > 0 and np.isfinite(array).all():
            return [row for row in array], True
    template = gallery.get("templates", {}).get(identity, {}).get(modality)
    if template is None:
        return [], False
    return [_as_vector(template, modality)], False


def gallery_vector_summary(gallery: dict) -> dict[str, Any]:
    summary: dict[str, Any] = {"users": len(gallery.get("templates", {})), "identities": {}}
    for identity in sorted(gallery.get("templates", {})):
        details: dict[str, Any] = {}
        for modality in MODALITY_DIMS:
            values, individual = sample_vectors(gallery, identity, modality)
            details[modality] = {"count": len(values), "individual": individual, "dimension": MODALITY_DIMS[modality]}
        summary["identities"][identity] = details
    return summary


def _fallback(modality: str) -> dict[str, Any]:
    return dict(FALLBACK_CALIBRATION[modality])


def _threshold_from_scores(scores: list[float], target_far: float) -> tuple[float, float, float]:
    """Choose a higher-is-better threshold using negative and positive scores."""
    positives = np.asarray(scores[0], dtype=np.float64)
    negatives = np.asarray(scores[1], dtype=np.float64)
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("Both genuine and impostor scores are required for calibration")
    best: tuple[float, float, float, float] | None = None
    for threshold in np.unique(np.concatenate([positives, negatives])):
        far = float(np.mean(negatives >= threshold))
        tar = float(np.mean(positives >= threshold))
        frr = 1.0 - tar
        if far <= target_far + 1e-12:
            candidate = (tar, -far, -float(threshold), float(threshold))
            if best is None or candidate > best:
                best = candidate
    if best is None:
        threshold = float(np.nextafter(np.max(negatives), np.inf))
        tar = float(np.mean(positives >= threshold))
        far = float(np.mean(negatives >= threshold))
        return threshold, far, 1.0 - tar
    return best[3], -best[1], 1.0 - best[0]


def calibrate_score_threshold(
    genuine_scores: list[float] | np.ndarray,
    impostor_scores: list[float] | np.ndarray,
    target_far: float = TARGET_VALIDATION_FAR,
) -> dict[str, float | int]:
    """Calibrate a higher-is-better similarity threshold on validation only."""
    threshold, far, frr = _threshold_from_scores(
        ([float(value) for value in genuine_scores], [float(value) for value in impostor_scores]),
        target_far,
    )
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)
    return {
        "threshold": float(threshold),
        "far": float(far),
        "frr": float(frr),
        "tar": float(1.0 - frr),
        "genuine_count": int(len(genuine)),
        "impostor_count": int(len(impostor)),
        "target_far": float(target_far),
    }


def calibrate_gallery(
    gallery: dict,
    modality: str,
    project_root: str | Path | None = None,
    method: str | None = None,
    target_far: float = TARGET_VALIDATION_FAR,
    ambiguity_margin: float | None = None,
) -> dict[str, Any]:
    """Calibrate on private leave-one-out genuine and cross-user impostor pairs.

    Cross-user pairs are not the same as independent unknown people.  The
    returned status explicitly says so; a one-user gallery therefore uses the
    documented public validation fallback and remains provisional.
    """
    selected_method = method or DEFAULT_METHODS[modality]
    identities = sorted(gallery.get("templates", {}))
    stats = _stats(gallery, modality, project_root)
    genuine: list[float] = []
    impostor: list[float] = []
    individual: dict[str, list[np.ndarray]] = {}
    for identity in identities:
        values, has_individual = sample_vectors(gallery, identity, modality)
        if has_individual and len(values) >= 2:
            individual[identity] = values
    for identity, values in individual.items():
        for index, query in enumerate(values):
            reference = values[:index] + values[index + 1:]
            genuine.append(float(_score_details(query, np.stack(reference), modality, selected_method, stats)["score"]))
            for other, other_values in individual.items():
                if other != identity:
                    impostor.append(float(_score_details(query, np.stack(other_values), modality, selected_method, stats)["score"]))
    if len(individual) >= 2 and genuine and impostor:
        threshold, far, frr = _threshold_from_scores((genuine, impostor), target_far)
        return {
            "method": selected_method,
            "threshold": threshold,
            "threshold_raw": (1.0 / threshold) - 1.0 if selected_method.startswith("zdist") else threshold,
            "higher_is_better": True,
            "source": "private_leave_one_out_and_cross_user_pairs",
            "status": "local_cross_user_provisional",
            "target_far": target_far,
            "calibration_genuine_count": len(genuine),
            "calibration_impostor_count": len(impostor),
            "calibration_far": far,
            "calibration_frr": frr,
            "independent_unknowns_available": False,
            "ambiguity_margin": float(ambiguity_margin if ambiguity_margin is not None else 0.05),
        }
    fallback = _fallback(modality)
    fallback["calibration_genuine_count"] = len(genuine)
    fallback["calibration_impostor_count"] = len(impostor)
    fallback["independent_unknowns_available"] = False
    fallback["ambiguity_margin"] = float(ambiguity_margin if ambiguity_margin is not None else 0.05)
    return fallback


def load_reference_calibration(path: str | Path, modality: str) -> dict[str, Any] | None:
    try:
        data = __import__("json").loads(Path(path).read_text(encoding="utf-8"))
        result = data.get("modalities", {}).get(modality)
        if not isinstance(result, dict) or result.get("method") not in METHOD_SPECS:
            return None
        if not np.isfinite(float(result.get("threshold"))):
            return None
        return dict(result)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def calibration_for(
    gallery: dict,
    modality: str,
    project_root: str | Path | None = None,
    calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the versioned calibration used by the deployed matcher.

    Voice is deliberately different from face here.  A private gallery's
    leave-one-out and cross-user pairs are useful diagnostics, but they are
    enrollment data, not independent validation data.  Replacing the
    versioned voice threshold with a value recomputed from the current gallery
    made the Flask runtime silently change as users were added and produced a
    threshold that was not the one selected by the research benchmark.  When
    the exported voice calibration is present, Flask and the command path now
    use it verbatim; local-gallery calibration remains available through
    ``calibrate_gallery`` for diagnostics only.

    The existing face behavior is intentionally preserved.
    """
    if calibration_path is not None:
        saved = load_reference_calibration(calibration_path, modality)
        if saved is not None:
            result = saved
        else:
            result = calibrate_gallery(gallery, modality, project_root)
    else:
        result = calibrate_gallery(gallery, modality, project_root)

    if modality == "voice" and calibration_path is not None and saved is not None:
        result.setdefault("ambiguity_margin", 0.05)
        result["runtime_policy"] = "versioned_reference_only"
        result["local_gallery_calibration_required"] = True
        return result

    # With two or more private identities, recalibrate from the current
    # gallery.  This prevents a stale public fallback from being treated as a
    # final local threshold.  This branch remains unchanged for face.  Voice
    # returns above whenever its versioned calibration is available.
    if len(gallery.get("templates", {})) >= 2:
        # Preserve the versioned research-selected method while recalibrating
        # its threshold against the current private leave-one-out gallery.
        local = calibrate_gallery(
            gallery,
            modality,
            project_root,
            method=result.get("method"),
            ambiguity_margin=float(result.get("ambiguity_margin", 0.05)),
        )
        if local.get("status") == "local_cross_user_provisional":
            result = local
    result.setdefault("ambiguity_margin", 0.05)
    return result


def rank_gallery(
    vector: np.ndarray,
    gallery: dict,
    modality: str,
    calibration: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    calibration = calibration or calibration_for(gallery, modality, project_root)
    method = str(calibration["method"])
    stats = _stats(gallery, modality, project_root)
    ranking: list[dict[str, Any]] = []
    for identity in sorted(gallery.get("templates", {})):
        references, individual = sample_vectors(gallery, identity, modality)
        if not references:
            continue
        details = _score_details(vector, np.stack(references), modality, method, stats)
        ranking.append({
            "identity": identity,
            "score": float(details["score"]),
            "distance": float(details["distance"]),
            "sample_count": len(references),
            "individual_templates": individual,
        })
    ranking.sort(key=lambda item: item["score"], reverse=True)
    if not ranking:
        return {
            "accepted": False,
            "identity": "Unknown",
            "code": "FACE_UNKNOWN" if modality == "image" else "VOICE_LOW_CONFIDENCE",
            "reason": "No local templates are available",
            "ranking": [],
            "calibration": calibration,
        }
    best = ranking[0]
    second = ranking[1] if len(ranking) > 1 else None
    threshold = float(calibration["threshold"])
    margin = float(best["score"] - second["score"]) if second else None
    ambiguous = bool(second is not None and margin is not None and margin < float(calibration.get("ambiguity_margin", 0.05)))
    accepted = bool(best["score"] >= threshold and not ambiguous)
    if accepted:
        code = "FACE_ACCEPTED" if modality == "image" else "VOICE_ACCEPTED"
        identity = best["identity"]
        reason = "Local gallery match accepted"
    elif ambiguous:
        code = "FACE_AMBIGUOUS" if modality == "image" else "VOICE_AMBIGUOUS"
        identity = "Unknown"
        reason = "Top local candidates are too close to distinguish safely"
    elif modality == "image":
        code = "FACE_UNKNOWN"
        identity = "Unknown"
        reason = "Best local face score is below the calibrated threshold"
    else:
        code = "VOICE_LOW_CONFIDENCE"
        identity = "Unknown"
        reason = "Best local voice score is below the calibrated threshold"
    return {
        "accepted": accepted,
        "identity": identity,
        "candidate": best["identity"],
        "code": code,
        "reason": reason,
        "score": float(best["score"]),
        "threshold": threshold,
        "distance": float(best["distance"]),
        "margin": margin,
        "runner_up": second["identity"] if second else None,
        "runner_up_score": float(second["score"]) if second else None,
        "ranking": ranking,
        "calibration": calibration,
    }


def verify_claim(
    vector: np.ndarray,
    gallery: dict,
    identity: str,
    modality: str,
    calibration: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a modality against the face-selected identity and all candidates."""
    ranked = rank_gallery(vector, gallery, modality, calibration, project_root)
    target = next((item for item in ranked.get("ranking", []) if item["identity"] == identity), None)
    if target is None:
        return {**ranked, "accepted": False, "identity": "Unknown", "code": "VOICE_LOW_CONFIDENCE"}
    threshold = float(ranked["threshold"])
    best_identity = ranked.get("candidate")
    target_score = float(target["score"])
    if best_identity != identity and target_score < threshold:
        code = "VOICE_MISMATCH" if modality == "voice" else "FACE_UNKNOWN"
        return {**ranked, "accepted": False, "identity": "Unknown", "code": code, "claim_score": target_score}
    if best_identity != identity:
        code = "VOICE_MISMATCH" if modality == "voice" else "FACE_AMBIGUOUS"
        return {**ranked, "accepted": False, "identity": "Unknown", "code": code, "claim_score": target_score}
    if target_score < threshold:
        code = "VOICE_LOW_CONFIDENCE" if modality == "voice" else "FACE_LOW_CONFIDENCE"
        return {**ranked, "accepted": False, "identity": "Unknown", "code": code, "claim_score": target_score}
    if ranked.get("code") in {"VOICE_AMBIGUOUS", "FACE_AMBIGUOUS"}:
        return {**ranked, "accepted": False, "identity": "Unknown", "code": ranked["code"], "claim_score": target_score}
    return {**ranked, "accepted": True, "identity": identity, "code": "VOICE_ACCEPTED" if modality == "voice" else "FACE_ACCEPTED", "claim_score": target_score}


def hierarchical_verify(
    face_vector: np.ndarray,
    voice_vector: np.ndarray,
    gallery: dict,
    project_root: str | Path | None = None,
    calibration_path: str | Path | None = None,
    permissions: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Shared face -> same-candidate voice -> permission decision."""
    face_calibration = calibration_for(gallery, "image", project_root, calibration_path)
    face = rank_gallery(face_vector, gallery, "image", face_calibration, project_root)
    if not face.get("accepted"):
        return {
            "accepted": False,
            "identity": "Unknown",
            "code": face.get("code", "FACE_UNKNOWN"),
            "stage": "face",
            "face": face,
            "voice": None,
            "reason": face.get("reason", "Face was not accepted"),
        }
    candidate = str(face["identity"])
    voice_calibration = calibration_for(gallery, "voice", project_root, calibration_path)
    voice = verify_claim(voice_vector, gallery, candidate, "voice", voice_calibration, project_root)
    if not voice.get("accepted"):
        return {
            "accepted": False,
            "identity": "Unknown",
            "code": voice.get("code", "VOICE_LOW_CONFIDENCE"),
            "stage": "voice",
            "candidate": candidate,
            "face": face,
            "voice": voice,
            "reason": voice.get("reason", "Voice did not match the face candidate"),
        }
    authorized = True if permissions is None else bool(permissions.get(candidate, False))
    if not authorized:
        return {
            "accepted": False,
            "identity": "Unknown",
            "code": "PERMISSION_DENIED",
            "stage": "permission",
            "candidate": candidate,
            "face": face,
            "voice": voice,
            "reason": "Identity matched but door permission is denied",
        }
    return {
        "accepted": True,
        "identity": candidate,
        "code": "ACCESS_GRANTED",
        "stage": "permission",
        "candidate": candidate,
        "face": face,
        "voice": voice,
        "reason": "Face, same-candidate voice, and permission accepted",
    }


def enrollment_samples_from_arrays(face_vectors: np.ndarray, voice_vectors: np.ndarray) -> dict[str, np.ndarray]:
    """Validate and serialize individual enrollment vectors for the gallery."""
    faces = np.asarray(face_vectors, dtype=np.float32)
    voices = np.asarray(voice_vectors, dtype=np.float32)
    if faces.ndim != 2 or faces.shape[1] != core.FACE_DIM or not np.isfinite(faces).all():
        raise ValueError("Invalid individual face enrollment vectors")
    if voices.ndim != 2 or voices.shape[1] != core.VOICE_DIM or not np.isfinite(voices).all():
        raise ValueError("Invalid individual voice enrollment vectors")
    return {"image": faces, "voice": voices}


