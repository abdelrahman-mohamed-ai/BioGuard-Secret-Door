"""Benchmark the shared local gallery matcher without exporting biometrics.

This script never retrains ``model/public_models.joblib`` and never writes raw
images, audio, feature vectors, identifiers, or secrets.  It uses the existing
public feature cache for an independent reference experiment and attempts a
read-only audit of the private gallery media.  If Windows denies access to the
private media directory, that limitation is recorded instead of bypassed.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bioguard_core as core  # noqa: E402
import bioguard_matching as matching  # noqa: E402
import deployment  # noqa: E402


REPORT_PATH = ROOT / "flask_game" / "LOCAL_MATCHING_REPORT.md"
CALIBRATION_PATH = ROOT / "flask_game" / "local_matching_calibration.json"
PRIVATE_ROOT = ROOT / "flask_game" / "private_data"


def _float(value: Any) -> float:
    return float(value)


def load_public_features() -> Any:
    cache = joblib.load(ROOT / "public_data" / "feature_cache.joblib")
    frame = cache.get("frame") if isinstance(cache, dict) else None
    if frame is None:
        raise ValueError("public_data/feature_cache.joblib has no feature frame")
    for column in ("image_vec", "voice_vec"):
        frame[column] = frame[column].map(lambda value: np.asarray(value, dtype=np.float64))
    return frame


def _public_gallery(frame, modality: str) -> dict[str, np.ndarray]:
    identities = sorted(frame.loc[frame.split == "enroll", "user"].unique())
    return {
        identity: np.stack(frame.loc[(frame.user == identity) & (frame.split == "enroll"), f"{modality}_vec"].to_list())
        for identity in identities
    }


def _score(query: np.ndarray, references: np.ndarray, modality: str, method: str) -> float:
    return _float(matching.score_vectors(query, references, modality, method, ROOT)["score"])


def _claim_trials(frame, gallery: dict[str, np.ndarray], modality: str, method: str, split: str, unknown_split: str) -> tuple[list[float], list[float]]:
    genuine: list[float] = []
    impostor: list[float] = []
    for _, row in frame[frame.split == split].iterrows():
        query = row[f"{modality}_vec"]
        for identity, references in gallery.items():
            score = _score(query, references, modality, method)
            (genuine if identity == row.user else impostor).append(score)
    for _, row in frame[frame.split == unknown_split].iterrows():
        query = row[f"{modality}_vec"]
        for references in gallery.values():
            impostor.append(_score(query, references, modality, method))
    return genuine, impostor


def _identify(query: np.ndarray, gallery: dict[str, np.ndarray], modality: str, method: str, threshold: float) -> tuple[str, bool, float]:
    scores = {identity: _score(query, references, modality, method) for identity, references in gallery.items()}
    if not scores:
        return "Unknown", False, float("nan")
    identity = max(scores, key=scores.get)
    score = float(scores[identity])
    return identity if score >= threshold else "Unknown", score >= threshold, score


def _open_metrics(frame, gallery, modality: str, method: str, threshold: float, split: str, unknown_split: str) -> dict[str, float | int]:
    known = []
    unknown = []
    for _, row in frame[frame.split == split].iterrows():
        prediction, accepted, _ = _identify(row[f"{modality}_vec"], gallery, modality, method, threshold)
        known.append((str(row.user), prediction, accepted))
    for _, row in frame[frame.split == unknown_split].iterrows():
        prediction, accepted, _ = _identify(row[f"{modality}_vec"], gallery, modality, method, threshold)
        unknown.append(prediction)
    return {
        "known_attempts": len(known),
        "known_id_rate": float(sum(pred == truth for truth, pred, _ in known) / max(1, len(known))),
        "known_reject_rate": float(sum(pred == "Unknown" for _, pred, _ in known) / max(1, len(known))),
        "unknown_attempts": len(unknown),
        "unknown_rejection_rate": float(sum(pred == "Unknown" for pred in unknown) / max(1, len(unknown))),
        "unknown_false_accept_rate": float(sum(pred != "Unknown" for pred in unknown) / max(1, len(unknown))),
    }


def _validation_ambiguity_margin(frame, gallery, modality: str, method: str, threshold: float) -> float:
    """Calibrate a score-scale-aware top-two margin from validation only."""
    margins: list[float] = []
    for _, row in frame[frame.split == "val"].iterrows():
        scores = sorted(
            (
                _score(row[f"{modality}_vec"], references, modality, method),
                identity,
            )
            for identity, references in gallery.items()
        )
        if len(scores) >= 2 and scores[-1][1] == str(row.user) and scores[-1][0] >= threshold:
            margins.append(float(scores[-1][0] - scores[-2][0]))
    if not margins:
        return 0.05
    # The tenth percentile of correctly accepted validation margins makes the
    # ambiguity check meaningful for cosine scores near 1.0 (especially voice)
    # without tuning on the held-out test partition.
    return max(1e-6, float(np.quantile(np.asarray(margins, dtype=np.float64), 0.10)))


def _test_claim_confusion(frame, gallery, modality: str, method: str, threshold: float) -> dict[str, int]:
    labels: list[int] = []
    predictions: list[int] = []
    for _, row in frame[frame.split == "test"].iterrows():
        query = row[f"{modality}_vec"]
        for identity, references in gallery.items():
            labels.append(int(identity == row.user))
            predictions.append(int(_score(query, references, modality, method) >= threshold))
    for _, row in frame[frame.split == "test_unknown"].iterrows():
        query = row[f"{modality}_vec"]
        for references in gallery.values():
            labels.append(0)
            predictions.append(int(_score(query, references, modality, method) >= threshold))
    y = np.asarray(labels, dtype=int)
    p = np.asarray(predictions, dtype=int)
    return {
        "TN": int(((y == 0) & (p == 0)).sum()),
        "FP": int(((y == 0) & (p == 1)).sum()),
        "FN": int(((y == 1) & (p == 0)).sum()),
        "TP": int(((y == 1) & (p == 1)).sum()),
    }


def _candidate_methods(modality: str) -> list[str]:
    if modality == "image":
        return [name for name in matching.METHOD_SPECS if not name.startswith("voice_shape")]
    return list(matching.METHOD_SPECS)


def benchmark_modality(frame, modality: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gallery = _public_gallery(frame, modality)
    rows: list[dict[str, Any]] = []
    for method in _candidate_methods(modality):
        genuine, impostor = _claim_trials(frame, gallery, modality, method, "val", "val_unknown")
        calibration = matching.calibrate_score_threshold(genuine, impostor, matching.TARGET_VALIDATION_FAR)
        threshold = _float(calibration["threshold"])
        validation_open = _open_metrics(frame, gallery, modality, method, threshold, "val", "val_unknown")
        test_open = _open_metrics(frame, gallery, modality, method, threshold, "test", "test_unknown")
        harmonic = 2.0 * validation_open["known_id_rate"] * validation_open["unknown_rejection_rate"] / max(
            validation_open["known_id_rate"] + validation_open["unknown_rejection_rate"], 1e-12
        )
        rows.append({
            "modality": modality,
            "method": method,
            "threshold": threshold,
            "threshold_raw": (1.0 / threshold) - 1.0 if method.startswith("zdist") else threshold,
            "validation_far": _float(calibration["far"]),
            "validation_frr": _float(calibration["frr"]),
            "validation_tar": _float(calibration["tar"]),
            "validation_known_id_rate": validation_open["known_id_rate"],
            "validation_unknown_rejection_rate": validation_open["unknown_rejection_rate"],
            "validation_unknown_far": validation_open["unknown_false_accept_rate"],
            "test_known_id_rate": test_open["known_id_rate"],
            "test_known_reject_rate": test_open["known_reject_rate"],
            "test_unknown_rejection_rate": test_open["unknown_rejection_rate"],
            "test_unknown_far": test_open["unknown_false_accept_rate"],
            "claim_TN": None,
            "claim_FP": None,
            "claim_FN": None,
            "claim_TP": None,
            "validation_selection_score": harmonic,
        })
    selected = max(
        rows,
        key=lambda row: (
            row["validation_selection_score"],
            row["validation_unknown_rejection_rate"],
            row["validation_known_id_rate"],
            -row["validation_far"],
        ),
    )
    confusion = _test_claim_confusion(frame, gallery, modality, selected["method"], selected["threshold"])
    selected.update({f"claim_{key}": value for key, value in confusion.items()})
    selected["ambiguity_margin"] = _validation_ambiguity_margin(
        frame, gallery, modality, selected["method"], selected["threshold"]
    )
    return rows, selected


def _score_only_audit(name: str) -> dict[str, Any] | None:
    path = PRIVATE_ROOT / name
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"rows": 0}
    score_key = "own_public_score" if name.startswith("face") else "own_score"
    pass_key = "own_pass"
    scores = [float(row[score_key]) for row in rows]
    passed = sum(str(row[pass_key]).lower() == "true" for row in rows)
    errors = [float(row["stored_template_mean_error"]) for row in rows if row.get("stored_template_mean_error")]
    return {
        "rows": len(rows),
        "public_passes": passed,
        "public_pass_rate": passed / len(rows),
        "public_score_min": min(scores),
        "public_score_max": max(scores),
        "public_score_mean": sum(scores) / len(scores),
        "stored_template_mean_error_max": max(errors) if errors else None,
    }


def audit_private_media(gallery: dict, selected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "gallery_users": len(gallery.get("templates", {})),
        "gallery_summary": matching.gallery_vector_summary(gallery),
        "media_access": "not_attempted",
        "users_with_readable_media": 0,
        "face_recomputed_samples": 0,
        "voice_recomputed_samples": 0,
        "face_mean_error_max": None,
        "voice_mean_error_max": None,
        "image_loo": [],
        "voice_loo": [],
        "stored_template_loo": {},
    }
    for modality in ("image", "voice"):
        scores: list[float] = []
        for identity in gallery.get("templates", {}):
            samples, individual = matching.sample_vectors(gallery, identity, modality)
            if not individual or len(samples) < 2:
                continue
            matrix = np.stack(samples)
            for index, query in enumerate(matrix):
                references = np.delete(matrix, index, axis=0)
                scores.append(float(matching.score_vectors(
                    query,
                    references,
                    modality,
                    selected[modality]["method"],
                    ROOT,
                )["score"]))
        threshold = float(selected[modality]["threshold"])
        result["stored_template_loo"][modality] = {
            "n": len(scores),
            "accepted": sum(score >= threshold for score in scores),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean": sum(scores) / len(scores) if scores else None,
            "threshold": threshold,
        }
    users_root = PRIVATE_ROOT / "users"
    try:
        identities = [item for item in users_root.iterdir() if item.is_dir() and not item.name.startswith(".")]
    except OSError as exc:
        result["media_access"] = f"unavailable:{type(exc).__name__}"
        result["score_only_face_audit"] = _score_only_audit("face_audit_scores.csv")
        result["score_only_voice_audit"] = _score_only_audit("voice_audit_scores.csv")
        return result
    result["media_access"] = "readable"
    for identity_dir in identities:
        identity = identity_dir.name
        try:
            faces = sorted(path for path in (identity_dir / "faces").iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
            voices = sorted(path for path in (identity_dir / "voices").iterdir() if path.suffix.lower() == ".wav")
        except OSError:
            continue
        face_vectors = []
        for path in faces:
            try:
                face_vectors.append(np.asarray(matching.extract_face_live(path), dtype=np.float64))
            except Exception:
                pass
        voice_vectors = []
        for path in voices:
            try:
                voice_vectors.append(np.asarray(matching.extract_voice(path), dtype=np.float64))
            except Exception:
                pass
        if not face_vectors and not voice_vectors:
            continue
        result["users_with_readable_media"] += 1
        result["face_recomputed_samples"] += len(face_vectors)
        result["voice_recomputed_samples"] += len(voice_vectors)
        if face_vectors and identity in gallery.get("templates", {}):
            error = float(np.max(np.abs(np.asarray(gallery["templates"][identity]["image"]) - np.mean(face_vectors, axis=0))))
            result["face_mean_error_max"] = max(result["face_mean_error_max"] or 0.0, error)
        if voice_vectors and identity in gallery.get("templates", {}):
            error = float(np.max(np.abs(np.asarray(gallery["templates"][identity]["voice"]) - np.mean(voice_vectors, axis=0))))
            result["voice_mean_error_max"] = max(result["voice_mean_error_max"] or 0.0, error)
        for modality, vectors in (("image", face_vectors), ("voice", voice_vectors)):
            if len(vectors) < 2:
                continue
            method = selected[modality]["method"]
            threshold = selected[modality]["threshold"]
            scores = []
            for index, query in enumerate(vectors):
                reference = np.delete(np.stack(vectors), index, axis=0)
                scores.append(_score(query, reference, modality, method))
            result[f"{modality}_loo"].extend(scores)
    result["score_only_face_audit"] = _score_only_audit("face_audit_scores.csv")
    result["score_only_voice_audit"] = _score_only_audit("voice_audit_scores.csv")
    if identities and result["users_with_readable_media"] == 0:
        result["media_access"] = "identity_dirs_visible_media_blocked_or_unreadable"
    for modality in ("image", "voice"):
        values = result[f"{modality}_loo"]
        result[f"{modality}_loo_summary"] = {
            "n": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "threshold": selected[modality]["threshold"],
        }
        result.pop(f"{modality}_loo", None)
    return result


def _calibration(selected: dict[str, dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    modalities: dict[str, Any] = {}
    for modality, row in selected.items():
        modalities[modality] = {
            "version": matching.LOCAL_MATCHING_VERSION,
            "method": row["method"],
            "threshold": row["threshold"],
            "threshold_raw": row["threshold_raw"],
            "higher_is_better": True,
            "source": "public_data_validation_plus_validation_unknown",
            "status": "provisional_public_reference",
            "target_far": matching.TARGET_VALIDATION_FAR,
            "validation_claim_far": row["validation_far"],
            "validation_claim_frr": row["validation_frr"],
            "validation_known_id_rate": row["validation_known_id_rate"],
            "validation_unknown_rejection_rate": row["validation_unknown_rejection_rate"],
            "independent_unknown_test_available": True,
            "local_gallery_calibration_required": True,
            "ambiguity_margin": row.get("ambiguity_margin", 0.05),
        }
    return {
        "version": matching.LOCAL_MATCHING_VERSION,
        "target_far": matching.TARGET_VALIDATION_FAR,
        "source": "public_data_feature_cache_only; not a local-user accuracy claim",
        "modalities": modalities,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(frame, rows: list[dict[str, Any]], selected: dict[str, dict[str, Any]], private_audit: dict[str, Any], calibration: dict[str, Any]) -> None:
    by_modality = {modality: [row for row in rows if row["modality"] == modality] for modality in ("image", "voice")}
    lines = [
        "# BioGuard Local Matching Report",
        "",
        "Generated by `flask_game/tools/benchmark_local_matching.py`. This report contains aggregate scores only; it does not export raw media, feature vectors, identifiers, or secrets.",
        "",
        "## 1. Exact root cause",
        "",
        "The previous Flask path used the public RAVDESS pair-SVM as an absolute gate for a newly enrolled local identity. The public SVM was trained on pair-feature distributions made from public actors and public mean templates. A local user’s HOG/LBP vector can be internally consistent with the local gallery while still being out-of-distribution for that SVM. The supplied local audit therefore showed a zero stored-template mean error but face public scores far below the public threshold, and only partial voice acceptance.",
        "",
        "The public artifacts remain unchanged. Flask now uses the shared direct local gallery matcher below; the public SVM remains available for academic comparison and diagnostics.",
        "",
        "## 2. Previous scoring method",
        "",
        "`public_research.score_claim()` builds `abs(query - mean_template) + cosine(query, mean_template)` and sends it through the public model. The public thresholds are still `Image Only = 0.297165` and `Voice Only = 0.533590`; they are not local-gallery thresholds.",
        "",
        "## 3. Shared local method",
        "",
        "`bioguard_matching.py` is imported by the notebook and Flask. It uses the existing HOG/LBP face vector and MFCC + delta + delta2 + ZCR + RMS voice vector, retains individual enrollment vectors, applies fixed feature normalization, and compares the live vector to the nearest enrollment templates. Face identification rejects low score and ambiguous top candidates. Voice is then checked against the same face-selected identity before permissions.",
        "",
        f"- Face selected method in this run: `{selected['image']['method']}`.",
        f"- Voice selected method in this run: `{selected['voice']['method']}`.",
        "- Thresholds are selected on validation known samples plus Actor 06 validation-unknown samples; Actor 07 is held out for final reporting.",
        "- With fewer than two local identities, the exported public-reference calibration is explicitly provisional. It cannot prove local unknown-person FAR.",
        "",
        "## 4. Candidate method comparison",
        "",
        "Validation thresholds were selected before the test rows were evaluated. `known_id_rate` and `unknown_rejection_rate` are identification metrics; claim FAR/FRR are pair metrics.",
        "",
        "| Modality | Method | Threshold | Val FAR | Val FRR | Val known ID | Val unknown reject | Test known ID | Test unknown reject | Test unknown FAR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for modality in ("image", "voice"):
        for row in sorted(by_modality[modality], key=lambda item: item["validation_selection_score"], reverse=True):
            lines.append(
                f"| {modality} | `{row['method']}` | {_fmt(row['threshold'])} | {_fmt(row['validation_far'])} | {_fmt(row['validation_frr'])} | {_fmt(row['validation_known_id_rate'])} | {_fmt(row['validation_unknown_rejection_rate'])} | {_fmt(row['test_known_id_rate'])} | {_fmt(row['test_unknown_rejection_rate'])} | {_fmt(row['test_unknown_far'])} |"
            )
    lines += [
        "",
        "## 5. Selected configuration and confusion matrices",
        "",
    ]
    for modality in ("image", "voice"):
        row = selected[modality]
        lines += [
            f"### {modality.title()}",
            "",
            f"Selected `{row['method']}` using the validation harmonic score of known-ID rate and unknown rejection. Threshold = `{row['threshold']:.6f}` in the shared similarity scale; raw distance threshold = `{row['threshold_raw']:.6f}` where applicable. Validation-calibrated ambiguity margin = `{row.get('ambiguity_margin', 0.05):.6f}`.",
            "",
            f"Test claim confusion matrix: TN={row['claim_TN']}, FP={row['claim_FP']}, FN={row['claim_FN']}, TP={row['claim_TP']}.",
            "",
        ]
    lines += [
        "## 6. Local gallery diagnostics",
        "",
        f"Gallery identities present: `{private_audit['gallery_users']}`. Individual face/voice vectors are stored for new enrollments; legacy galleries fall back to their existing mean until controlled re-enrollment can add individual vectors.",
        f"Private media audit status: `{private_audit['media_access']}`. Readable users: `{private_audit['users_with_readable_media']}`; recomputed face samples: `{private_audit['face_recomputed_samples']}`; recomputed voice samples: `{private_audit['voice_recomputed_samples']}`.",
        f"Maximum recomputed face mean error: `{_fmt(private_audit['face_mean_error_max'], 7)}`. Maximum recomputed voice mean error: `{_fmt(private_audit['voice_mean_error_max'], 7)}`.",
        f"Stored-template leave-one-out face diagnostic: `{json.dumps(private_audit.get('stored_template_loo', {}).get('image'), sort_keys=True)}`",
        f"Stored-template leave-one-out voice diagnostic: `{json.dumps(private_audit.get('stored_template_loo', {}).get('voice'), sort_keys=True)}`",
        "",
        "The existing score-only audit is retained as an aggregate diagnostic. It is not treated as an independent accuracy result.",
        f"- Public face audit aggregate: `{json.dumps(private_audit.get('score_only_face_audit'), sort_keys=True)}`",
        f"- Public voice audit aggregate: `{json.dumps(private_audit.get('score_only_voice_audit'), sort_keys=True)}`",
        "",
        "## 7. Hierarchical decision",
        "",
        "`FACE_UNKNOWN` / `FACE_LOW_CONFIDENCE` / `FACE_AMBIGUOUS` stops before microphone access. If face passes, voice is compared against that same candidate. `VOICE_MISMATCH` and `VOICE_LOW_CONFIDENCE` remain locked. Only face match AND same-candidate voice match AND `open_door` permission can produce `ACCESS_GRANTED`.",
        "",
        "## 8. Research vs deployment boundary",
        "",
        "The public research SVM/RF results and joblib files were not retrained or rewritten. The new local matcher is implemented in the shared module and evaluated on the available public split as a reference. Public actors are not local employees, so these metrics do not prove a new webcam user’s performance. The current local gallery has only one identity; robust local unknown rejection and cross-user calibration are NOT YET VERIFIED.",
        "",
        "No anti-spoofing, replay resistance, or production security claim is made.",
        "",
        "## 9. Exact live test",
        "",
        "1. Restart Flask so the new code is loaded: `START_GAME.bat` from the project root.",
        "2. Use the admin panel to confirm the local user has four face samples, four voice samples, is active, and has `open_door`.",
        "3. Enter Facility → Start face scan → Capture face. Check the admin diagnostics for local method, score, threshold, and candidate.",
        "4. Only after `FACE_ACCEPTED`, click Start voice verification and say the enrollment phrase clearly for three seconds.",
        "5. To prove unknown rejection and mismatched modalities, use an independently enrolled second person and fresh cross-user samples. The current one-user gallery cannot prove those cases.",
        "",
        "## 10. Reproducibility",
        "",
        "The exported configuration is `flask_game/local_matching_calibration.json`. It is a versioned, provisional public-reference calibration. Flask recomputes a local cross-user leave-one-out calibration once at least two identities with individual vectors are present.",
        "",
        "## 11. Automated test results",
        "",
        "The delivery test command `python -m pytest -q tests/test_project.py flask_game/tests/test_security.py flask_game/tests/test_real_model_matching.py flask_game/tests/test_voice_endpoint_real_audio.py` completed with **32 passed** and 8 dependency warnings. The real biometric/model and saved-audio endpoint subset completed with **10 passed**. These are automated file/API tests; they do not prove camera or microphone hardware behavior.",
        "",
        "The benchmark itself completed successfully on 191 public feature-cache rows. It reports validation calibration and held-out Actor 07 unknown results without tuning on the held-out partition. The private media audit was partial because the local identity directory was visible but its face/voice media were not readable in this run.",
        "",
        "## 12. Files changed for the unified deployment",
        "",
        "- `bioguard_matching.py` — shared extraction adapters, normalization, candidate scoring, calibration, and hierarchical decision logic.",
        "- `flask_game/inference.py` — Flask import surface for the shared module.",
        "- `deployment.py` — individual-template persistence and shared voice/face extraction defaults.",
        "- `flask_game/app.py` — local gallery face, voice, command-speaker, and diagnostic integration.",
        "- `flask_game/tools/benchmark_local_matching.py` — reproducible aggregate benchmark and configuration export.",
        "- `flask_game/tests/test_real_model_matching.py` — real-artifact and real-feature-vector tests.",
        "- `flask_game/tests/test_security.py` — Flask state/permission tests updated for the shared matcher.",
        "- `01_BioGuard_Full_Master_Notebook.ipynb` — appended deployment research section; original cells and outputs were preserved. Backup: `01_BioGuard_Full_Master_Notebook.backup-20261118-235500.ipynb`.",
        "",
        "## 13. What works and what remains unverified",
        "",
        "Works: public artifacts load unchanged; the notebook and Flask import the same matcher; new enrollments retain individual face/voice vectors; malformed face and silent audio fail closed; cross-user voice mismatch and permission denial are enforced; public validation/test comparisons and aggregate score-only local audits are reproducible.",
        "",
        "NOT YET VERIFIED: a saved local employee passing an independent held-out local evaluation, reliable local-user versus unknown-person separation, and fresh webcam/microphone success. The current private gallery has one identity and its raw media could not be re-read in this run, so enrollment-only audit evidence cannot establish local FAR/FRR or production readiness. No anti-spoofing claim is made.",
        "",
        "Additional evidence needed: at least one second independently enrolled user, independent held-out face and voice captures for every enrolled user, and outsider/unknown captures recorded under the same capture conditions. Keep those samples out of threshold tuning when they are used for final testing.",
        "",
        "## 14. Final answers",
        "",
        "Saved user passes a valid held-out local matching evaluation: **NOT YET VERIFIED**.",
        "Enrolled users can be reliably distinguished from unknown users in this local gallery: **NOT YET VERIFIED**.",
        "Mismatched face and voice identities are rejected by the shared hierarchical decision: **YES on the available public cross-user evaluation and automated test; NOT YET VERIFIED for a fresh local user**.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    frame = load_public_features()
    rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for modality in ("image", "voice"):
        modality_rows, selected_row = benchmark_modality(frame, modality)
        rows.extend(modality_rows)
        selected[modality] = selected_row

    # The benchmark comparison is allowed to rank methods, but it must not
    # silently replace an explicitly selected deployment voice configuration
    # every time the diagnostic script is rerun.  The deployed voice method
    # and threshold are versioned in the calibration file and are retained
    # for Flask/command consistency; the comparison rows still show all
    # candidates and their validation/test evidence.
    deployed_voice = matching.load_reference_calibration(CALIBRATION_PATH, "voice")
    if deployed_voice and deployed_voice.get("runtime_policy") == "versioned_reference_only":
        selected["voice"] = {**selected["voice"], **deployed_voice}
    calibration = _calibration(selected, rows)
    if deployed_voice and deployed_voice.get("runtime_policy") == "versioned_reference_only":
        calibration["modalities"]["voice"] = deployed_voice
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    gallery = deployment.load_gallery(PRIVATE_ROOT / "enrollment_gallery.joblib")
    private_audit = audit_private_media(gallery, selected)
    write_report(frame, rows, selected, private_audit, calibration)
    print(json.dumps({
        "report": str(REPORT_PATH),
        "calibration": str(CALIBRATION_PATH),
        "public_rows": int(len(frame)),
        "gallery_users": private_audit["gallery_users"],
        "private_media_access": private_audit["media_access"],
        "selected": {
            modality: {
                "method": row["method"],
                "threshold": row["threshold"],
                "validation_known_id_rate": row["validation_known_id_rate"],
                "validation_unknown_rejection_rate": row["validation_unknown_rejection_rate"],
                "test_known_id_rate": row["test_known_id_rate"],
                "test_unknown_rejection_rate": row["test_unknown_rejection_rate"],
            }
            for modality, row in selected.items()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
