from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bioguard_core as core
import bioguard_matching as shared_matching
from flask_game import inference as flask_inference
from flask_game.tools.benchmark_local_matching import load_public_features


def _public_gallery(identity_count: int = 2) -> tuple[object, dict]:
    frame = load_public_features()
    gallery = {"templates": {}, "sample_vectors": {}, "permissions": {}}
    enrollment = frame[frame.split == "enroll"]
    for identity in sorted(enrollment.user.unique())[:identity_count]:
        image = np.stack(enrollment[enrollment.user == identity].image_vec.to_list())
        voice = np.stack(enrollment[enrollment.user == identity].voice_vec.to_list())
        gallery["templates"][identity] = {
            "image": image.mean(axis=0),
            "voice": voice.mean(axis=0),
        }
        gallery["sample_vectors"][identity] = {"image": image, "voice": voice}
        gallery["permissions"][identity] = True
    return frame, gallery


def test_real_joblib_artifacts_load_and_shared_api_is_used():
    public = joblib.load(ROOT / "model" / "public_models.joblib")
    commands = joblib.load(ROOT / "model" / "commands_model.joblib")
    fingerprint = joblib.load(ROOT / "model" / "fingerprint_model.joblib")
    assert isinstance(public, dict) and public.get("models")
    assert isinstance(commands, dict)
    assert isinstance(fingerprint, dict)
    assert flask_inference.extract_face_live is shared_matching.extract_face_live
    assert flask_inference.extract_voice is shared_matching.extract_voice
    assert flask_inference.hierarchical_verify is shared_matching.hierarchical_verify


def test_saved_face_audit_reproduces_public_gate_failure_without_raw_media_output():
    path = ROOT / "flask_game" / "private_data" / "face_audit_scores.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(row["own_pass"].lower() == "false" for row in rows)
    assert max(float(row["own_public_score"]) for row in rows) < float(rows[0]["public_threshold"])
    assert max(float(row["stored_template_mean_error"]) for row in rows) == pytest.approx(0.0)


def test_voice_threshold_sources_and_runtime_mode_are_explicit():
    calibration_path = ROOT / "flask_game" / "local_matching_calibration.json"
    _, one_identity = _public_gallery(identity_count=1)
    one = shared_matching.calibration_for(one_identity, "voice", ROOT, calibration_path)
    assert one["source"] == "public_data_validation_plus_validation_unknown"
    assert one["method"] == "zcos_mean"
    assert one["threshold"] == pytest.approx(0.3908886485400538)
    assert one["runtime_policy"] == "versioned_reference_only"

    _, two_identities = _public_gallery(identity_count=2)
    two = shared_matching.calibration_for(two_identities, "voice", ROOT, calibration_path)
    assert two["source"] == "public_data_validation_plus_validation_unknown"
    assert two["method"] == "zcos_mean"
    assert two["threshold"] == pytest.approx(0.3908886485400538)
    assert two["runtime_policy"] == "versioned_reference_only"


def test_real_local_face_and_voice_matchers_use_individual_templates():
    frame, gallery = _public_gallery()
    calibration_path = ROOT / "flask_game" / "local_matching_calibration.json"
    image_calibration = shared_matching.calibration_for(
        gallery, "image", ROOT, calibration_path
    )
    voice_calibration = shared_matching.calibration_for(
        gallery, "voice", ROOT, calibration_path
    )
    known = frame[(frame.split == "test") & (frame.user == "actor_01")].iloc[0]
    face = shared_matching.rank_gallery(
        known.image_vec, gallery, "image", image_calibration, ROOT
    )
    assert face["candidate"] == "actor_01"
    assert face["ranking"][0]["sample_count"] >= 4
    assert np.isfinite(float(face["score"]))

    voice = shared_matching.verify_claim(
        known.voice_vec, gallery, "actor_01", "voice", voice_calibration, ROOT
    )
    assert np.isfinite(float(voice.get("claim_score", voice["score"])))


def test_real_cross_user_voice_is_rejected_after_face_candidate():
    frame, gallery = _public_gallery()
    image_calibration = shared_matching.calibration_for(
        gallery, "image", ROOT, ROOT / "flask_game" / "local_matching_calibration.json"
    )
    voice_calibration = shared_matching.calibration_for(
        gallery, "voice", ROOT, ROOT / "flask_game" / "local_matching_calibration.json"
    )
    found_mismatch = False
    for _, face_row in frame[(frame.split == "val") & (frame.user == "actor_01")].iterrows():
        face = shared_matching.rank_gallery(
            face_row.image_vec, gallery, "image", image_calibration, ROOT
        )
        if not face["accepted"] or face["identity"] != "actor_01":
            continue
        for _, voice_row in frame[(frame.split == "val") & (frame.user == "actor_02")].iterrows():
            voice = shared_matching.verify_claim(
                voice_row.voice_vec,
                gallery,
                "actor_01",
                "voice",
                voice_calibration,
                ROOT,
            )
            if not voice["accepted"] and voice["code"] == "VOICE_MISMATCH":
                found_mismatch = True
                break
        if found_mismatch:
            break
    assert found_mismatch


def test_real_hierarchical_permission_gate_and_same_input_result():
    frame, gallery = _public_gallery()
    calibration_path = ROOT / "flask_game" / "local_matching_calibration.json"
    # This is an enrollment-template plumbing smoke test, not an accuracy
    # claim: the independent held-out public clips are intentionally not used
    # to manufacture a successful door grant when this voice configuration
    # rejects them.  Held-out acceptance is covered by the benchmark report.
    enrollment = frame[(frame.split == "enroll") & (frame.user == "actor_01")].iloc[0]
    face_row, voice_row = enrollment, enrollment
    found = shared_matching.hierarchical_verify(
        face_row.image_vec,
        voice_row.voice_vec,
        gallery,
        ROOT,
        calibration_path,
        {"actor_01": True, "actor_02": True},
    )
    assert found["accepted"] is True
    denied = shared_matching.hierarchical_verify(
        face_row.image_vec,
        voice_row.voice_vec,
        gallery,
        ROOT,
        calibration_path,
        {"actor_01": False, "actor_02": True},
    )
    assert denied["accepted"] is False
    assert denied["code"] == "PERMISSION_DENIED"

    via_flask_import = flask_inference.hierarchical_verify(
        face_row.image_vec,
        voice_row.voice_vec,
        gallery,
        ROOT,
        calibration_path,
        {"actor_01": True, "actor_02": True},
    )
    assert via_flask_import["code"] == shared_matching.hierarchical_verify(
        face_row.image_vec,
        voice_row.voice_vec,
        gallery,
        ROOT,
        calibration_path,
        {"actor_01": True, "actor_02": True},
    )["code"]


def test_invalid_face_and_silent_audio_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        shared_matching.extract_face_live(np.zeros((32, 32), dtype=np.uint8))

    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(3 * core.SR, dtype=np.float32), core.SR)
    with pytest.raises(ValueError):
        shared_matching.extract_voice(silent)
