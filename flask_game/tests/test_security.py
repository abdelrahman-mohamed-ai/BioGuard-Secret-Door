from __future__ import annotations

import io
import importlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pytest


game_app = importlib.import_module("flask_game.app")


@pytest.fixture()
def app(tmp_path):
    application = game_app.create_app(
        {
            "TESTING": True,
            "REQUIRE_ENV_SECRETS": False,
            "ADMIN_PASSWORD": "unit-test-password",
            "SECRET_KEY": "unit-test-secret-key",
            "PRIVATE_ROOT": tmp_path / "private",
            "GALLERY_FILE": tmp_path / "model" / "enrollment_gallery.joblib",
            "MODEL_DIR": Path(game_app.PROJECT_ROOT) / "model",
        }
    )
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


def csrf(client):
    response = client.get("/admin/login")
    match = re.search(rb'name="csrf_token" value="([^"]+)"', response.data)
    assert match, response.data[:500]
    return match.group(1).decode()


def login(client):
    token = csrf(client)
    response = client.post(
        "/admin/login",
        data={"password": "unit-test-password", "csrf_token": token},
    )
    assert response.status_code == 302
    # Login clears the pre-login session and rotates the CSRF token.
    token = csrf(client)
    return token


def media_payload(face_count=4, voice_count=4, consent=True, identifier="abdo"):
    data = {
        "identifier": identifier,
        "display_name": "Abdo Test",
        "role": "operator",
        "active": "on",
        "access_permissions": ["open_door", "command_open", "command_close"],
        "face_images": [(io.BytesIO(b"face"), f"face-{i}.png") for i in range(face_count)],
        "voice_recordings": [(io.BytesIO(b"voice"), f"voice-{i}.wav") for i in range(voice_count)],
    }
    if consent:
        data["consent"] = "on"
    return data


def patch_media_validation(monkeypatch):
    monkeypatch.setattr(game_app, "_validate_face_file", lambda _path: None)

    def fake_convert(_source, target, min_seconds, command=False):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"local-test-audio")
        return target

    monkeypatch.setattr(game_app, "_convert_audio", fake_convert)


def fake_enroll_user(name, image_paths, audio_paths, allowed=True, gallery_file=None, **_kwargs):
    gallery_path = Path(gallery_file)
    gallery = joblib.load(gallery_path)
    gallery["templates"][name] = {
        "image": np.zeros(1774, dtype=np.float32),
        "voice": np.zeros(82, dtype=np.float32),
    }
    gallery["permissions"][name] = bool(allowed)
    gallery["meta"][name] = {
        "created_utc": "unit-test",
        "photos": len(image_paths),
        "recordings": len(audio_paths),
    }
    joblib.dump(gallery, gallery_path)
    return gallery["meta"][name]


def test_factory_starts_in_first_run_setup_without_secrets(monkeypatch, tmp_path):
    monkeypatch.delenv("BIOGUARD_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("BIOGUARD_SECRET_KEY", raising=False)
    application = game_app.create_app(
        {
            "TESTING": True,
            "PRIVATE_ROOT": tmp_path / "private",
            "GALLERY_FILE": tmp_path / "model" / "enrollment_gallery.joblib",
            "MODEL_DIR": Path(game_app.PROJECT_ROOT) / "model",
        }
    )
    assert application.config["SETUP_REQUIRED"] is True
    response = application.test_client().get("/admin/login")
    assert response.status_code == 302
    assert "/admin/setup" in response.headers["Location"]


def test_first_run_setup_persists_hashed_admin_secret(monkeypatch, tmp_path):
    monkeypatch.delenv("BIOGUARD_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("BIOGUARD_SECRET_KEY", raising=False)
    application = game_app.create_app(
        {
            "TESTING": True,
            "PRIVATE_ROOT": tmp_path / "private",
            "GALLERY_FILE": tmp_path / "model" / "enrollment_gallery.joblib",
            "MODEL_DIR": Path(game_app.PROJECT_ROOT) / "model",
        }
    )
    client = application.test_client()
    response = client.get("/admin/setup")
    match = re.search(rb'name="csrf_token" value="([^"]+)"', response.data)
    assert match
    token = match.group(1).decode()
    response = client.post(
        "/admin/setup",
        data={"password": "first-run-pin", "confirm_password": "first-run-pin", "csrf_token": token},
    )
    assert response.status_code == 302
    config_path = application.config["ADMIN_CONFIG_FILE"]
    assert config_path.is_file()
    assert "first-run-pin" not in config_path.read_text(encoding="utf-8")
    token = csrf(client)
    response = client.post("/admin/login", data={"password": "first-run-pin", "csrf_token": token})
    assert response.status_code == 302


def test_diagnostics_are_admin_only(client, monkeypatch):
    patch_media_validation(monkeypatch)
    monkeypatch.setattr(
        game_app.matching,
        "extract_face_live",
        lambda _path: np.zeros(1774, dtype=np.float32),
    )
    monkeypatch.setattr(
        game_app.matching,
        "extract_voice",
        lambda _path: np.zeros(82, dtype=np.float32),
    )
    monkeypatch.setattr(
        game_app.matching,
        "hierarchical_verify",
        lambda *args, **kwargs: {
            "accepted": False,
            "identity": "Unknown",
            "candidate": None,
            "reason": "Rejected",
            "code": "VOICE_LOW_CONFIDENCE",
            "face": {"score": 0.12, "threshold": 0.50, "distance": 1.0, "margin": None, "calibration": {"method": "test", "source": "test"}},
            "voice": {"score": 0.22, "claim_score": 0.22, "threshold": 0.50, "distance": 1.0, "margin": None, "calibration": {"method": "test", "source": "test"}},
        },
    )
    token = csrf(client)
    data = {
        "face_image": (io.BytesIO(b"face"), "capture.png"),
        "voice_recording": (io.BytesIO(b"voice"), "capture.wav"),
    }
    response = client.post("/api/verify", headers={"X-CSRFToken": token}, data=data, content_type="multipart/form-data")
    assert "diagnostics" not in response.get_json()
    token = login(client)
    data = {
        "face_image": (io.BytesIO(b"face"), "capture.png"),
        "voice_recording": (io.BytesIO(b"voice"), "capture.wav"),
    }
    response = client.post("/api/verify", headers={"X-CSRFToken": token}, data=data, content_type="multipart/form-data")
    assert response.get_json()["diagnostics"]["face_threshold"] > 0


def test_face_stage_rejects_unknown_before_voice(client, app, monkeypatch):
    patch_media_validation(monkeypatch)
    gallery = {
        "templates": {"user-one": {"image": np.zeros(1774, dtype=np.float32), "voice": np.zeros(82, dtype=np.float32)}},
        "permissions": {"user-one": True},
        "meta": {"user-one": {"display_name": "User One", "active": True}},
    }
    app.config["GALLERY_FILE"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gallery, app.config["GALLERY_FILE"])
    monkeypatch.setattr(
        game_app.matching,
        "extract_face_live",
        lambda _path: np.zeros(1774, dtype=np.float32),
    )
    monkeypatch.setattr(
        game_app.matching,
        "rank_gallery",
        lambda *args, **kwargs: {
            "accepted": False,
            "identity": "Unknown",
            "candidate": "user-one",
            "score": 0.02,
            "threshold": 0.5,
            "distance": 10.0,
            "margin": None,
            "code": "FACE_UNKNOWN",
            "reason": "Unknown face",
            "calibration": {"method": "test", "source": "test"},
            "ranking": [],
        },
    )
    token = csrf(client)
    response = client.post(
        "/api/face-candidate",
        headers={"X-CSRFToken": token},
        data={"face_image": (io.BytesIO(b"unknown-face"), "capture.png")},
        content_type="multipart/form-data",
    )
    body = response.get_json()
    assert body["accepted"] is False
    assert body["identity"] == "Unknown"
    assert body["failure_stage"] == "face"
    with client.session_transaction() as session:
        assert "face_candidate" not in session


def test_face_validation_failure_is_explained(client, monkeypatch):
    monkeypatch.setattr(game_app, "_validate_face_file", lambda _path: (_ for _ in ()).throw(ValueError("No frontal face detected")))
    token = csrf(client)
    response = client.post(
        "/api/face-candidate",
        headers={"X-CSRFToken": token},
        data={"face_image": (io.BytesIO(b"bad-face"), "capture.png")},
        content_type="multipart/form-data",
    )
    body = response.get_json()
    assert response.status_code == 200
    assert body["accepted"] is False
    assert body["failure_stage"] == "face"
    assert "No frontal face detected" in body["detail"]


def test_voice_validation_failure_is_explained_after_face_candidate(client, monkeypatch):
    monkeypatch.setattr(game_app, "_convert_audio", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Recording is too short")))
    with client.session_transaction() as session:
        session["face_candidate"] = "user-one"
        session["face_candidate_at"] = game_app.time.time()
        session["face_score"] = 0.81
    token = csrf(client)
    response = client.post(
        "/api/verify",
        headers={"X-CSRFToken": token},
        data={"voice_recording": (io.BytesIO(b"short"), "capture.webm")},
        content_type="multipart/form-data",
    )
    body = response.get_json()
    assert response.status_code == 200
    assert body["accepted"] is False
    assert body["failure_stage"] == "voice"
    assert "Recording is too short" in body["detail"]


def test_fingerprint_lab_reports_real_decision_fields(client):
    sample = game_app.PROJECT_ROOT / "fingerprint_data" / "selected" / "real" / "26b33b47a1a2_00.png"
    token = csrf(client)
    with sample.open("rb") as handle:
        response = client.post(
            "/fingerprint",
            data={
                "csrf_token": token,
                "fingerprint_image": (io.BytesIO(handle.read()), sample.name),
            },
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    assert b"REAL MODEL DECISION" in response.data
    assert b"MATCH" in response.data
    assert b"0.8078" in response.data


def test_voice_stage_uses_server_bound_face_candidate(client, app, monkeypatch):
    patch_media_validation(monkeypatch)
    gallery = {
        "templates": {"user-one": {"image": np.zeros(1774, dtype=np.float32), "voice": np.zeros(82, dtype=np.float32)}},
        "permissions": {"user-one": True},
        "meta": {"user-one": {"display_name": "User One", "active": True}},
    }
    app.config["GALLERY_FILE"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gallery, app.config["GALLERY_FILE"])
    monkeypatch.setattr(game_app.matching, "extract_face_live", lambda _path: np.zeros(1774, dtype=np.float32))
    monkeypatch.setattr(game_app.matching, "extract_voice", lambda _path: np.zeros(82, dtype=np.float32))
    monkeypatch.setattr(
        game_app.matching,
        "rank_gallery",
        lambda *args, **kwargs: {"accepted": True, "identity": "user-one", "candidate": "user-one", "score": 0.8, "threshold": 0.5, "distance": 0.2, "margin": None, "code": "FACE_ACCEPTED", "calibration": {"method": "test", "source": "test"}, "ranking": []},
    )
    monkeypatch.setattr(
        game_app.matching,
        "verify_claim",
        lambda *args, **kwargs: {"accepted": True, "identity": "user-one", "candidate": "user-one", "score": 0.9, "claim_score": 0.9, "threshold": 0.5, "distance": 0.1, "margin": None, "code": "VOICE_ACCEPTED", "calibration": {"method": "test", "source": "test"}, "ranking": []},
    )
    token = csrf(client)
    response = client.post(
        "/api/face-candidate",
        headers={"X-CSRFToken": token},
        data={"face_image": (io.BytesIO(b"face"), "capture.png")},
        content_type="multipart/form-data",
    )
    assert response.get_json()["accepted"] is True
    token = csrf(client)
    response = client.post(
        "/api/verify",
        headers={"X-CSRFToken": token},
        data={"voice_recording": (io.BytesIO(b"voice"), "capture.wav")},
        content_type="multipart/form-data",
    )
    assert response.get_json()["accepted"] is True
    assert response.get_json()["identity"] == "User One"


def test_admin_mutations_are_protected(client):
    response = client.post("/api/admin/enroll")
    assert response.status_code == 401
    response = client.get("/admin/users")
    assert response.status_code == 302


def test_admin_diagnostics_show_gallery_and_thresholds(client):
    login(client)
    response = client.get("/api/admin/diagnostics")
    assert response.status_code == 200
    body = response.get_json()
    assert body["enrolled_users"] == 0
    assert body["current_auth_state"] == "IDLE"
    assert body["face_threshold"] > 0
    assert body["voice_threshold"] > 0


def test_game_page_exposes_explicit_face_and_voice_actions(client):
    response = client.get("/game")
    assert response.status_code == 200
    assert b"Start face scan" in response.data
    assert b"Capture face" in response.data
    assert b"Start voice verification" in response.data
    assert b"FACE VERIFIED" in client.get("/static/js/game.js").data


def test_empty_gallery_denies_without_identity_leak(client, monkeypatch):
    patch_media_validation(monkeypatch)
    monkeypatch.setattr(
        game_app.deployment,
        "verify_new_attempt",
        lambda *args, **kwargs: {"accepted": False, "identity": "Unknown", "reason": "No users enrolled"},
    )
    token = csrf(client)
    response = client.post(
        "/api/verify",
        headers={"X-CSRFToken": token},
        data={
            "face_image": (io.BytesIO(b"face"), "capture.png"),
            "voice_recording": (io.BytesIO(b"voice"), "capture.wav"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["accepted"] is False
    assert body["identity"] == "Unknown"
    assert body["reason"] == "Access Denied"


def test_enrollment_needs_four_of_each_and_consent(client):
    token = login(client)
    too_few = media_payload(face_count=3, voice_count=3, consent=True)
    too_few["csrf_token"] = token
    response = client.post("/admin/enroll", data=too_few, content_type="multipart/form-data")
    assert response.status_code == 400
    no_consent = media_payload(consent=False, identifier="ziad")
    no_consent["csrf_token"] = token
    response = client.post("/admin/enroll", data=no_consent, content_type="multipart/form-data")
    assert response.status_code == 400


def test_api_enrollment_reports_only_failed_face_slot_and_keeps_gallery_empty(client, app, monkeypatch):
    token = login(client)

    def validate_face(path):
        if path.name.startswith("face-01-"):
            raise ValueError("Face image has too little contrast")

    patch_media_validation(monkeypatch)
    monkeypatch.setattr(game_app, "_validate_face_file", validate_face)
    payload = media_payload(identifier="slot-face")
    response = client.post(
        "/api/admin/enroll",
        headers={"X-CSRFToken": token},
        data=payload,
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    body = response.get_json()
    assert body["errors"]["face.0"]["code"] == "FACE_INVALID"
    assert "Face slot 1" in body["errors"]["face.0"]["message"]
    assert set(game_app.deployment.load_gallery(app.config["GALLERY_FILE"])["templates"]) == set()


def test_api_enrollment_reports_only_failed_voice_slot_and_keeps_faces_uncommitted(client, app, monkeypatch):
    token = login(client)
    monkeypatch.setattr(game_app, "_validate_face_file", lambda _path: None)

    def convert(source, target, min_seconds, command=False):
        if source.name.startswith("voice-raw-02-"):
            raise game_app.AudioProcessingError("AUDIO_SILENT", "No usable speech was detected")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"local-test-audio")
        return target

    monkeypatch.setattr(game_app, "_convert_audio", convert)
    payload = media_payload(identifier="slot-voice")
    response = client.post(
        "/api/admin/enroll",
        headers={"X-CSRFToken": token},
        data=payload,
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    body = response.get_json()
    assert body["errors"]["voice.1"]["code"] == "AUDIO_SILENT"
    assert "Voice slot 2" in body["errors"]["voice.1"]["message"]
    assert set(game_app.deployment.load_gallery(app.config["GALLERY_FILE"])["templates"]) == set()


def test_normal_enrollment_error_rerenders_user_information(client, monkeypatch):
    token = login(client)
    patch_media_validation(monkeypatch)
    monkeypatch.setattr(
        game_app,
        "_validate_face_file",
        lambda path: (_ for _ in ()).throw(ValueError("Lighting is too low")) if path.name.startswith("face-01-") else None,
    )
    payload = media_payload(identifier="retain-info")
    payload["display_name"] = "Information Remains"
    payload["csrf_token"] = token
    response = client.post("/admin/enroll", data=payload, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b'value="Information Remains"' in response.data
    assert b'value="retain-info"' in response.data
    assert b"Face slot 1" in response.data


def test_enrollment_commit_rolls_back_media_when_gallery_commit_fails(client, app, monkeypatch):
    patch_media_validation(monkeypatch)
    monkeypatch.setattr(game_app.deployment, "enroll_user", fake_enroll_user)
    monkeypatch.setattr(game_app, "_atomic_dump_gallery", lambda _gallery: (_ for _ in ()).throw(RuntimeError("controlled commit failure")))
    token = login(client)
    payload = media_payload(identifier="rollback-user")
    payload["csrf_token"] = token
    response = client.post("/admin/enroll", data=payload, content_type="multipart/form-data")
    assert response.status_code == 500
    gallery = game_app.deployment.load_gallery(app.config["GALLERY_FILE"])
    assert "rollback-user" not in gallery["templates"]
    assert not (app.config["PRIVATE_ROOT"] / "users" / "rollback-user").exists()


def test_enrollment_page_keeps_information_controls_and_slot_draft_hooks(client):
    token = login(client)
    response = client.get("/admin/enroll")
    assert response.status_code == 200
    assert b'id="user-information"' in response.data
    assert b'data-api="/api/admin/enroll"' in response.data
    assert b'novalidate' in response.data
    assert b'face-images' in response.data and b'voice-recordings' in response.data
    javascript = client.get("/static/js/enrollment.js").data
    assert b"const draft" in javascript
    assert b"applyServerErrors" in javascript
    assert b"Nothing was saved" in javascript


def test_enrollment_updates_gallery_and_disabled_user_fails_closed(client, app, monkeypatch):
    patch_media_validation(monkeypatch)
    monkeypatch.setattr(game_app.deployment, "enroll_user", fake_enroll_user)
    token = login(client)
    payload = media_payload()
    payload["csrf_token"] = token
    response = client.post("/admin/enroll", data=payload, content_type="multipart/form-data")
    assert response.status_code == 302
    gallery = joblib.load(app.config["GALLERY_FILE"])
    assert set(gallery["templates"]) == {"abdo"}
    assert gallery["meta"]["abdo"]["display_name"] == "Abdo Test"
    assert (app.config["PRIVATE_ROOT"] / "users" / "abdo" / "faces").is_dir()
    assert len(list((app.config["PRIVATE_ROOT"] / "users" / "abdo" / "faces").iterdir())) == 4

    token = csrf(client)
    response = client.post(
        "/api/admin/users/abdo",
        headers={"X-CSRFToken": token},
        json={"active": False, "access_permissions": []},
    )
    assert response.status_code == 200
    gallery = joblib.load(app.config["GALLERY_FILE"])
    assert gallery["permissions"]["abdo"] is False


def test_api_enrollment_success_returns_confirmed_redirect(client, app, monkeypatch):
    patch_media_validation(monkeypatch)
    monkeypatch.setattr(game_app.deployment, "enroll_user", fake_enroll_user)
    token = login(client)
    payload = media_payload(identifier="api-save")
    response = client.post(
        "/api/admin/enroll",
        headers={"X-CSRFToken": token},
        data=payload,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["redirect"].endswith("/admin/users")
    gallery = joblib.load(app.config["GALLERY_FILE"])
    assert set(gallery["templates"]) == {"api-save"}


def test_face_voice_mismatch_is_uniformly_denied(client, monkeypatch):
    patch_media_validation(monkeypatch)
    monkeypatch.setattr(
        game_app.deployment,
        "verify_new_attempt",
        lambda *args, **kwargs: {"accepted": False, "identity": "Unknown", "reason": "Voice rejected"},
    )
    token = csrf(client)
    response = client.post(
        "/api/verify",
        headers={"X-CSRFToken": token},
        data={
            "face_image": (io.BytesIO(b"ziad-face"), "ziad.png"),
            "voice_recording": (io.BytesIO(b"abdo-voice"), "abdo.wav"),
        },
        content_type="multipart/form-data",
    )
    body = response.get_json()
    assert body["accepted"] is False
    assert body["identity"] == "Unknown"
    assert "Voice" not in json.dumps(body)


def test_command_is_independent_of_face_session_but_requires_media(client, monkeypatch):
    token = csrf(client)
    response = client.post(
        "/api/command",
        headers={"X-CSRFToken": token},
        data={
            "speaker_audio": (io.BytesIO(b"speaker"), "speaker.wav"),
            "command_audio": (io.BytesIO(b"go"), "go.wav"),
        },
        content_type="multipart/form-data",
    )
    # The command endpoint must not demand a face/door ticket. Media processing
    # may fail without a trained model; it must not say "Verify at the door first".
    assert response.status_code != 401
    assert "Verify at the door first" not in json.dumps(response.get_json())


def test_command_uses_separate_speaker_gate_and_server_permission(client, app, monkeypatch):
    patch_media_validation(monkeypatch)
    gallery = {
        "templates": {"abdo": {"voice": np.zeros(82, dtype=np.float32)}},
        "permissions": {"abdo": True},
        "meta": {"abdo": {"display_name": "Abdo Test", "access_permissions": ["command_open"]}},
    }
    app.config["GALLERY_FILE"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gallery, app.config["GALLERY_FILE"])
    monkeypatch.setattr(game_app.matching, "extract_voice", lambda _path: np.zeros(82, dtype=np.float32))
    monkeypatch.setattr(
        game_app.matching,
        "rank_gallery",
        lambda *args, **kwargs: {"accepted": True, "identity": "abdo", "candidate": "abdo", "score": 0.9, "threshold": 0.5, "distance": 0.1, "margin": None, "code": "VOICE_ACCEPTED", "calibration": {"method": "test", "source": "test"}, "ranking": []},
    )
    monkeypatch.setattr(
        game_app.commands_research,
        "predict_command",
        lambda *args, **kwargs: {"recognized": True, "word": "go", "action": "open", "score": 0.9},
    )
    # No previous face/camera login: voice-only is an independent mode.
    token = csrf(client)

    def fake_convert(_source, target, min_seconds, command=False):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"command")
        return target

    monkeypatch.setattr(game_app, "_convert_audio", fake_convert)
    response = client.post(
        "/api/command",
        headers={"X-CSRFToken": token},
        data={
            "speaker_audio": (io.BytesIO(b"speaker"), "speaker.wav"),
            "command_audio": (io.BytesIO(b"go"), "go.wav"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.get_json()["accepted"] is True

    monkeypatch.setattr(
        game_app.commands_research,
        "predict_command",
        lambda *args, **kwargs: {"recognized": True, "word": "stop", "action": "close", "score": 0.9},
    )
    response = client.post(
        "/api/command",
        headers={"X-CSRFToken": token},
        data={
            "speaker_audio": (io.BytesIO(b"speaker"), "speaker.wav"),
            "command_audio": (io.BytesIO(b"stop"), "stop.wav"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.get_json()["accepted"] is False

