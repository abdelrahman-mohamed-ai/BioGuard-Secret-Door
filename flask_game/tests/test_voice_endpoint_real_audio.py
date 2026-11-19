from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
import io

import joblib
import numpy as np
import soundfile as sf
import importlib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask_game.app import create_app
from flask_game.tools.benchmark_local_matching import load_public_features

game_app = importlib.import_module("flask_game.app")


def _csrf(client) -> str:
    response = client.get("/game")
    match = re.search(rb'<meta name="csrf-token" content="([^"]+)"', response.data)
    assert match, response.data[:500]
    return match.group(1).decode()


def _public_gallery() -> dict:
    frame = load_public_features()
    enrollment = frame[frame.split == "enroll"]
    gallery = {"templates": {}, "sample_vectors": {}, "permissions": {}, "meta": {}}
    for identity in sorted(enrollment.user.unique())[:2]:
        faces = np.stack(enrollment[enrollment.user == identity].image_vec.to_list())
        voices = np.stack(enrollment[enrollment.user == identity].voice_vec.to_list())
        gallery["templates"][identity] = {
            "image": faces.mean(axis=0).astype("float32"),
            "voice": voices.mean(axis=0).astype("float32"),
        }
        gallery["sample_vectors"][identity] = {
            "image": faces.astype("float32"),
            "voice": voices.astype("float32"),
        }
        gallery["permissions"][identity] = True
        gallery["meta"][identity] = {
            "display_name": "Saved WAV test identity",
            "role": "operator",
            "active": True,
            "access_permissions": ["open_door"],
        }
    return gallery


def test_saved_wav_reaches_real_conversion_extraction_and_matcher(tmp_path, capsys):
    frame = load_public_features()
    held_out = frame[(frame.split == "test") & (frame.user == "actor_01")].iloc[0]
    saved_wav = Path(held_out.voice)
    gallery_path = tmp_path / "private" / "enrollment_gallery.joblib"
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_public_gallery(), gallery_path)
    app = create_app(
        {
            "TESTING": True,
            "REQUIRE_ENV_SECRETS": False,
            "ADMIN_PASSWORD": "voice-test-password",
            "SECRET_KEY": "voice-test-secret",
            "PRIVATE_ROOT": tmp_path / "private",
            "GALLERY_FILE": gallery_path,
            "MODEL_DIR": ROOT / "model",
        }
    )
    client = app.test_client()
    token = _csrf(client)
    with client.session_transaction() as session:
        # The face stage is intentionally bound as an already accepted server
        # ticket so this test isolates the post-recording voice path.
        session["face_candidate"] = "actor_01"
        session["face_candidate_at"] = time.time()
        session["face_score"] = 0.95
        session["admin_authenticated"] = True

    with saved_wav.open("rb") as handle:
        response = client.post(
            "/api/verify",
            headers={"X-CSRFToken": token},
            data={"voice_recording": (handle, "saved-test.wav")},
            content_type="multipart/form-data",
        )
    body = response.get_json()
    assert response.status_code == 200
    assert isinstance(body, dict)
    assert body.get("error_code") in {None, "VOICE_LOW_CONFIDENCE", "VOICE_MISMATCH", "VOICE_AMBIGUOUS", "PERMISSION_DENIED"}
    diagnostics = body.get("diagnostics") or {}
    print(json.dumps({
        "http_status": response.status_code,
        "accepted": body.get("accepted"),
        "error_code": body.get("error_code"),
        "failure_stage": body.get("failure_stage"),
        "voice_score": diagnostics.get("voice_score"),
        "voice_threshold": diagnostics.get("voice_threshold"),
        "detail": body.get("detail"),
    }, sort_keys=True))
    assert diagnostics.get("voice_score") is not None
    assert diagnostics.get("voice_threshold") is not None
    assert "Record Failed" not in json.dumps(body)


def test_real_audio_failures_return_distinct_codes(tmp_path):
    gallery_path = tmp_path / "private" / "enrollment_gallery.joblib"
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_public_gallery(), gallery_path)
    app = create_app(
        {
            "TESTING": True,
            "REQUIRE_ENV_SECRETS": False,
            "ADMIN_PASSWORD": "voice-test-password",
            "SECRET_KEY": "voice-test-secret",
            "PRIVATE_ROOT": tmp_path / "private",
            "GALLERY_FILE": gallery_path,
            "MODEL_DIR": ROOT / "model",
        }
    )
    client = app.test_client()
    token = _csrf(client)
    observed_codes = []

    def submit(data, filename):
        with client.session_transaction() as session:
            session["face_candidate"] = "actor_01"
            session["face_candidate_at"] = time.time()
        response = client.post(
            "/api/verify",
            headers={"X-CSRFToken": token},
            data={"voice_recording": (io.BytesIO(data), filename)},
            content_type="multipart/form-data",
        )
        return response.status_code, response.get_json()

    short_buffer = io.BytesIO()
    sf.write(short_buffer, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    status, body = submit(short_buffer.getvalue(), "short.wav")
    observed_codes.append(body["error_code"])
    assert status == 200 and body["error_code"] == "AUDIO_TOO_SHORT"

    silent_buffer = io.BytesIO()
    sf.write(silent_buffer, np.zeros(3 * 16000, dtype=np.float32), 16000, format="WAV")
    status, body = submit(silent_buffer.getvalue(), "silent.wav")
    observed_codes.append(body["error_code"])
    assert status == 200 and body["error_code"] == "AUDIO_SILENT"

    status, body = submit(b"not an audio stream", "broken.webm")
    observed_codes.append(body["error_code"])
    assert status == 200 and body["error_code"] == "AUDIO_DECODE_FAILED"

    status, body = submit(b"", "empty.wav")
    observed_codes.append(body["error_code"])
    assert status == 200 and body["error_code"] == "AUDIO_UPLOAD_FAILED"
    print(json.dumps({"audio_error_codes": observed_codes}, sort_keys=True))


def test_unexpected_voice_server_failure_returns_server_error(tmp_path, monkeypatch):
    frame = load_public_features()
    saved_wav = Path(frame[(frame.split == "test") & (frame.user == "actor_01")].iloc[0].voice)
    gallery_path = tmp_path / "private" / "enrollment_gallery.joblib"
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_public_gallery(), gallery_path)
    app = create_app(
        {
            "TESTING": True,
            "REQUIRE_ENV_SECRETS": False,
            "ADMIN_PASSWORD": "voice-test-password",
            "SECRET_KEY": "voice-test-secret",
            "PRIVATE_ROOT": tmp_path / "private",
            "GALLERY_FILE": gallery_path,
            "MODEL_DIR": ROOT / "model",
        }
    )
    client = app.test_client()
    token = _csrf(client)
    with client.session_transaction() as session:
        session["face_candidate"] = "actor_01"
        session["face_candidate_at"] = time.time()

    def crash(*_args, **_kwargs):
        raise RuntimeError("controlled test failure")

    monkeypatch.setattr(game_app.matching, "verify_claim", crash)
    with saved_wav.open("rb") as handle:
        response = client.post(
            "/api/verify",
            headers={"X-CSRFToken": token},
            data={"voice_recording": (handle, "saved-test.wav")},
            content_type="multipart/form-data",
        )
    body = response.get_json()
    assert response.status_code == 500
    assert body["error_code"] == "SERVER_ERROR"
