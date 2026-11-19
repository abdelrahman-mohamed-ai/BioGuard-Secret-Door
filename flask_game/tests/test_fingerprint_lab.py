from __future__ import annotations

import csv
import io
from pathlib import Path
import re

import joblib
import numpy as np
import pytest

import bioguard_core as core
import public_research
from flask_game import app as game_app
from flask_game import fingerprint_inference


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "model" / "fingerprint_model.joblib"
MANIFEST_PATH = ROOT / "fingerprint_data" / "selected" / "manifest.csv"


def _rows() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _path(row: dict[str, str]) -> Path:
    return ROOT / Path(row["path"])


def _gallery_row(kind: str) -> dict[str, str]:
    package = joblib.load(MODEL_PATH)
    gallery = set(package["demo_gallery"])
    return next(row for row in _rows() if row["kind"] == kind and row["finger"] in gallery)


def _unknown_real_row() -> dict[str, str]:
    package = joblib.load(MODEL_PATH)
    gallery = set(package["demo_gallery"])
    return next(row for row in _rows() if row["kind"] == "real" and row["finger"] not in gallery)


@pytest.fixture()
def app(tmp_path):
    return game_app.create_app(
        {
            "TESTING": True,
            "REQUIRE_ENV_SECRETS": False,
            "ADMIN_PASSWORD": "fingerprint-test-password",
            "SECRET_KEY": "fingerprint-test-secret",
            "PRIVATE_ROOT": tmp_path / "private",
            "GALLERY_FILE": tmp_path / "model" / "enrollment_gallery.joblib",
            "MODEL_DIR": ROOT / "model",
        }
    )


@pytest.fixture()
def client(app):
    return app.test_client()


def _csrf(client) -> str:
    response = client.get("/fingerprint")
    match = re.search(rb'<meta name="csrf-token" content="([^"]+)"', response.data)
    assert match, response.data[:400]
    return match.group(1).decode("utf-8")


def _post_image(client, image_path: Path, token: str, filename: str | None = None):
    with image_path.open("rb") as handle:
        return client.post(
            "/fingerprint",
            headers={"Accept": "application/json", "X-CSRFToken": token},
            data={
                "fingerprint_image": (io.BytesIO(handle.read()), filename or image_path.name),
            },
            content_type="multipart/form-data",
        )


def test_real_artifact_schema_and_prediction_visualizations():
    package = fingerprint_inference.load_fingerprint_package(MODEL_PATH)
    assert package["model_name"] == "SVM"
    assert float(package["threshold"]) == pytest.approx(0.8077645335329597)
    assert len(package["demo_gallery"]) == 25
    assert int(package["model"].n_features_in_) == 30

    row = _gallery_row("real")
    result = fingerprint_inference.verify_upload_with_visuals(_path(row), MODEL_PATH)
    assert result["ok"] is True
    assert result["feature_dim"] == core.FINGER_DIM == 29
    assert result["pair_feature_dim"] == 30
    assert result["match"] is True
    assert float(result["score"]) >= float(result["threshold"])
    assert all(result["visualizations"][name].startswith("data:image/png;base64,") for name in ("original", "grayscale", "sobel", "lbp"))


def test_real_same_finger_probe_and_unknown_real_are_decided_by_artifact():
    altered = _gallery_row("altered")
    unknown = _unknown_real_row()
    altered_result = fingerprint_inference.verify_upload_with_visuals(_path(altered), MODEL_PATH)
    unknown_result = fingerprint_inference.verify_upload_with_visuals(_path(unknown), MODEL_PATH)
    assert altered_result["match"] is True
    assert unknown_result["match"] is False
    assert float(unknown_result["score"]) < float(unknown_result["threshold"])


def test_cross_reference_pair_is_a_real_negative_case():
    package = fingerprint_inference.load_fingerprint_package(MODEL_PATH)
    row = _gallery_row("altered")
    query = core.extract_fingerprint(_path(row))
    scores = []
    for finger, reference in package["demo_gallery"].items():
        if finger == row["finger"]:
            continue
        pair = core.pair_difference(query, np.asarray(reference, dtype=np.float32))[None, :]
        scores.append(float(public_research.proba(package["model"], pair)[0]))
    assert scores
    assert min(scores) < float(package["threshold"])


def test_fingerprint_route_returns_real_json_and_keeps_upload_private(client, app):
    row = _gallery_row("real")
    before_static = {path.relative_to(game_app.GAME_ROOT / "static") for path in (game_app.GAME_ROOT / "static").rglob("*") if path.is_file()}
    token = _csrf(client)
    response = _post_image(client, _path(row), token)
    body = response.get_json()
    assert response.status_code == 200
    assert body["ok"] is True
    assert body["match"] is True
    assert body["model_name"] == "SVM"
    assert body["feature_dim"] == 29
    assert body["pair_feature_dim"] == 30
    assert set(body["visualizations"]) == {"original", "grayscale", "sobel", "lbp"}
    after_static = {path.relative_to(game_app.GAME_ROOT / "static") for path in (game_app.GAME_ROOT / "static").rglob("*") if path.is_file()}
    assert after_static == before_static
    assert not list((app.config["PRIVATE_ROOT"] / ".runtime").glob("finger-*"))


def test_route_accepts_repeated_real_uploads_without_reload(client):
    rows = [_gallery_row("real"), _gallery_row("altered")]
    token = _csrf(client)
    first = _post_image(client, _path(rows[0]), token)
    second = _post_image(client, _path(rows[1]), token)
    assert first.status_code == 200 and first.get_json()["ok"] is True
    assert second.status_code == 200 and second.get_json()["ok"] is True


def test_model_loading_failure_is_explicit(client, monkeypatch):
    def fail_load(_model_path):
        raise fingerprint_inference.FingerprintInferenceError(
            "MODEL_LOAD_FAILED", "The fingerprint model could not be loaded."
        )

    monkeypatch.setattr(fingerprint_inference, "load_fingerprint_package", fail_load)
    row = _gallery_row("real")
    token = _csrf(client)
    response = _post_image(client, _path(row), token)
    body = response.get_json()
    assert response.status_code == 503
    assert body["ok"] is False
    assert body["error_code"] == "MODEL_LOAD_FAILED"


@pytest.mark.parametrize(
    ("filename", "payload", "expected_code", "expected_status"),
    [
        ("", b"", "NO_FILE", 400),
        ("scan.txt", b"not an image", "INVALID_EXTENSION", 400),
        ("scan.png", b"not a png", "IMAGE_DECODE_FAILED", 400),
        ("scan.png", b"", "EMPTY_UPLOAD", 400),
    ],
)
def test_invalid_and_empty_uploads_fail_closed(client, filename, payload, expected_code, expected_status):
    token = _csrf(client)
    data = {"fingerprint_image": (io.BytesIO(payload), filename)} if filename else {}
    response = client.post(
        "/fingerprint",
        headers={"Accept": "application/json", "X-CSRFToken": token},
        data=data,
        content_type="multipart/form-data",
    )
    body = response.get_json()
    assert response.status_code == expected_status
    assert body["ok"] is False
    assert body["error_code"] == expected_code
    assert body["match"] is False


def test_get_page_exposes_preview_visualization_and_repeat_controls(client):
    response = client.get("/fingerprint")
    assert response.status_code == 200
    assert b"fingerprint-preview" in response.data
    assert b"fingerprint-original" in response.data
    assert b"fingerprint-new-upload" in response.data
    assert b"fingerprint.js" in response.data
    assert b"fingerprint.css" in response.data
