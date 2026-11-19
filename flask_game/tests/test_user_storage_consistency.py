from __future__ import annotations

import io
import re
from pathlib import Path

import joblib
import numpy as np
import pytest

from flask_game import app as game_app


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def app(tmp_path):
    return game_app.create_app(
        {
            "TESTING": True,
            "REQUIRE_ENV_SECRETS": False,
            "ADMIN_PASSWORD": "storage-test-password",
            "SECRET_KEY": "storage-test-secret",
            "PRIVATE_ROOT": tmp_path / "private",
            "GALLERY_FILE": tmp_path / "model" / "enrollment_gallery.joblib",
            "MODEL_DIR": ROOT / "model",
        }
    )


@pytest.fixture()
def client(app):
    return app.test_client()


def _csrf(client) -> str:
    response = client.get("/admin/users")
    match = re.search(rb'<meta name="csrf-token" content="([^"]+)"', response.data)
    assert match, response.data[:500]
    return match.group(1).decode("utf-8")


def _token_from(response) -> str:
    match = re.search(rb'<meta name="csrf-token" content="([^"]+)"', response.data)
    if not match:
        match = re.search(rb'name="csrf_token" value="([^"]+)"', response.data)
    assert match, response.data[:500]
    return match.group(1).decode("utf-8")


def _login(client) -> str:
    response = client.get("/admin/login")
    token = _token_from(response)
    response = client.post(
        "/admin/login",
        data={"password": "storage-test-password", "csrf_token": token},
    )
    assert response.status_code == 302
    return _csrf(client)


def _payload(identifier: str) -> dict:
    return {
        "identifier": identifier,
        "display_name": f"Temporary {identifier}",
        "role": "operator",
        "active": "on",
        "access_permissions": ["open_door", "command_open"],
        "consent": "on",
        "face_images": [(io.BytesIO(b"face"), f"{identifier}-face-{index}.png") for index in range(4)],
        "voice_recordings": [(io.BytesIO(b"voice"), f"{identifier}-voice-{index}.wav") for index in range(4)],
    }


def _patch_storage_only_enrollment(monkeypatch):
    monkeypatch.setattr(game_app, "_validate_face_file", lambda _path: None)

    def fake_convert(_source, target, min_seconds, command=False):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"temporary-audio")
        return target

    def fake_enroll_user(name, image_paths, audio_paths, allowed=True, gallery_file=None, **_kwargs):
        path = Path(gallery_file)
        gallery = joblib.load(path)
        gallery.setdefault("sample_vectors", {})
        gallery["templates"][name] = {
            "image": np.zeros(1774, dtype=np.float32),
            "voice": np.zeros(82, dtype=np.float32),
        }
        gallery["sample_vectors"][name] = {
            "image": np.zeros((4, 1774), dtype=np.float32),
            "voice": np.zeros((4, 82), dtype=np.float32),
        }
        gallery["permissions"][name] = bool(allowed)
        gallery["meta"][name] = {
            "created_utc": "storage-test",
            "photos": len(image_paths),
            "recordings": len(audio_paths),
        }
        joblib.dump(gallery, path)
        return gallery["meta"][name]

    monkeypatch.setattr(game_app, "_convert_audio", fake_convert)
    monkeypatch.setattr(game_app.deployment, "enroll_user", fake_enroll_user)


def _enroll(client, identifier: str, token: str):
    response = client.post(
        "/admin/enroll",
        data={**_payload(identifier), "csrf_token": token},
        content_type="multipart/form-data",
    )
    assert response.status_code == 302
    return _csrf(client)


def _delete_json(client, identifier: str, token: str):
    return client.post(
        f"/api/admin/users/{identifier}",
        headers={"Accept": "application/json", "X-CSRFToken": token},
        data={"action": "delete", "csrf_token": token},
    )


def test_delete_restart_and_reenroll_same_identifier_preserves_other_user(client, app, monkeypatch):
    _patch_storage_only_enrollment(monkeypatch)
    token = _login(client)
    token = _enroll(client, "temporary-a", token)
    token = _enroll(client, "temporary-b", token)

    private_root = app.config["PRIVATE_ROOT"] / "users"
    deleted_target = private_root / "temporary-a"
    preserved_target = private_root / "temporary-b"
    assert deleted_target.is_dir()
    assert preserved_target.is_dir()

    response = _delete_json(client, "temporary-a", _csrf(client))
    body = response.get_json()
    assert response.status_code == 200
    assert body["ok"] is True
    assert body["storage_cleanup"] == "complete"
    assert not deleted_target.exists()
    assert preserved_target.is_dir()

    gallery = joblib.load(app.config["GALLERY_FILE"])
    assert "temporary-a" not in gallery["templates"]
    assert "temporary-a" not in gallery["permissions"]
    assert "temporary-a" not in gallery["meta"]
    assert "temporary-b" in gallery["templates"]
    assert "temporary-b" in gallery["permissions"]

    restarted = game_app.create_app(app.config.copy())
    restarted_client = restarted.test_client()
    token = _login(restarted_client)
    _enroll(restarted_client, "temporary-a", token)
    assert (restarted.config["PRIVATE_ROOT"] / "users" / "temporary-a").is_dir()
    restarted_gallery = joblib.load(restarted.config["GALLERY_FILE"])
    assert set(restarted_gallery["templates"]) == {"temporary-a", "temporary-b"}


def test_delete_failure_keeps_registry_and_private_storage_consistent(client, app, monkeypatch):
    _patch_storage_only_enrollment(monkeypatch)
    _enroll(client, "rollback-user", _login(client))
    target = app.config["PRIVATE_ROOT"] / "users" / "rollback-user"
    before = joblib.load(app.config["GALLERY_FILE"])

    def fail_gallery_dump(_gallery):
        raise OSError("simulated registry write failure")

    monkeypatch.setattr(game_app, "_atomic_dump_gallery", fail_gallery_dump)
    response = _delete_json(client, "rollback-user", _csrf(client))
    body = response.get_json()
    assert response.status_code == 500
    assert body["ok"] is False
    assert body["error_code"] == "STORAGE_DELETE_FAILED"
    assert target.is_dir()

    after = joblib.load(app.config["GALLERY_FILE"])
    assert set(after["templates"]) == set(before["templates"]) == {"rollback-user"}
    assert set(after["permissions"]) == set(before["permissions"]) == {"rollback-user"}


def test_private_cleanup_failure_is_not_reported_as_success(client, app, monkeypatch):
    _patch_storage_only_enrollment(monkeypatch)
    _enroll(client, "cleanup-user", _login(client))
    target = app.config["PRIVATE_ROOT"] / "users" / "cleanup-user"

    monkeypatch.setattr(game_app, "_safe_remove", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("simulated cleanup failure")))
    response = _delete_json(client, "cleanup-user", _csrf(client))
    body = response.get_json()
    assert response.status_code == 500
    assert body["ok"] is False
    assert body["error_code"] == "PRIVATE_STORAGE_CLEANUP_PENDING"
    assert body["storage_cleanup"] == "pending"
    assert not target.exists()
    gallery = joblib.load(app.config["GALLERY_FILE"])
    assert "cleanup-user" not in gallery["templates"]
    assert list((app.config["PRIVATE_ROOT"] / ".backups" / "deleted-users").glob("cleanup-user-*.pending"))


def test_orphan_requires_admin_recovery_before_reenrollment(client, app, monkeypatch):
    _patch_storage_only_enrollment(monkeypatch)
    gallery_path = app.config["GALLERY_FILE"]
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"templates": {}, "permissions": {}, "meta": {}, "sample_vectors": {}}, gallery_path)
    orphan = app.config["PRIVATE_ROOT"] / "users" / "orphan-user"
    (orphan / "faces").mkdir(parents=True, exist_ok=True)
    (orphan / "voices").mkdir(parents=True, exist_ok=True)
    (orphan / "faces" / "marker.bin").write_bytes(b"temporary-marker")

    token = _login(client)
    page = client.get("/admin/users")
    assert page.status_code == 200
    assert b"orphan-user" in page.data
    assert b"Orphaned private enrollment storage" in page.data

    blocked = client.post(
        "/api/admin/enroll",
        headers={"Accept": "application/json", "X-CSRFToken": token},
        data={**_payload("orphan-user"), "csrf_token": token},
        content_type="multipart/form-data",
    )
    assert blocked.status_code == 409
    assert blocked.get_json()["error_code"] == "ORPHANED_PRIVATE_STORAGE"
    assert orphan.is_dir()

    no_confirmation = client.post(
        "/admin/storage-recovery/orphan-user",
        headers={"Accept": "application/json", "X-CSRFToken": token},
        data={"csrf_token": token},
    )
    assert no_confirmation.status_code == 400
    assert no_confirmation.get_json()["error_code"] == "EXPLICIT_CONFIRMATION_REQUIRED"
    assert orphan.is_dir()

    recovered = client.post(
        "/admin/storage-recovery/orphan-user",
        headers={"Accept": "application/json", "X-CSRFToken": token},
        data={"csrf_token": token, "confirm": "yes"},
    )
    recovered_body = recovered.get_json()
    assert recovered.status_code == 200
    assert recovered_body["ok"] is True
    assert not orphan.exists()
    quarantined = list((app.config["PRIVATE_ROOT"] / ".backups" / "orphaned-users").glob("orphan-user-*.quarantine"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "faces" / "marker.bin").is_file()

    token = _csrf(client)
    enrolled = client.post(
        "/api/admin/enroll",
        headers={"Accept": "application/json", "X-CSRFToken": token},
        data={**_payload("orphan-user"), "csrf_token": token},
        content_type="multipart/form-data",
    )
    assert enrolled.status_code == 200
    assert enrolled.get_json()["ok"] is True
    assert (app.config["PRIVATE_ROOT"] / "users" / "orphan-user").is_dir()
    final_gallery = joblib.load(app.config["GALLERY_FILE"])
    assert "orphan-user" in final_gallery["templates"]
