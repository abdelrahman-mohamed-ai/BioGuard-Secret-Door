"""BioGuard Secret Door: a local Flask experience around the trained models.

The application deliberately keeps the public research models fixed.  New
people are represented by templates computed with the same HOG/LBP and
MFCC/ZCR/RMS extractors used by the research code; no classifier is retrained
when an employee is enrolled.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename

import joblib
import numpy as np
import soundfile as sf

GAME_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = GAME_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bioguard_core as core  # noqa: E402
import bioguard_matching as matching  # noqa: E402
import commands_bonus  # noqa: E402
import deployment  # noqa: E402
import fingerprint_research  # noqa: E402
import public_research  # noqa: E402
import speech_commands_research as commands_research  # noqa: E402

try:
    from . import fingerprint_inference as fingerprint_adapter  # noqa: E402
except ImportError:  # Allows `python flask_game/app.py` from PowerShell.
    import fingerprint_inference as fingerprint_adapter  # type: ignore[no-redef]

try:
    import cv2
except Exception:  # pragma: no cover - the dependency is in requirements
    cv2 = None

try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover - surfaced as a setup error when needed
    imageio_ffmpeg = None

try:
    from .auth import (
        admin_password_configured,
        admin_required,
        check_password,
        csrf_is_valid,
        csrf_token,
        log_in_admin,
        log_out_admin,
        rate_limit,
        require_csrf,
    )
    from .inference import extract_face_live, crop_single_face
except ImportError:  # Allows `python flask_game/app.py` from PowerShell.
    from auth import (  # type: ignore[no-redef]
        admin_password_configured,
        admin_required,
        check_password,
        csrf_is_valid,
        csrf_token,
        log_in_admin,
        log_out_admin,
        rate_limit,
        require_csrf,
    )
    from inference import extract_face_live, crop_single_face  # type: ignore[no-redef]


ALLOWED_FACE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FINGERPRINT_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".aac"}
ACCESS_PERMISSIONS = {
    "open_door": "Open the Secret Door",
    "command_open": "Voice command: GO",
    "command_close": "Voice command: STOP",
    "view_status": "View system status",
}
ROLES = ("operator", "supervisor", "security", "observer")
USERNAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{2,40}\Z")


class AudioProcessingError(ValueError):
    """A safe, user-visible audio failure with a stable API error code."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(detail)


class EnrollmentValidationError(ValueError):
    """Per-field/per-slot enrollment errors that are safe to return to the UI."""

    def __init__(self, errors: dict[str, dict[str, str]], detail: str = "Enrollment needs attention"):
        self.errors = errors
        super().__init__(detail)


class StorageConsistencyError(ValueError):
    """A safe, administrator-visible enrollment storage failure."""

    def __init__(self, code: str, detail: str, status: int = 409):
        self.code = code
        self.detail = detail
        self.status = status
        super().__init__(detail)


def _audio_error_from_value_error(exc: Exception) -> AudioProcessingError:
    """Map extractor/converter failures without exposing filesystem details."""
    message = str(exc).lower()
    if "silent" in message or "too quiet" in message:
        return AudioProcessingError("AUDIO_SILENT", "No usable speech was detected. Record again closer to the microphone.")
    if "too short" in message or "short" in message:
        return AudioProcessingError("AUDIO_TOO_SHORT", "The recording was too short. Record the full voice phrase.")
    if "decode" in message or "audio" in message or "wav" in message:
        return AudioProcessingError("AUDIO_DECODE_FAILED", "The recording could not be decoded locally. Use WAV or record again.")
    return AudioProcessingError("AUDIO_DECODE_FAILED", "The recording could not be processed locally. Use WAV or record again.")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(value) -> Path:
    return Path(value).expanduser().resolve()


def _module_app() -> Flask:
    """Keep ``flask_game.app:app`` importable for test discovery.

    A real launch still calls ``create_app()`` without overrides.  On a clean
    installation that intentionally opens the local first-run Admin setup;
    only an unexpected factory error is converted into an import-safe object.
    """
    try:
        return create_app()
    except RuntimeError as exc:
        return create_app(
            {
                "SECRET_KEY": secrets.token_urlsafe(32),
                "ADMIN_PASSWORD": "",
                "REQUIRE_ENV_SECRETS": False,
                "SECURITY_CONFIGURATION_ERROR": True,
                "SECURITY_CONFIGURATION_MESSAGE": str(exc),
            }
        )


def create_app(config: dict | None = None) -> Flask:
    """Create a configured application.

    ``config`` is intentionally explicit in tests so tests never need to put a
    real administrator password in a process-wide environment variable.  A
    normal first launch persists a hashed Admin PIN in ``private_data``.
    """
    env_password = os.environ.get("BIOGUARD_ADMIN_PASSWORD")
    env_secret = os.environ.get("BIOGUARD_SECRET_KEY")
    defaults = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "GAME_ROOT": GAME_ROOT,
        "MODEL_DIR": PROJECT_ROOT / "model",
        "PRIVATE_ROOT": GAME_ROOT / "private_data",
        # Enrollment templates are user biometric data, so the Flask game
        # keeps them beside the private media rather than beside public model
        # artifacts.  The research artifacts under model/ remain untouched.
        "GALLERY_FILE": GAME_ROOT / "private_data" / "enrollment_gallery.joblib",
        "ADMIN_CONFIG_FILE": GAME_ROOT / "private_data" / "admin_config.json",
        "ADMIN_PASSWORD": env_password,
        "ADMIN_PASSWORD_HASH": None,
        "SECRET_KEY": env_secret,
        "REQUIRE_ENV_SECRETS": False,
        "SECURITY_CONFIGURATION_ERROR": False,
        "SECURITY_CONFIGURATION_MESSAGE": "",
        "MAX_CONTENT_LENGTH": 64 * 1024 * 1024,
        "MAX_VERIFY_ATTEMPTS": 5,
        "VERIFY_WINDOW_SECONDS": 300,
        "COMMAND_WINDOW_SECONDS": 60,
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_SECURE": False,
        "TESTING": False,
    }
    if config:
        defaults.update(config)
        if "PRIVATE_ROOT" in config and "ADMIN_CONFIG_FILE" not in config:
            defaults["ADMIN_CONFIG_FILE"] = Path(config["PRIVATE_ROOT"]) / "admin_config.json"
    if defaults["REQUIRE_ENV_SECRETS"] and (not defaults.get("ADMIN_PASSWORD") or not defaults.get("SECRET_KEY")):
        raise RuntimeError(
            "BioGuard requires BIOGUARD_ADMIN_PASSWORD and BIOGUARD_SECRET_KEY "
            "before starting. Set both in PowerShell."
        )

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_mapping(defaults)
    for key in ("PROJECT_ROOT", "GAME_ROOT", "MODEL_DIR", "PRIVATE_ROOT", "GALLERY_FILE", "ADMIN_CONFIG_FILE"):
        app.config[key] = _path(app.config[key])
    app.config["PRIVATE_ROOT"].mkdir(parents=True, exist_ok=True)
    (app.config["PRIVATE_ROOT"] / "users").mkdir(parents=True, exist_ok=True)
    (app.config["PRIVATE_ROOT"] / ".runtime").mkdir(parents=True, exist_ok=True)
    (app.config["PRIVATE_ROOT"] / ".staging").mkdir(parents=True, exist_ok=True)

    stored_admin = _read_json(app.config["ADMIN_CONFIG_FILE"])
    if not app.config.get("ADMIN_PASSWORD") and not app.config.get("ADMIN_PASSWORD_HASH"):
        app.config["ADMIN_PASSWORD_HASH"] = stored_admin.get("password_hash") or None
    app.config["SETUP_REQUIRED"] = not bool(
        app.config.get("ADMIN_PASSWORD") or app.config.get("ADMIN_PASSWORD_HASH")
    )
    if not app.config.get("SECRET_KEY"):
        secret_file = app.config["PRIVATE_ROOT"] / "session_secret.txt"
        if secret_file.is_file():
            app.config["SECRET_KEY"] = secret_file.read_text(encoding="utf-8").strip()
        if not app.config.get("SECRET_KEY"):
            app.config["SECRET_KEY"] = secrets.token_urlsafe(48)
            secret_file.write_text(app.config["SECRET_KEY"], encoding="utf-8")

    app.extensions["bioguard_rate_buckets"] = {}

    @app.context_processor
    def inject_globals():
        return {
            "csrf_token": csrf_token,
            "artifact_status": artifact_status(),
            "access_permissions": ACCESS_PERMISSIONS,
            "roles": ROLES,
            "setup_required": bool(app.config.get("SETUP_REQUIRED")),
            "admin_entry_url": url_for("admin_setup") if app.config.get("SETUP_REQUIRED") else url_for("admin_login", next="/admin/enroll"),
        }

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error):
        return jsonify({"ok": False, "error": "Upload exceeds the local size limit"}), 413

    @app.get("/")
    def index():
        return render_template("index.html", title="BioGuard Secret Door")

    @app.get("/game")
    def game():
        return render_template("game.html", title="Secret Door")

    @app.get("/fingerprint")
    def fingerprint_lab():
        return render_template("fingerprint.html", title="Fingerprint Lab", result=None)

    @app.route("/admin/setup", methods=["GET", "POST"])
    def admin_setup():
        if not app.config.get("SETUP_REQUIRED"):
            return redirect(url_for("admin_login"))
        if request.method == "GET":
            return render_template("admin_setup.html", title="First-run setup", error=None)
        if not csrf_is_valid():
            return render_template("admin_setup.html", title="First-run setup", error="CSRF validation failed."), 400
        allowed, retry = _rate("admin-setup", 5, 600)
        if not allowed:
            return render_template(
                "admin_setup.html", title="First-run setup", error=f"Too many attempts. Retry in about {retry} seconds."
            ), 429
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if len(password) < 6:
            return render_template(
                "admin_setup.html", title="First-run setup", error="Use at least 6 characters for the Admin PIN/password."
            ), 400
        if password != confirm:
            return render_template(
                "admin_setup.html", title="First-run setup", error="The two password fields do not match."
            ), 400
        password_hash = generate_password_hash(password)
        _write_admin_config(password_hash)
        app.config["ADMIN_PASSWORD_HASH"] = password_hash
        app.config["SETUP_REQUIRED"] = False
        _audit("admin_setup", "granted", "local_admin_secret_created")
        flash("Admin access is ready. Sign in to open the control panel.", "success")
        return redirect(url_for("admin_login"))

    @app.post("/fingerprint")
    @require_csrf
    def fingerprint_upload():
        wants_json = (
            request.form.get("response_format") == "json"
            or "application/json" in request.headers.get("Accept", "")
        )

        def respond(payload: dict, status: int = 200):
            if wants_json:
                return jsonify(payload), status
            return render_template("fingerprint.html", title="Fingerprint Lab", result=payload), status

        def error_payload(code: str, detail: str) -> dict:
            return {
                "ok": False,
                "match": False,
                "decision": "NO MATCH",
                "error_code": code,
                "reason": detail,
                "detail": detail,
            }

        def upload_error(exc: Exception) -> tuple[str, str, int]:
            if isinstance(exc, fingerprint_adapter.FingerprintInferenceError):
                status = 503 if exc.code in {"MODEL_LOAD_FAILED", "MODEL_INCOMPATIBLE"} else 400
                return exc.code, exc.detail, status
            message = str(exc).lower()
            if "empty" in message:
                return "EMPTY_UPLOAD", "The selected file is empty. Choose a fingerprint image and try again.", 400
            if "unsupported" in message or "file type" in message or "extension" in message:
                return "INVALID_EXTENSION", "Use a PNG, JPG, JPEG, or BMP fingerprint image.", 400
            if "image" in message or "decode" in message:
                return "INVALID_UPLOAD", "The selected file is not a supported fingerprint image.", 400
            return "FINGERPRINT_PROCESSING_FAILED", "Fingerprint verification failed safely. Try another image.", 500

        upload = request.files.get("fingerprint_image")
        if not upload or not upload.filename:
            return respond(error_payload("NO_FILE", "Choose a fingerprint image first."), 400)
        if not _rate("fingerprint", 12, 60)[0]:
            return respond(error_payload("RATE_LIMITED", "Too many scans; try again shortly."), 429)
        runtime = _runtime_dir()
        temp_dir = Path(tempfile.mkdtemp(prefix="finger-", dir=runtime))
        try:
            image_path = _write_upload(upload, temp_dir, "fingerprint", ALLOWED_FINGERPRINT_EXTENSIONS)
            result = fingerprint_adapter.verify_upload_with_visuals(
                image_path,
                model_path=_model_path("fingerprint_model.joblib"),
            )
            _audit("fingerprint", "granted" if result.get("match") else "denied", "real_model_decision")
            return respond(result, 200)
        except Exception as exc:
            _audit("fingerprint", "denied", "fingerprint_processing_error")
            code, detail, status = upload_error(exc)
            return respond(error_payload(code, detail), status)
        finally:
            _safe_remove(temp_dir, runtime)

    @app.get("/admin/login")
    def admin_login():
        if app.config.get("SETUP_REQUIRED"):
            return redirect(url_for("admin_setup"))
        if not admin_password_configured():
            return render_template(
                "admin_login.html",
                title="Admin Login",
                error=app.config.get("SECURITY_CONFIGURATION_MESSAGE")
                or "Set BIOGUARD_ADMIN_PASSWORD and BIOGUARD_SECRET_KEY before using the admin console.",
            ), 503
        return render_template("admin_login.html", title="Admin Login", error=None)

    @app.post("/admin/login")
    @require_csrf
    def admin_login_submit():
        if app.config.get("SETUP_REQUIRED"):
            return redirect(url_for("admin_setup"))
        if not admin_password_configured():
            return render_template(
                "admin_login.html",
                title="Admin Login",
                error="Administrator authentication is not configured.",
            ), 503
        allowed, retry = _rate("admin-login", 5, 600)
        if not allowed:
            return render_template(
                "admin_login.html",
                title="Admin Login",
                error=f"Too many attempts. Retry in about {retry} seconds.",
            ), 429
        password = request.form.get("password", "")
        if not check_password(password):
            _audit("admin_login", "denied", "invalid_password")
            return render_template("admin_login.html", title="Admin Login", error="Access denied."), 401
        log_in_admin()
        _audit("admin_login", "granted", "password_authenticated")
        next_path = request.args.get("next") or request.form.get("next") or url_for("admin_users")
        if not next_path.startswith("/") or next_path.startswith("//"):
            next_path = url_for("admin_users")
        return redirect(next_path)

    @app.get("/admin/logout")
    def admin_logout():
        log_out_admin()
        return redirect(url_for("index"))

    @app.get("/admin/users")
    @admin_required
    def admin_users():
        storage = _orphaned_private_storage()
        return render_template(
            "admin_users.html",
            title="Enrolled Users",
            users=_users_for_view(),
            orphaned_storage=storage["orphaned"],
            storage_error=storage["error"],
        )

    @app.post("/admin/storage-recovery/<identifier>")
    @admin_required
    @require_csrf
    def admin_storage_recovery(identifier: str):
        wants_json = bool(request.is_json or "application/json" in (request.headers.get("Accept") or ""))
        data = request.get_json(silent=True) or request.form
        confirmation = str(data.get("confirm", "")).lower()
        if confirmation not in {"yes", "true", "on", "approved"}:
            payload = {
                "ok": False,
                "error_code": "EXPLICIT_CONFIRMATION_REQUIRED",
                "error": "Explicit administrator confirmation is required before moving orphaned storage.",
            }
            if wants_json:
                return jsonify(payload), 400
            flash(payload["error"], "error")
            return redirect(url_for("admin_users"))
        try:
            result = _quarantine_orphan_storage(identifier)
            _audit("storage_recovery", "granted", "orphan_quarantined")
            message = "Orphaned private storage was moved to protected recovery storage. No biometric data was deleted."
            if wants_json:
                return jsonify({"ok": True, "message": message, **result})
            flash(message, "success")
            return redirect(url_for("admin_users"))
        except StorageConsistencyError as exc:
            _audit("storage_recovery", "denied", exc.code)
            payload = {"ok": False, "error_code": exc.code, "error": exc.detail}
            if wants_json:
                return jsonify(payload), exc.status
            flash(exc.detail, "error")
            return redirect(url_for("admin_users"))

    @app.route("/admin/enroll", methods=["GET", "POST"])
    @admin_required
    def admin_enroll():
        re_enroll = request.args.get("re_enroll", "") or request.form.get("replace_identifier", "")
        current = _users_for_view()
        previous = next((item for item in current if item["identifier"] == re_enroll), None)
        if request.method == "GET":
            return render_template(
                "admin_enroll.html",
                title="Add User",
                error=None,
                success=None,
                re_enroll=previous,
                form_state={},
            )
        if not csrf_is_valid():
            return render_template(
                "admin_enroll.html",
                title="Add User",
                error="CSRF validation failed.",
                success=None,
                re_enroll=previous,
                form_state=_enrollment_form_state(),
            ), 400
        try:
            payload = _enrollment_payload()
            _enroll_from_request(payload, replace_identifier=re_enroll or None)
            _audit("enrollment", "granted", "template_gallery_updated")
            flash(f"{payload['display_name'].upper()} ENROLLED SUCCESSFULLY. Available immediately in the local gallery.", "success")
            return redirect(url_for("admin_users"))
        except EnrollmentValidationError as exc:
            _audit("enrollment", "denied", "slot_validation_error")
            first_error = next(iter(exc.errors.values()), {}).get("message", str(exc))
            return render_template(
                "admin_enroll.html",
                title="Add User",
                error=first_error,
                success=None,
                re_enroll=previous,
                form_state=_enrollment_form_state(),
                enrollment_errors=exc.errors,
            ), 400
        except StorageConsistencyError as exc:
            _audit("enrollment", "denied", exc.code)
            return render_template(
                "admin_enroll.html",
                title="Add User",
                error=exc.detail,
                success=None,
                re_enroll=previous,
                form_state=_enrollment_form_state(),
            ), exc.status
        except ValueError as exc:
            _audit("enrollment", "denied", "validation_error")
            return render_template(
                "admin_enroll.html",
                title="Add User",
                error=str(exc),
                success=None,
                re_enroll=previous,
                form_state=_enrollment_form_state(),
            ), 400
        except Exception:
            _audit("enrollment", "denied", "internal_error")
            return render_template(
                "admin_enroll.html",
                title="Add User",
                error="Enrollment failed closed. Check the local server log for setup details.",
                success=None,
                re_enroll=previous,
                form_state=_enrollment_form_state(),
            ), 500

    @app.get("/admin/analytics")
    @admin_required
    def admin_analytics():
        report_names = (
            "public_mode_comparison.csv",
            "open_set_comparison.csv",
            "commands_comparison.csv",
            "fingerprint_comparison.csv",
        )
        reports = {name: _read_report(name) for name in report_names}
        manifest = _read_json(_model_path("model_manifest.json"))
        return render_template("results.html", title="Research & Audit Reports", reports=reports, manifest=manifest)

    @app.post("/api/face-candidate")
    @require_csrf
    def api_face_candidate():
        """Run shared local-gallery face identification before voice."""
        allowed, retry = _rate("verify-face", app.config["MAX_VERIFY_ATTEMPTS"], app.config["VERIFY_WINDOW_SECONDS"])
        if not allowed:
            return jsonify(_denied_payload("retry_limit", retry_after=retry)), 429
        session.pop("face_candidate", None)
        session.pop("face_candidate_at", None)
        session.pop("face_score", None)
        session.pop("voice_score", None)
        session.pop("last_candidate", None)
        session.pop("verified_identity", None)
        session.pop("verified_at", None)
        session.pop("verified_ticket", None)
        session.pop("voice_score", None)
        if not _model_exists("public_models.joblib"):
            _audit("face_candidate", "denied", "public_verifier_missing")
            return jsonify(_denied_payload("public_verifier_missing")), 503
        face_upload = request.files.get("face_image") or request.files.get("face")
        if not face_upload:
            _audit("face_candidate", "denied", "missing_face")
            return jsonify(_denied_payload("missing_face")), 400
        temp_dir = Path(tempfile.mkdtemp(prefix="face-candidate-", dir=_runtime_dir()))
        package = None
        gallery = None
        try:
            face_path = _write_upload(face_upload, temp_dir, "live-face", ALLOWED_FACE_EXTENSIONS)
            _validate_face_file(face_path)
            gallery = deployment.load_gallery(_gallery_path())
            if not gallery.get("templates"):
                _audit("face_candidate", "denied", "no_users_enrolled")
                return jsonify(_denied_payload("no_users_enrolled", stage="face"))
            face_vector = matching.extract_face_live(face_path)
            calibration = matching.calibration_for(
                gallery,
                "image",
                PROJECT_ROOT,
                GAME_ROOT / "local_matching_calibration.json",
            )
            local_decision = matching.rank_gallery(face_vector, gallery, "image", calibration, PROJECT_ROOT)
            decision = {
                **local_decision,
                "scores": {"face": float(local_decision.get("score", 0.0))},
                "local": {"face": local_decision},
            }
            if not local_decision.get("accepted") or local_decision.get("identity") not in gallery.get("templates", {}):
                code = str(local_decision.get("code") or "FACE_UNKNOWN")
                _audit("face_candidate", "denied", _reason_code(code))
                payload = _denied_payload(
                    code,
                    stage="face",
                    detail=_local_failure_detail(code, "The face was not accepted by the local enrollment gallery."),
                )
                diagnostics = _face_diagnostics(decision, _safe_public_package(), gallery, state="FACE_REJECTED")
                if diagnostics:
                    payload["diagnostics"] = diagnostics
                return jsonify(payload)
            identity = str(local_decision["identity"])
            session["face_candidate"] = identity
            session["face_candidate_at"] = time.time()
            session["face_score"] = float(local_decision.get("score", 0.0))
            session["last_candidate"] = identity
            _audit("face_candidate", "granted", "local_face_candidate_ready_for_voice")
            payload = {
                "ok": True,
                "accepted": True,
                "phase": "voice_required",
                "reason": "Local face candidate recognized. Voice required.",
                "message": "Face verified. Record voice for the same candidate.",
            }
            diagnostics = _face_diagnostics(decision, _safe_public_package(), gallery, identity=identity, state="FACE_ACCEPTED")
            if diagnostics:
                payload["diagnostics"] = diagnostics
            return jsonify(payload)
        except ValueError as exc:
            _audit("face_candidate", "denied", "face_validation_error")
            payload = _denied_payload(
                "FACE_UNKNOWN",
                stage="face",
                detail=str(exc)[:240] or "The face capture could not be validated. Try again with one well-lit face.",
            )
            if isinstance(gallery, dict):
                diagnostics = _face_diagnostics({"accepted": False}, _safe_public_package(), gallery, state="FACE_REJECTED")
                if diagnostics:
                    payload["diagnostics"] = diagnostics
            return jsonify(payload), 200
        except Exception:
            _audit("face_candidate", "denied", "inference_error")
            return jsonify(_denied_payload("processing_error")), 200
        finally:
            _safe_remove(temp_dir, _runtime_dir())

    @app.post("/api/verify")
    @require_csrf
    def api_verify():
        allowed, retry = _rate("verify-voice", app.config["MAX_VERIFY_ATTEMPTS"], app.config["VERIFY_WINDOW_SECONDS"])
        if not allowed:
            return jsonify(_denied_payload("retry_limit", retry_after=retry)), 429
        session.pop("verified_identity", None)
        session.pop("verified_at", None)
        session.pop("verified_ticket", None)
        if not _model_exists("public_models.joblib"):
            _audit("door", "denied", "public_verifier_missing")
            return jsonify(_denied_payload("public_verifier_missing")), 503
        voice_upload = request.files.get("voice_recording") or request.files.get("voice")
        face_upload = request.files.get("face_image") or request.files.get("face")
        pending_candidate = session.get("face_candidate")
        pending_at = float(session.get("face_candidate_at", 0) or 0)
        pending_valid = bool(pending_candidate and time.time() - pending_at <= 120)
        if not voice_upload or (not pending_valid and not face_upload):
            _audit("door", "denied", "missing_modality")
            return jsonify(_denied_payload("missing_modality")), 400
        # A fresh voice-only request is bound to the server-side candidate
        # created by /api/face-candidate.  The face+voice fallback preserves
        # the original non-browser API contract for direct integrations.
        use_pending_candidate = pending_valid and not face_upload
        temp_dir = Path(tempfile.mkdtemp(prefix="attempt-", dir=_runtime_dir()))
        try:
            voice_raw = _write_upload(voice_upload, temp_dir, "live-audio", ALLOWED_AUDIO_EXTENSIONS)
            voice_path = _convert_audio(voice_raw, temp_dir / "live-voice.wav", min_seconds=0.6)
            gallery = deployment.load_gallery(_gallery_path())
            package = _safe_public_package()
            door_permissions = {
                uid: bool(gallery.get("permissions", {}).get(uid, False) and gallery.get("meta", {}).get(uid, {}).get("active", True))
                for uid in gallery.get("templates", {})
            }
            if use_pending_candidate:
                identity = str(pending_candidate)
                session["last_candidate"] = identity
                if identity not in gallery.get("templates", {}):
                    decision = {
                        "accepted": False,
                        "identity": "Unknown",
                        "candidate": identity,
                        "failure_code": "FACE_UNKNOWN",
                        "reason": "Face candidate is no longer available in the local gallery",
                    }
                else:
                    voice_vector = _extract_voice_checked(voice_path)
                    voice_calibration = matching.calibration_for(
                        gallery,
                        "voice",
                        PROJECT_ROOT,
                        GAME_ROOT / "local_matching_calibration.json",
                    )
                    voice_check = matching.verify_claim(
                        voice_vector,
                        gallery,
                        identity,
                        "voice",
                        voice_calibration,
                        PROJECT_ROOT,
                    )
                    authorized = bool(door_permissions.get(identity, False))
                    accepted = bool(voice_check.get("accepted") and authorized)
                    failure_code = None
                    if not voice_check.get("accepted"):
                        failure_code = voice_check.get("code", "VOICE_LOW_CONFIDENCE")
                    elif not authorized:
                        failure_code = "PERMISSION_DENIED"
                    decision = {
                        "accepted": accepted,
                        "identity": identity if accepted else "Unknown",
                        "candidate": identity,
                        "reason": "Both face and voice accepted" if accepted else voice_check.get("reason", "Voice or permission rejected"),
                        "failure_code": failure_code,
                        "scores": {
                            "face": float(session.get("face_score", 0.0)),
                            "voice": float(voice_check.get("claim_score", voice_check.get("score", 0.0))),
                        },
                        "local": {"voice": voice_check},
                    }
                    session["voice_score"] = float(voice_check.get("claim_score", voice_check.get("score", 0.0)))
            else:
                face_path = _write_upload(face_upload, temp_dir, "live-face", ALLOWED_FACE_EXTENSIONS)
                _validate_face_file(face_path)
                face_vector = matching.extract_face_live(face_path)
                voice_vector = _extract_voice_checked(voice_path)
                local_decision = matching.hierarchical_verify(
                    face_vector,
                    voice_vector,
                    gallery,
                    PROJECT_ROOT,
                    GAME_ROOT / "local_matching_calibration.json",
                    door_permissions,
                )
                decision = {
                    "accepted": bool(local_decision.get("accepted")),
                    "identity": local_decision.get("identity", "Unknown"),
                    "candidate": local_decision.get("candidate"),
                    "reason": local_decision.get("reason", "Local verification rejected"),
                    "failure_code": None if local_decision.get("accepted") else local_decision.get("code", "TECHNICAL_ERROR"),
                    "scores": {
                        "face": float((local_decision.get("face") or {}).get("score", 0.0)),
                        "voice": float((local_decision.get("voice") or {}).get("claim_score", (local_decision.get("voice") or {}).get("score", 0.0))),
                    },
                    "local": {
                        "face": local_decision.get("face") or {},
                        "voice": local_decision.get("voice") or {},
                    },
                    "stage": local_decision.get("stage"),
                }
                session["last_candidate"] = local_decision.get("candidate")
                session["voice_score"] = decision["scores"]["voice"]
            session.pop("face_candidate", None)
            session.pop("face_candidate_at", None)
            session.pop("face_score", None)
            if not decision.get("accepted"):
                _audit("door", "denied", _reason_code(decision.get("reason", "rejected")))
                session.pop("verified_identity", None)
                session.pop("verified_at", None)
                session.pop("verified_ticket", None)
                failure_code = _verification_failure_code(decision, package, gallery, pending_candidate)
                payload = _denied_payload(failure_code)
                diagnostics = _decision_diagnostics(
                    decision,
                    package,
                    gallery,
                    identity=pending_candidate if use_pending_candidate else None,
                    state=failure_code.upper(),
                )
                if diagnostics:
                    payload["diagnostics"] = diagnostics
                return jsonify(payload)
            identity = str(decision.get("identity"))
            meta = gallery.get("meta", {}).get(identity, {})
            display_name = meta.get("display_name") or identity
            # A short-lived server-side ticket is required for command mode;
            # the command endpoint independently verifies the speaker again.
            session["verified_identity"] = identity
            session["verified_at"] = time.time()
            session["verified_ticket"] = secrets.token_urlsafe(24)
            _audit("door", "granted", "face_voice_permissions_passed")
            payload = {
                    "ok": True,
                    "accepted": True,
                    "identity": display_name,
                    "role": meta.get("role", "operator"),
                    "reason": "Access Granted",
                    "message": f"Welcome, {display_name}.",
                    "door": "open",
                }
            diagnostics = _decision_diagnostics(
                decision,
                package,
                gallery,
                identity=identity,
                state="ACCESS_GRANTED",
            )
            if diagnostics:
                payload["diagnostics"] = diagnostics
            return jsonify(payload)
        except AudioProcessingError as exc:
            _audit("door", "denied", exc.code)
            return jsonify(
                _denied_payload(
                    exc.code,
                    stage="voice",
                    detail=exc.detail,
                )
            ), 200
        except ValueError as exc:
            _audit("door", "denied", "voice_validation_error" if use_pending_candidate else "input_validation_error")
            if use_pending_candidate:
                return jsonify(
                    _denied_payload(
                        "voice_gate",
                        stage="voice",
                        detail=str(exc)[:240] or "The voice recording could not be validated. Speak clearly and retry.",
                    )
                ), 200
            return jsonify(
                _denied_payload(
                    "biometric_gate",
                    stage="technical",
                    detail="The submitted face and voice captures could not be validated. Retry both captures.",
                )
            ), 200
        except Exception as exc:
            session.pop("face_candidate", None)
            session.pop("face_candidate_at", None)
            session.pop("face_score", None)
            session.pop("verified_identity", None)
            session.pop("verified_at", None)
            session.pop("verified_ticket", None)
            app.logger.exception("Voice verification server error")
            _audit("door", "denied", f"SERVER_ERROR:{type(exc).__name__}")
            return jsonify(_denied_payload("SERVER_ERROR", stage="technical")), 500
        finally:
            _safe_remove(temp_dir, _runtime_dir())

    @app.post("/api/command")
    @require_csrf
    def api_command():
        allowed, retry = _rate("command", 10, app.config["COMMAND_WINDOW_SECONDS"])
        if not allowed:
            return jsonify(_denied_payload("retry_limit", retry_after=retry)), 429
        # Voice-only bonus must NOT require a preceding face/door session.
        # The command is authorized by fresh speaker biometrics, the isolated
        # word model, and local permissions below, all evaluated per request.
        if not _model_exists("commands_model.joblib"):
            _audit("command", "denied", "command_model_missing")
            return jsonify(
                {
                    "ok": False,
                    "accepted": False,
                    "reason": "Command model unavailable.",
                    "failure_stage": "voice_command",
                    "detail": "The trained GO/STOP command artifact is unavailable.",
                }
            ), 503
        command_upload = request.files.get("command_audio") or request.files.get("command")
        speaker_upload = request.files.get("speaker_audio") or request.files.get("speaker")
        if not command_upload or not speaker_upload:
            return jsonify(
                {
                    "ok": False,
                    "accepted": False,
                    "reason": "Separate speaker challenge and command recording required.",
                    "failure_stage": "voice_command",
                    "detail": "Record the speaker check first, then record one isolated GO or STOP word.",
                }
            ), 400
        temp_dir = Path(tempfile.mkdtemp(prefix="command-", dir=_runtime_dir()))
        try:
            command_raw = _write_upload(command_upload, temp_dir, "command", ALLOWED_AUDIO_EXTENSIONS)
            command_wav = _convert_audio(command_raw, temp_dir / "command.wav", min_seconds=0.13, command=True)
            speaker_raw = _write_upload(speaker_upload, temp_dir, "speaker", ALLOWED_AUDIO_EXTENSIONS)
            speaker_wav = _convert_audio(speaker_raw, temp_dir / "speaker.wav", min_seconds=0.6)
            command_package = joblib.load(_model_path("commands_model.joblib"))
            gallery = deployment.load_gallery(_gallery_path())
            policy = _command_policy(gallery)
            speaker_vector = _extract_voice_checked(speaker_wav)
            speaker_calibration = matching.calibration_for(
                gallery,
                "voice",
                PROJECT_ROOT,
                GAME_ROOT / "local_matching_calibration.json",
            )
            speaker_check = matching.rank_gallery(
                speaker_vector,
                gallery,
                "voice",
                speaker_calibration,
                PROJECT_ROOT,
            )
            command_check = None
            if not speaker_check.get("accepted"):
                decision = {
                    "accepted": False,
                    "identity": "Unknown",
                    "reason": speaker_check.get("reason", "Speaker unverified"),
                    "failure_code": "speaker_command_gate",
                }
            else:
                candidate = str(speaker_check["identity"])
                command_check = commands_research.predict_command(
                    command_wav,
                    path=str(_model_path("commands_model.joblib")),
                )
                if not command_check.get("recognized"):
                    decision = {
                        "accepted": False,
                        "identity": candidate,
                        "failure_code": "COMMAND_UNCERTAIN",
                        "reason": (
                            "Command uncertain: model predicted "
                            f"{command_check.get('predicted_word', 'an unknown word').upper()} "
                            f"with confidence {float(command_check.get('confidence', command_check.get('score', 0.0))):.3f}, "
                            f"below the {float(command_check.get('threshold', 0.0)):.3f} threshold."
                        ),
                    }
                else:
                    action = command_check.get("action")
                    allowed_command = action in policy.get(candidate, set())
                    decision = {
                        "accepted": bool(allowed_command),
                        "identity": candidate,
                        "command": command_check.get("word"),
                        "action": action,
                        "failure_code": None if allowed_command else ("COMMAND_NOT_ACTION" if not action else "PERMISSION_DENIED"),
                        "reason": "Verified speaker + allowed word" if allowed_command else (
                            f"Recognized {str(command_check.get('word') or command_check.get('predicted_word') or 'word').upper()}, "
                            "but it is not an authorized door action."
                            if not action else "Missing action permission"
                        ),
                    }
            server_permission = commands_bonus.authorized_command(
                decision.get("identity"),
                decision.get("action"),
                bool(decision.get("accepted")),
                policy,
            )
            if (
                not decision.get("accepted")
                or not server_permission
            ):
                failure_code = decision.get("failure_code")
                if not failure_code:
                    failure_code = "PERMISSION_DENIED" if not server_permission else "speaker_command_gate"
                _audit("command", "denied", str(failure_code))
                payload = _denied_payload(
                    failure_code,
                    stage="voice_command",
                    detail=str(decision.get("reason") or "Speaker, command confidence, or permission was rejected."),
                )
                diagnostics = _command_diagnostics(command_check, state=str(failure_code))
                if diagnostics:
                    payload["diagnostics"] = diagnostics
                return jsonify(payload)
            _audit("command", "granted", "speaker_and_permission_passed")
            verified_speaker = str(decision["identity"])
            meta = gallery.get("meta", {}).get(verified_speaker, {})
            payload = {
                "ok": True,
                "accepted": True,
                "command": decision.get("command"),
                "action": decision.get("action"),
                "identity": meta.get("display_name") or verified_speaker,
                "reason": "Command authorized",
            }
            diagnostics = _command_diagnostics(command_check, state="COMMAND_ACCEPTED")
            if diagnostics:
                payload["diagnostics"] = diagnostics
            return jsonify(payload)
        except AudioProcessingError as exc:
            _audit("command", "denied", exc.code)
            return jsonify(_denied_payload(exc.code, stage="voice_command", detail=exc.detail)), 200
        except Exception as exc:
            app.logger.exception("Voice command server error")
            _audit("command", "denied", "command_processing_error")
            return jsonify(_denied_payload("SERVER_ERROR", stage="voice_command")), 500
        finally:
            _safe_remove(temp_dir, _runtime_dir())

    @app.get("/api/status")
    def api_status():
        return jsonify(artifact_status())

    @app.get("/api/admin/diagnostics")
    @admin_required
    def api_admin_diagnostics():
        """Return the current server-side auth state to an authenticated admin."""
        try:
            package = _safe_public_package()
            gallery = deployment.load_gallery(_gallery_path())
            identity = session.get("verified_identity") or session.get("face_candidate") or session.get("last_candidate")
            local = {}
            if identity and identity in gallery.get("templates", {}):
                face_calibration = matching.calibration_for(
                    gallery, "image", PROJECT_ROOT, GAME_ROOT / "local_matching_calibration.json"
                )
                voice_calibration = matching.calibration_for(
                    gallery, "voice", PROJECT_ROOT, GAME_ROOT / "local_matching_calibration.json"
                )
                local = {
                    "face": {
                        "score": session.get("face_score"),
                        "threshold": face_calibration.get("threshold"),
                        "distance": None,
                        "margin": None,
                        "calibration": face_calibration,
                    },
                    "voice": {
                        "score": session.get("voice_score"),
                        "threshold": voice_calibration.get("threshold"),
                        "distance": None,
                        "margin": None,
                        "calibration": voice_calibration,
                    },
                }
            decision = {
                "candidate": identity,
                "scores": {
                    "face": session.get("face_score"),
                    "voice": session.get("voice_score"),
                },
                "local": local,
            }
            if session.get("verified_identity"):
                state = "ACCESS_GRANTED"
            elif session.get("face_candidate"):
                state = "FACE_ACCEPTED"
            elif session.get("voice_score") is not None:
                state = "VOICE_REJECTED"
            else:
                state = "IDLE"
            return jsonify(_admin_diagnostics(package, gallery, decision=decision, identity=identity, state=state))
        except Exception:
            return jsonify(
                {
                    "enrolled_users": 0,
                    "face_candidate": "Unavailable",
                    "face_score": None,
                    "face_threshold": None,
                    "voice_score": None,
                    "voice_threshold": None,
                    "user_role": "Unavailable",
                    "door_permission": False,
                    "current_auth_state": "TECHNICAL_ERROR",
                }
            ), 200

    @app.post("/api/admin/enroll")
    @admin_required
    @require_csrf
    def api_admin_enroll():
        try:
            payload = _enrollment_payload()
            _enroll_from_request(payload, replace_identifier=request.form.get("replace_identifier") or None)
            _audit("enrollment", "granted", "api_gallery_updated")
            return jsonify(
                {
                    "ok": True,
                    "message": "User enrolled and available immediately.",
                    "redirect": url_for("admin_users"),
                }
            )
        except EnrollmentValidationError as exc:
            _audit("enrollment", "denied", "slot_validation_error")
            return jsonify(
                {
                    "ok": False,
                    "error": "Fix the highlighted enrollment fields or samples, then try again.",
                    "errors": exc.errors,
                }
            ), 400
        except StorageConsistencyError as exc:
            _audit("enrollment", "denied", exc.code)
            return jsonify({"ok": False, "error_code": exc.code, "error": exc.detail}), exc.status
        except ValueError as exc:
            _audit("enrollment", "denied", "validation_error")
            return jsonify({"ok": False, "error": str(exc)}), 400
        except Exception:
            _audit("enrollment", "denied", "internal_error")
            return jsonify({"ok": False, "error": "Enrollment failed closed."}), 500

    @app.route("/api/admin/users/<identifier>", methods=["POST", "PATCH", "DELETE"])
    @admin_required
    @require_csrf
    def api_admin_user(identifier: str):
        try:
            deployment.validate_new_username(identifier)
        except ValueError:
            return jsonify({"ok": False, "error": "Invalid user identifier"}), 400
        wants_json = bool(request.is_json or "application/json" in (request.headers.get("Accept") or ""))
        if request.method == "DELETE" or request.form.get("action") == "delete" or (
            (request.get_json(silent=True) or {}).get("action") == "delete"
        ):
            try:
                result = _delete_user(identifier)
                if result["cleanup"] != "complete":
                    _audit("user_delete", "denied", "private_cleanup_pending")
                    payload = {
                        "ok": False,
                        "error_code": "PRIVATE_STORAGE_CLEANUP_PENDING",
                        "error": result["message"],
                        "storage_cleanup": result["cleanup"],
                    }
                    if wants_json:
                        return jsonify(payload), 500
                    flash(payload["error"], "error")
                    return redirect(url_for("admin_users"))
                audit_reason = "gallery_and_private_data_removed" if result["cleanup"] == "complete" else "gallery_removed_private_cleanup_pending"
                _audit("user_delete", "granted", audit_reason)
                message = result["message"]
                if wants_json:
                    return jsonify({"ok": True, "message": message, "storage_cleanup": result["cleanup"]})
                flash(message, "success")
                return redirect(url_for("admin_users"))
            except KeyError:
                payload = {"ok": False, "error_code": "USER_NOT_FOUND", "error": "User not found"}
                if wants_json:
                    return jsonify(payload), 404
                flash(payload["error"], "error")
                return redirect(url_for("admin_users"))
            except StorageConsistencyError as exc:
                _audit("user_delete", "denied", exc.code)
                payload = {"ok": False, "error_code": exc.code, "error": exc.detail}
                if wants_json:
                    return jsonify(payload), exc.status
                flash(exc.detail, "error")
                return redirect(url_for("admin_users"))
            except OSError:
                _audit("user_delete", "denied", "STORAGE_DELETE_FAILED")
                payload = {
                    "ok": False,
                    "error_code": "STORAGE_DELETE_FAILED",
                    "error": "User deletion failed safely. The registry and private storage were not reported as removed.",
                }
                if wants_json:
                    return jsonify(payload), 500
                flash(payload["error"], "error")
                return redirect(url_for("admin_users"))
        data = request.get_json(silent=True) or request.form
        try:
            updated = _update_user(identifier, data)
            _audit("user_update", "granted", "permissions_updated")
            if wants_json:
                return jsonify({"ok": True, "user": updated})
            return redirect(url_for("admin_users"))
        except KeyError:
            payload = {"ok": False, "error_code": "USER_NOT_FOUND", "error": "User not found"}
            if wants_json:
                return jsonify(payload), 404
            flash(payload["error"], "error")
            return redirect(url_for("admin_users"))
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    return app


def _config(key: str):
    from flask import current_app

    return current_app.config[key]


def _model_path(name: str) -> Path:
    return _path(_config("MODEL_DIR") / name)


def _gallery_path() -> Path:
    return _path(_config("GALLERY_FILE"))


def _users_root() -> Path:
    return _path(_config("PRIVATE_ROOT") / "users")


def _user_storage_path(identifier: str) -> Path:
    """Resolve one validated private user directory beneath the users root."""
    deployment.validate_new_username(identifier)
    root = _users_root()
    target = root / identifier
    if target.parent != root:
        raise StorageConsistencyError("INVALID_STORAGE_PATH", "The private enrollment path is invalid.", status=400)
    return target


def _orphaned_private_storage(gallery: dict | None = None) -> dict:
    """List private user directories without a corresponding gallery record.

    This is intentionally read-only. Orphans are never removed as a side effect
    of rendering the admin users page.
    """
    try:
        current_gallery = gallery if gallery is not None else deployment.load_gallery(_gallery_path())
        registered = set(current_gallery.get("templates", {}))
        root = _users_root()
        root.mkdir(parents=True, exist_ok=True)
        entries = list(root.iterdir())
    except OSError:
        return {"orphaned": [], "error": "Private enrollment storage could not be listed safely."}
    except (KeyError, TypeError, ValueError):
        return {"orphaned": [], "error": "The local enrollment registry could not be read safely."}

    orphaned = []
    for entry in entries:
        if entry.name.startswith(".") or not entry.is_dir() or entry.name in registered:
            continue
        readable = True
        try:
            next(entry.iterdir(), None)
        except OSError:
            readable = False
        orphaned.append(
            {
                "identifier": entry.name,
                "readable": readable,
                "reason": "Private directory has no matching registry record.",
            }
        )
    return {"orphaned": sorted(orphaned, key=lambda item: item["identifier"]), "error": None}


def _quarantine_orphan_storage(identifier: str) -> dict:
    """Move one explicitly approved orphan into private recovery storage.

    The move is reversible and does not delete biometric media. It is only
    reachable through an authenticated admin POST with explicit confirmation.
    """
    target = _user_storage_path(identifier)
    gallery = deployment.load_gallery(_gallery_path())
    if identifier in gallery.get("templates", {}):
        raise StorageConsistencyError("NOT_ORPHANED", "This identifier has an active registry record.")
    try:
        if not target.exists():
            raise StorageConsistencyError("ORPHAN_NOT_FOUND", "The private orphaned directory no longer exists.", status=404)
        recovery_root = _path(_config("PRIVATE_ROOT") / ".backups" / "orphaned-users")
        recovery_root.mkdir(parents=True, exist_ok=True)
        quarantine = recovery_root / f"{identifier}-{uuid.uuid4().hex}.quarantine"
        target.rename(quarantine)
    except StorageConsistencyError:
        raise
    except OSError as exc:
        raise StorageConsistencyError(
            "STORAGE_RECOVERY_BLOCKED",
            "The orphaned directory could not be moved safely. No biometric data was deleted; operating-system access is required.",
        ) from exc
    return {"identifier": identifier, "quarantine_name": quarantine.name}


def _runtime_dir() -> Path:
    path = _path(_config("PRIVATE_ROOT") / ".runtime")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rate(scope: str, limit: int, window: int) -> tuple[bool, int]:
    return rate_limit(scope, limit, window)


def _model_exists(name: str) -> bool:
    return _model_path(name).is_file()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


def _write_admin_config(password_hash: str) -> None:
    target = _path(_config("ADMIN_CONFIG_FILE"))
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps({"version": 1, "password_hash": password_hash}, indent=2), encoding="utf-8")
    os.replace(temp, target)


def _read_report(name: str) -> dict:
    path = _model_path(name)
    if not path.is_file():
        return {"headers": [], "rows": [], "missing": True}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return {"headers": reader.fieldnames or [], "rows": list(reader), "missing": False}
    except (OSError, csv.Error):
        return {"headers": [], "rows": [], "missing": True}


def artifact_status() -> dict:
    model_dir = _path(_config("MODEL_DIR"))
    try:
        gallery = deployment.load_gallery(_gallery_path())
        gallery_ok = True
    except Exception:
        # A malformed local gallery is a setup problem, never a reason to
        # expose a green status or to grant a door decision.
        gallery = {"templates": {}}
        gallery_ok = False
    return {
        "public_verifier": (model_dir / "public_models.joblib").is_file(),
        "local_matcher": (GAME_ROOT / "local_matching_calibration.json").is_file(),
        "commands": (model_dir / "commands_model.joblib").is_file(),
        "fingerprint": (model_dir / "fingerprint_model.joblib").is_file(),
        "enrolled_users": len(gallery.get("templates", {})),
        "gallery_ok": gallery_ok,
        "security_configured": admin_password_configured(),
    }


def _audit(event: str, outcome: str, reason: str) -> None:
    """Append only timestamp/outcome/reason; never raw media or feature data."""
    path = _path(_config("PRIVATE_ROOT") / "audit.log")
    line = json.dumps({"timestamp_utc": _utc_now(), "event": event, "outcome": outcome, "reason": reason})
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        # A logging failure must not turn a denial into an access grant.
        pass


def _reason_code(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")[:80] or "rejected"


_SAFE_FAILURE_DETAILS = {
    "retry_limit": ("technical", "Retry limit reached. Wait briefly, then try again."),
    "public_verifier_missing": ("technical", "The trained verifier artifact is unavailable."),
    "missing_face": ("face", "No face image reached the verifier."),
    "no_users_enrolled": ("face", "No local users are enrolled yet."),
    "face_gate": ("face", "The face was not accepted by the local enrollment gallery."),
    "voice_gate": ("voice", "The voice sample did not match the same face candidate."),
    "permission_gate": ("permission", "Identity matched, but this user is not authorized for the door."),
    "missing_modality": ("technical", "The required capture data was missing. Retry the scan."),
    "processing_error": ("technical", "Local biometric processing failed closed. Check the capture and retry."),
    "speaker_command_gate": ("voice_command", "Speaker, command confidence, or command permission was rejected."),
    "command_processing_error": ("voice_command", "Voice command processing failed closed. Retry the command."),
    "COMMAND_UNCERTAIN": ("voice_command", "The command confidence was below the trained threshold. Say GO or STOP clearly and retry."),
    "COMMAND_NOT_ACTION": ("voice_command", "The recognized word is not an authorized door command. Say GO or STOP."),
    "biometric_gate": ("biometric", "The biometric verification chain did not pass."),
    "FACE_UNKNOWN": ("face", "No enrolled local face met the calibrated threshold."),
    "FACE_LOW_CONFIDENCE": ("face", "The local face score was below the calibrated threshold."),
    "FACE_AMBIGUOUS": ("face", "More than one local face candidate was too close to distinguish safely."),
    "VOICE_MISMATCH": ("voice", "The voice did not match the same face-selected candidate."),
    "VOICE_LOW_CONFIDENCE": ("voice", "The local voice score was below the calibrated threshold."),
    "VOICE_AMBIGUOUS": ("voice", "More than one local speaker candidate was too close to distinguish safely."),
    "PERMISSION_DENIED": ("permission", "Identity matched, but this user is not authorized for the door."),
    "AUDIO_CAPTURE_FAILED": ("voice", "The browser did not produce a usable recording. Check microphone permission and retry."),
    "AUDIO_DECODE_FAILED": ("voice", "The recording could not be decoded locally. Use WAV or record again."),
    "AUDIO_UPLOAD_FAILED": ("voice", "The recording could not be uploaded. Record again and retry."),
    "AUDIO_TOO_SHORT": ("voice", "The recording was too short. Record the full voice phrase."),
    "AUDIO_SILENT": ("voice", "No usable speech was detected. Move closer to the microphone and try again."),
    "SERVER_ERROR": ("technical", "The local server failed while processing the recording."),
    "TECHNICAL_ERROR": ("technical", "Local biometric processing failed closed. Retry the capture."),
}


def _denied_payload(
    reason: str,
    retry_after: int | None = None,
    stage: str | None = None,
    detail: str | None = None,
) -> dict:
    default_stage, default_detail = _SAFE_FAILURE_DETAILS.get(
        reason,
        ("technical", "Verification failed closed. Retry the attempt."),
    )
    payload = {
        "ok": False,
        "accepted": False,
        "identity": "Unknown",
        "reason": "Access Denied",
        "message": "Access Denied. The security door remains locked.",
        "door": "locked",
        "error_code": reason,
        "failure_stage": stage or default_stage,
        "detail": detail or default_detail,
    }
    if retry_after is not None:
        payload["retry_after"] = retry_after
    return payload


def _safe_public_package() -> dict | None:
    """Load public artifacts for diagnostics only; local matching is separate."""
    try:
        return deployment.load_verifier(_model_path("public_models.joblib"))
    except Exception:
        return None


def _local_failure_detail(code: str, fallback: str) -> str:
    return _SAFE_FAILURE_DETAILS.get(code, ("technical", fallback))[1]


def _admin_diagnostics(
    package: dict | None,
    gallery: dict | None = None,
    decision: dict | None = None,
    identity: str | None = None,
    state: str = "IDLE",
) -> dict | None:
    """Build diagnostics without exposing biometric scores to public visitors."""
    from flask import has_request_context

    if not has_request_context() or not session.get("admin_authenticated"):
        return None
    gallery = gallery or {"templates": {}, "permissions": {}, "meta": {}}
    package = package or {}
    scores = (decision or {}).get("scores") or {}
    local = (decision or {}).get("local") or {}
    local_face = local.get("face") or (decision or {}).get("face") or {}
    local_voice = local.get("voice") or (decision or {}).get("voice") or {}
    public_models = package.get("models", {})
    public_face = public_models.get("Image Only", {})
    public_voice = public_models.get("Voice Only", {})
    candidate = identity or (decision or {}).get("candidate") or (decision or {}).get("identity")
    candidate = str(candidate) if candidate and candidate != "Unknown" else None
    meta = gallery.get("meta", {}).get(candidate, {}) if candidate else {}
    face_score = local_face.get("score") if local_face else scores.get("face")
    # During the face-bound voice stage, ``claim_score`` is the score for the
    # same candidate selected by face.  ``score`` is the best score anywhere
    # in the gallery and can belong to a different speaker on a mismatch.
    # Showing the global score as the candidate score made the admin panel
    # report a misleadingly strong voice result for rejected cross-user
    # attempts.
    voice_claim_score = local_voice.get("claim_score") if local_voice else None
    voice_score = voice_claim_score if voice_claim_score is not None else (
        local_voice.get("score") if local_voice else scores.get("voice")
    )
    face_threshold = local_face.get("threshold") if local_face else public_face.get("threshold")
    voice_threshold = local_voice.get("threshold") if local_voice else public_voice.get("threshold")
    voice_ranking = local_voice.get("ranking") if local_voice else None
    voice_other_scores = [
        float(item.get("score"))
        for item in (voice_ranking or [])
        if candidate and item.get("identity") != candidate and item.get("score") is not None
    ]
    voice_claim_margin = None
    if voice_claim_score is not None and voice_other_scores:
        voice_claim_margin = float(voice_claim_score) - max(voice_other_scores)
    return {
        "enrolled_users": len(gallery.get("templates", {})),
        "face_candidate": candidate or "Unknown",
        "face_score": float(face_score) if face_score is not None else None,
        "face_threshold": float(face_threshold) if face_threshold is not None else None,
        "voice_score": float(voice_score) if voice_score is not None else None,
        "voice_claim_score": float(voice_claim_score) if voice_claim_score is not None else None,
        "voice_best_score": float(local_voice["score"]) if local_voice.get("score") is not None else None,
        "voice_claim_margin": voice_claim_margin,
        "voice_threshold": float(voice_threshold) if voice_threshold is not None else None,
        "face_method": local_face.get("calibration", {}).get("method") if local_face else "public_svm",
        "voice_method": local_voice.get("calibration", {}).get("method") if local_voice else "public_model_pending",
        "face_threshold_source": local_face.get("calibration", {}).get("source") if local_face else "public_models.joblib",
        "voice_threshold_source": local_voice.get("calibration", {}).get("source") if local_voice else "public_models.joblib",
        "face_distance": float(local_face["distance"]) if local_face.get("distance") is not None else None,
        "voice_distance": float(local_voice["distance"]) if local_voice.get("distance") is not None else None,
        "face_margin": float(local_face["margin"]) if local_face.get("margin") is not None else None,
        "voice_margin": float(local_voice["margin"]) if local_voice.get("margin") is not None else None,
        "public_face_threshold": float(public_face["threshold"]) if public_face.get("threshold") is not None else None,
        "public_voice_threshold": float(public_voice["threshold"]) if public_voice.get("threshold") is not None else None,
        "user_role": meta.get("role", "Unknown") if candidate else "Unknown",
        "door_permission": bool(
            candidate
            and gallery.get("permissions", {}).get(candidate, False)
            and meta.get("active", True)
        ),
        "current_auth_state": state,
    }


def _decision_diagnostics(
    decision: dict,
    package: dict,
    gallery: dict | None = None,
    identity: str | None = None,
    state: str = "IDLE",
) -> dict | None:
    """Return real scores only to an authenticated administrator session.

    The public game response intentionally exposes a uniform denial reason and
    never leaks identity or biometric scores.  Administrators can opt into the
    panel while demonstrating the fixed artifact thresholds live.
    """
    try:
        result = _admin_diagnostics(package, gallery, decision, identity=identity, state=state)
        if result is not None:
            result["decision"] = "accepted" if decision.get("accepted") else "denied"
        return result
    except (KeyError, TypeError, ValueError):
        return None


def _command_diagnostics(command_check: dict | None, state: str) -> dict | None:
    """Expose command-classifier diagnostics only to an authenticated admin."""
    if not session.get("admin_authenticated") or not command_check:
        return None
    try:
        top = command_check.get("top_predictions") or []
        return {
            "current_auth_state": state,
            "predicted_command": command_check.get("predicted_word") or command_check.get("word"),
            "command_score": float(command_check.get("confidence", command_check.get("score"))),
            "command_threshold": float(command_check.get("threshold")),
            "command_margin": float(command_check.get("margin")),
            "command_top_predictions": [
                {"word": str(item.get("word")), "confidence": float(item.get("confidence"))}
                for item in top[:3]
            ],
            "command_feature_dim": int(command_check.get("feature_dim")),
        }
    except (TypeError, ValueError, KeyError):
        return None


def _face_diagnostics(
    decision: dict,
    package: dict,
    gallery: dict | None = None,
    identity: str | None = None,
    state: str = "FACE_REJECTED",
) -> dict | None:
    """Return the optical-stage score only to an authenticated admin."""
    try:
        result = _admin_diagnostics(package, gallery, decision, identity=identity, state=state)
        if result is not None:
            result["decision"] = "accepted" if decision.get("accepted") else "denied"
        return result
    except (KeyError, TypeError, ValueError):
        return None


def _verification_failure_code(
    decision: dict,
    package: dict,
    gallery: dict,
    candidate: str | None = None,
) -> str:
    """Classify a denial for a helpful UI message without leaking identity."""
    explicit = decision.get("failure_code")
    if explicit in {
        "face_gate", "voice_gate", "permission_gate", "processing_error",
        "FACE_UNKNOWN", "FACE_LOW_CONFIDENCE", "FACE_AMBIGUOUS",
        "VOICE_MISMATCH", "VOICE_LOW_CONFIDENCE", "VOICE_AMBIGUOUS",
        "PERMISSION_DENIED", "AUDIO_DECODE_FAILED", "AUDIO_UPLOAD_FAILED",
        "AUDIO_TOO_SHORT", "AUDIO_SILENT", "SERVER_ERROR", "TECHNICAL_ERROR",
    }:
        return explicit
    scores = decision.get("scores") or {}
    try:
        if "face" in scores and float(scores["face"]) < float(package["models"]["Image Only"]["threshold"]):
            return "face_gate"
        if "voice" in scores and float(scores["voice"]) < float(package["models"]["Voice Only"]["threshold"]):
            return "voice_gate"
    except (KeyError, TypeError, ValueError):
        return "biometric_gate"
    identity = candidate or decision.get("identity")
    if identity and identity != "Unknown" and not gallery.get("permissions", {}).get(identity, False):
        return "permission_gate"
    return "biometric_gate"


def _safe_remove(path: Path, root: Path) -> None:
    """Remove one known child of ``root``; refuse broad or unresolved paths."""
    path = _path(path)
    root = _path(root)
    if path == root or root not in path.parents:
        raise ValueError("Refusing to remove a path outside the private app directory")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _atomic_dump_gallery(gallery: dict) -> None:
    target = _gallery_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".gallery-", suffix=".joblib", dir=target.parent)
        os.close(fd)
        joblib.dump(gallery, temp_name)
        os.replace(temp_name, target)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()


def _write_upload(upload, directory: Path, stem: str, extensions: set[str]) -> Path:
    filename = secure_filename(upload.filename or "")
    suffix = Path(filename).suffix.lower()
    is_audio = not (stem.startswith("live-face") or stem.startswith("fingerprint") or stem.startswith("face"))
    if not filename or suffix not in extensions:
        if is_audio:
            raise AudioProcessingError("AUDIO_UPLOAD_FAILED", f"Unsupported audio file type. Allowed: {', '.join(sorted(extensions))}")
        raise ValueError(f"Unsupported {stem} file type. Allowed: {', '.join(sorted(extensions))}")
    mime = (upload.mimetype or "").lower()
    if stem.startswith("live-face") or stem.startswith("fingerprint") or stem.startswith("face"):
        if mime and not (mime.startswith("image/") or mime == "application/octet-stream"):
            raise ValueError("Face/fingerprint upload must be an image")
    else:
        if mime and not (
            mime.startswith("audio/")
            or mime.startswith("video/")
            or mime == "application/octet-stream"
            or mime == "application/ogg"
        ):
            raise AudioProcessingError("AUDIO_UPLOAD_FAILED", "Voice upload must be an audio recording")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{stem}-{uuid.uuid4().hex}{suffix}"
    try:
        data = upload.stream.read(int(_config("MAX_CONTENT_LENGTH")) + 1)
    except OSError as exc:
        if is_audio:
            raise AudioProcessingError("AUDIO_UPLOAD_FAILED", "The audio upload could not be read") from exc
        raise ValueError("The upload could not be read") from exc
    if len(data) > int(_config("MAX_CONTENT_LENGTH")):
        if is_audio:
            raise AudioProcessingError("AUDIO_UPLOAD_FAILED", "The audio file exceeds the local size limit")
        raise ValueError("Media file exceeds the local size limit")
    if not data:
        if is_audio:
            raise AudioProcessingError("AUDIO_UPLOAD_FAILED", "The uploaded recording was empty")
        raise ValueError("Empty media file")
    try:
        destination.write_bytes(data)
    except OSError as exc:
        if is_audio:
            raise AudioProcessingError("AUDIO_UPLOAD_FAILED", "The audio upload could not be stored temporarily") from exc
        raise ValueError("The upload could not be stored temporarily") from exc
    return destination


def _validate_face_file(path: Path) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable; face verification cannot start")
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Image could not be decoded")
    height, width = image.shape[:2]
    if min(height, width) < 96:
        raise ValueError("Face image is too small; use at least 96 pixels per side")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 5.0:
        raise ValueError("Face image has too little contrast")
    # Use the same one-time Haar crop as the public data preparation step;
    # this avoids double-detecting already normalized face crops.
    crop_single_face(image, require_face=True)
    # This is the actual research extractor; validation above only rejects bad
    # captures before the feature vector is computed.
    vector = extract_face_live(path)
    if vector.shape != (core.FACE_DIM,) or not np.isfinite(vector).all():
        raise ValueError("Face feature extraction failed")


def _convert_audio(source: Path, target: Path, min_seconds: float, command: bool = False) -> Path:
    if imageio_ffmpeg is None:
        raise AudioProcessingError("AUDIO_DECODE_FAILED", "The local audio decoder is unavailable")
    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-vn",
                str(target),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=35,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AudioProcessingError("AUDIO_DECODE_FAILED", "The recording could not be decoded locally") from exc
    try:
        info = sf.info(str(target))
    except (RuntimeError, ValueError, OSError) as exc:
        raise AudioProcessingError("AUDIO_DECODE_FAILED", "The converted recording could not be read as audio") from exc
    if float(info.duration) < float(min_seconds):
        raise AudioProcessingError("AUDIO_TOO_SHORT", f"Recording is too short; capture at least {min_seconds:.1f} seconds")
    try:
        if command:
            commands_research.extract_command_features(target)
        else:
            matching.extract_voice(target)
    except (RuntimeError, ValueError, OSError) as exc:
        raise _audio_error_from_value_error(exc) from exc
    return target


def _extract_voice_checked(path: Path) -> np.ndarray:
    """Run the shared extractor and retain a stable audio failure code."""
    try:
        return matching.extract_voice(path)
    except AudioProcessingError:
        raise
    except ValueError as exc:
        raise _audio_error_from_value_error(exc) from exc


def _enrollment_payload() -> dict:
    identifier = (request.form.get("identifier") or "").strip()
    display_name = " ".join((request.form.get("display_name") or "").split())
    role = (request.form.get("role") or "operator").strip().lower()
    # An unchecked HTML checkbox is absent; absence must mean disabled, not an
    # accidental authorization grant.  The form renders it checked explicitly.
    active = str(request.form.get("active", "")).lower() in {"1", "true", "yes", "on", "active"}
    permission_values = request.form.getlist("access_permissions")
    if not permission_values and request.is_json:
        permission_values = (request.get_json(silent=True) or {}).get("access_permissions", [])
    permissions = sorted({str(item) for item in permission_values if str(item) in ACCESS_PERMISSIONS})
    if not USERNAME_PATTERN.fullmatch(identifier):
        raise ValueError("Identifier must be 2-40 Latin letters, digits, _ or -")
    if not display_name or len(display_name) > 80 or any(ord(char) < 32 for char in display_name):
        raise ValueError("Display name is required and must be at most 80 characters")
    if role not in ROLES:
        raise ValueError("Choose a supported role")
    if not request.form.get("consent") and not request.is_json:
        raise ValueError("Consent is required before storing biometric enrollment data")
    return {
        "identifier": deployment.validate_new_username(identifier),
        "display_name": display_name,
        "role": role,
        "active": active,
        "permissions": permissions,
    }


def _enrollment_form_state() -> dict:
    """Return non-biometric form values so a normal POST error can be retried."""
    return {
        "identifier": (request.form.get("identifier") or "").strip(),
        "display_name": " ".join((request.form.get("display_name") or "").split()),
        "role": (request.form.get("role") or "operator").strip().lower(),
        "active": str(request.form.get("active", "")).lower() in {"1", "true", "yes", "on", "active"},
        "permissions": sorted({
            str(item)
            for item in request.form.getlist("access_permissions")
            if str(item) in ACCESS_PERMISSIONS
        }),
        "consent": bool(request.form.get("consent")),
    }


def _enroll_from_request(payload: dict, replace_identifier: str | None = None) -> None:
    face_uploads = [item for item in request.files.getlist("face_images") if item and item.filename]
    voice_uploads = [item for item in request.files.getlist("voice_recordings") if item and item.filename]
    if replace_identifier and replace_identifier != payload["identifier"]:
        raise ValueError("Re-enrollment identifier does not match the selected user")
    staging_root = _path(_config("PRIVATE_ROOT") / ".staging")
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="enroll-", dir=staging_root))
    face_dir = staging / "faces"
    voice_dir = staging / "voices"
    face_dir.mkdir(parents=True, exist_ok=True)
    voice_dir.mkdir(parents=True, exist_ok=True)
    face_paths: list[Path] = []
    voice_paths: list[Path] = []
    errors: dict[str, dict[str, str]] = {}
    try:
        for index in range(len(face_uploads), 4):
            errors[f"face.{index}"] = {
                "code": "FACE_MISSING",
                "message": f"Face slot {index + 1} is required.",
            }
        for index in range(len(voice_uploads), 4):
            errors[f"voice.{index}"] = {
                "code": "VOICE_MISSING",
                "message": f"Voice slot {index + 1} is required.",
            }

        for index, upload in enumerate(face_uploads):
            slot = index + 1
            try:
                path = _write_upload(upload, face_dir, f"face-{slot:02d}", ALLOWED_FACE_EXTENSIONS)
                _validate_face_file(path)
                face_paths.append(path)
            except ValueError as exc:
                errors[f"face.{index}"] = {
                    "code": "FACE_INVALID",
                    "message": f"Face slot {slot}: {str(exc)[:220] or 'image quality validation failed.'}",
                }

        for index, upload in enumerate(voice_uploads):
            slot = index + 1
            try:
                raw = _write_upload(upload, staging, f"voice-raw-{slot:02d}", ALLOWED_AUDIO_EXTENSIONS)
                wav = voice_dir / f"voice-{slot:02d}.wav"
                _convert_audio(raw, wav, min_seconds=2.5)
                raw.unlink(missing_ok=True)
                voice_paths.append(wav)
            except AudioProcessingError as exc:
                errors[f"voice.{index}"] = {
                    "code": exc.code,
                    "message": f"Voice slot {slot}: {exc.detail[:220]}",
                }
            except ValueError as exc:
                errors[f"voice.{index}"] = {
                    "code": "VOICE_INVALID",
                    "message": f"Voice slot {slot}: {str(exc)[:220] or 'audio quality validation failed.'}",
                }

        if errors:
            raise EnrollmentValidationError(errors)
        _commit_enrollment(payload, face_paths, voice_paths, staging, replace=bool(replace_identifier))
        staging = None
    finally:
        if staging is not None and staging.exists():
            _safe_remove(staging, staging_root)


def _commit_enrollment(payload: dict, face_paths: list[Path], voice_paths: list[Path], staging: Path, replace: bool) -> None:
    identifier = payload["identifier"]
    gallery = deployment.load_gallery(_gallery_path())
    if identifier in gallery.get("templates", {}) and not replace:
        raise ValueError("User identifier is already enrolled; use Re-enroll from the admin list")
    target = _user_storage_path(identifier)
    try:
        target_exists = target.exists()
    except OSError as exc:
        raise StorageConsistencyError(
            "PRIVATE_STORAGE_UNAVAILABLE",
            "The private enrollment directory could not be inspected safely. No new enrollment was saved.",
        ) from exc
    if target_exists and not replace:
        raise StorageConsistencyError(
            "ORPHANED_PRIVATE_STORAGE",
            "A private enrollment directory exists without a registry record. Use the authenticated storage recovery action before re-enrolling this identifier.",
        )
    working = copy.deepcopy(gallery)
    for key in ("templates", "permissions", "meta", "sample_vectors"):
        working.setdefault(key, {})
    if replace:
        working["templates"].pop(identifier, None)
        working["permissions"].pop(identifier, None)
        working["meta"].pop(identifier, None)
        working["sample_vectors"].pop(identifier, None)
    temporary_gallery = staging / "working-gallery.joblib"
    joblib.dump(working, temporary_gallery)
    # deployment.enroll_user is the single enrollment feature-extraction path.
    deployment.enroll_user(
        identifier,
        face_paths,
        voice_paths,
        allowed=bool(payload["active"] and "open_door" in payload["permissions"]),
        gallery_file=temporary_gallery,
        face_feature_extractor=extract_face_live,
    )
    updated = deployment.load_gallery(temporary_gallery)
    meta = dict(updated["meta"].get(identifier, {}))
    meta.update(
        {
            "display_name": payload["display_name"],
            "role": payload["role"],
            "access_permissions": payload["permissions"],
            "active": bool(payload["active"]),
            "enrollment_source": "admin-consented-local-media",
        }
    )
    updated["meta"][identifier] = meta
    updated["permissions"][identifier] = bool(payload["active"] and "open_door" in payload["permissions"])
    updated["schema_version"] = 1
    users_root = _users_root()
    users_root.mkdir(parents=True, exist_ok=True)
    if target.exists() and not replace:
        raise ValueError("A private enrollment directory already exists for this identifier")

    backup = None
    moved = False
    gallery_committed = False
    try:
        if target.exists():
            backup = users_root / f".replaced-{identifier}-{uuid.uuid4().hex}"
            target.rename(backup)
        # Move the fully validated media first. The gallery is not changed
        # until this succeeds, so a failed save cannot create a gallery-only
        # or media-only user.
        shutil.move(str(staging), str(target))
        moved = True
        _atomic_dump_gallery(updated)
        gallery_committed = True
        try:
            (target / "working-gallery.joblib").unlink(missing_ok=True)
        except OSError:
            pass
        if backup is not None and backup.exists():
            try:
                _safe_remove(backup, users_root)
            except OSError:
                # The new committed record is valid; leave the old backup for
                # later cleanup rather than turning success into a false error.
                pass
    except Exception:
        if not gallery_committed:
            if moved and target.exists():
                rollback_dir = users_root / f".rollback-{uuid.uuid4().hex}"
                try:
                    target.rename(rollback_dir)
                    if backup is not None and backup.exists():
                        backup.rename(target)
                    else:
                        _safe_remove(rollback_dir, users_root)
                except OSError:
                    # Keep the failed staging/backup visible to the local
                    # administrator rather than deleting user data blindly.
                    pass
            elif backup is not None and backup.exists() and not target.exists():
                backup.rename(target)
        raise


def _users_for_view() -> list[dict]:
    gallery = deployment.load_gallery(_gallery_path())
    users = []
    for identifier in sorted(gallery.get("templates", {})):
        meta = dict(gallery.get("meta", {}).get(identifier, {}))
        users.append(
            {
                "identifier": identifier,
                "display_name": meta.get("display_name") or identifier,
                "role": meta.get("role") or "operator",
                "permissions": meta.get("access_permissions") or ([] if not gallery.get("permissions", {}).get(identifier) else ["open_door"]),
                "active": bool(meta.get("active", gallery.get("permissions", {}).get(identifier, False))),
                "created_utc": meta.get("created_utc", ""),
                "photos": meta.get("photos", 0),
                "recordings": meta.get("recordings", 0),
            }
        )
    return users


def _delete_user(identifier: str) -> dict:
    """Delete a registered user without creating a registry/storage split.

    Private media is first moved to a same-volume quarantine path. The gallery
    is committed only after that move succeeds; if the gallery write fails, the
    directory is restored. The final quarantine cleanup is best effort, so a
    protected recovery copy never blocks re-enrollment at the original ID.
    """
    gallery = deployment.load_gallery(_gallery_path())
    if identifier not in gallery.get("templates", {}):
        raise KeyError(identifier)

    target = _user_storage_path(identifier)
    quarantine = None
    recovery_root = None
    if target.exists():
        recovery_root = _path(_config("PRIVATE_ROOT") / ".backups" / "deleted-users")
        try:
            recovery_root.mkdir(parents=True, exist_ok=True)
            quarantine = recovery_root / f"{identifier}-{uuid.uuid4().hex}.pending"
            target.rename(quarantine)
        except OSError as exc:
            raise StorageConsistencyError(
                "PRIVATE_STORAGE_DELETE_FAILED",
                "The user was not deleted because private enrollment storage could not be moved safely. The registry was left unchanged.",
            ) from exc

    updated = copy.deepcopy(gallery)
    for key in ("templates", "permissions", "meta", "sample_vectors"):
        updated.setdefault(key, {}).pop(identifier, None)
    try:
        _atomic_dump_gallery(updated)
    except Exception:
        if quarantine is not None and quarantine.exists():
            try:
                quarantine.rename(target)
            except OSError as rollback_exc:
                raise StorageConsistencyError(
                    "DELETE_ROLLBACK_FAILED",
                    "The registry update failed and private enrollment storage could not be restored. Administrator recovery is required.",
                    status=500,
                ) from rollback_exc
        raise

    cleanup = "complete"
    if quarantine is not None:
        try:
            _safe_remove(quarantine, recovery_root)
        except (OSError, ValueError):
            cleanup = "pending"

    message = "User removed from the local gallery and private enrollment storage."
    if cleanup == "pending":
        message = "User removed from the local gallery. Private media is retained in protected recovery storage for cleanup."
    return {"cleanup": cleanup, "message": message}


def _update_user(identifier: str, data) -> dict:
    gallery = deployment.load_gallery(_gallery_path())
    if identifier not in gallery.get("templates", {}):
        raise KeyError(identifier)
    meta = dict(gallery.get("meta", {}).get(identifier, {}))
    if hasattr(data, "getlist"):
        permissions = data.getlist("access_permissions")
        role = data.get("role")
        active_value = data.get("active")
        display_name = data.get("display_name")
    else:
        permissions = data.get("access_permissions")
        role = data.get("role")
        active_value = data.get("active")
        display_name = data.get("display_name")
    if permissions is not None:
        if isinstance(permissions, str):
            permissions = [permissions]
        permissions = sorted({str(item) for item in permissions if str(item) in ACCESS_PERMISSIONS})
        meta["access_permissions"] = permissions
    if role is not None:
        if str(role) not in ROLES:
            raise ValueError("Choose a supported role")
        meta["role"] = str(role)
    if display_name is not None:
        clean_name = " ".join(str(display_name).split())
        if not clean_name or len(clean_name) > 80:
            raise ValueError("Display name is invalid")
        meta["display_name"] = clean_name
    if active_value is not None:
        meta["active"] = str(active_value).lower() in {"1", "true", "yes", "on"}
    else:
        meta["active"] = bool(meta.get("active", gallery["permissions"].get(identifier, False)))
    gallery.setdefault("meta", {})[identifier] = meta
    gallery.setdefault("permissions", {})[identifier] = bool(
        meta["active"] and "open_door" in meta.get("access_permissions", [])
    )
    _atomic_dump_gallery(gallery)
    return {
        "identifier": identifier,
        "display_name": meta.get("display_name") or identifier,
        "role": meta.get("role", "operator"),
        "permissions": meta.get("access_permissions", []),
        "active": meta["active"],
    }


def _command_policy(gallery: dict) -> dict[str, set[str]]:
    policy = {}
    for identifier, meta in gallery.get("meta", {}).items():
        permissions = set(meta.get("access_permissions", []))
        actions = set()
        if "command_open" in permissions:
            actions.add("open")
        if "command_close" in permissions:
            actions.add("close")
        policy[identifier] = actions
    return policy


try:  # Import-friendly object; production launch uses the strict factory below.
    app = _module_app()
except Exception:  # pragma: no cover
    app = None


def main() -> None:
    try:
        application = create_app()
    except RuntimeError as exc:
        raise SystemExit(f"BioGuard configuration error: {exc}") from None
    host = os.environ.get("BIOGUARD_HOST", "127.0.0.1")
    port = int(os.environ.get("BIOGUARD_PORT", "5000"))
    application.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
