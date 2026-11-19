"""Small, local-only authentication and request-protection helpers.

The game is an educational local demo, but its admin endpoints still fail
closed: the first run creates a local password hash, admin mutations require
a session and CSRF token, and repeated failures are rate limited in memory.
"""
from __future__ import annotations

from collections import defaultdict, deque
from functools import wraps
import hashlib
import hmac
import secrets
import time

from flask import current_app, jsonify, redirect, request, session, url_for
from werkzeug.security import check_password_hash


def _bucket(name: str) -> dict[str, deque[float]]:
    buckets = current_app.extensions.setdefault("bioguard_rate_buckets", {})
    return buckets.setdefault(name, defaultdict(deque))


def rate_limit(scope: str, limit: int, window: int, key: str | None = None) -> tuple[bool, int]:
    """Return ``(allowed, retry_after_seconds)`` for a small in-memory limit."""
    now = time.monotonic()
    bucket = _bucket(scope)
    identity = key or request.remote_addr or "local"
    events = bucket[identity]
    while events and now - events[0] >= window:
        events.popleft()
    if len(events) >= limit:
        retry = max(1, int(window - (now - events[0])))
        return False, retry
    events.append(now)
    return True, 0


def csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def _submitted_csrf() -> str:
    return request.headers.get("X-CSRFToken") or request.form.get("csrf_token") or (
        (request.get_json(silent=True) or {}).get("csrf_token", "")
    )


def csrf_is_valid() -> bool:
    token = session.get("csrf_token")
    submitted = _submitted_csrf()
    return bool(token and submitted and hmac.compare_digest(str(token), str(submitted)))


def require_csrf(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not csrf_is_valid():
            if request.accept_mimetypes.best == "application/json" or request.path.startswith("/api/"):
                return jsonify({"ok": False, "error": "CSRF validation failed"}), 400
            return "CSRF validation failed", 400
        return view(*args, **kwargs)

    return wrapped


def is_admin() -> bool:
    return bool(session.get("admin_authenticated"))


def admin_password_configured() -> bool:
    configured = current_app.config.get("ADMIN_PASSWORD") or current_app.config.get("ADMIN_PASSWORD_HASH")
    return bool(configured and not current_app.config.get("SECURITY_CONFIGURATION_ERROR"))


def check_password(candidate: str) -> bool:
    expected = str(current_app.config.get("ADMIN_PASSWORD") or "")
    stored_hash = current_app.config.get("ADMIN_PASSWORD_HASH")
    if current_app.config.get("SECURITY_CONFIGURATION_ERROR"):
        return False
    if stored_hash:
        try:
            return bool(check_password_hash(str(stored_hash), str(candidate)))
        except (ValueError, TypeError):
            return False
    if not expected:
        return False
    # Hash both sides before compare so a variable-length password does not
    # create a useful timing signal in this local demo.
    left = hashlib.sha256(str(candidate).encode("utf-8")).digest()
    right = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(left, right)


def log_in_admin() -> None:
    session.clear()
    session["admin_authenticated"] = True
    session["admin_login_at"] = int(time.time())
    csrf_token()


def log_out_admin() -> None:
    session.clear()


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not is_admin():
            if request.path.startswith("/api/") or request.accept_mimetypes.best == "application/json":
                return jsonify({"ok": False, "error": "Admin authentication required"}), 401
            return redirect(url_for("admin_login", next=request.path))
        return view(*args, **kwargs)

    return wrapped
