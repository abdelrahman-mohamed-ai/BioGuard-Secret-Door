r"""Read-only BioGuard face model audit (no images/templates exported).

Run from the project folder:
    .\.venv\Scripts\python.exe .\bioguard_face_audit.py
    .\.venv\Scripts\python.exe .\bioguard_face_audit.py --capture

Only local score-only CSV files are written under flask_game/private_data/. No
models, gallery, or raw media are modified. A camera probe is never saved.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "flask_game"))


def load():
    import deployment
    import public_research
    from inference import extract_face_live

    model_path = ROOT / "model" / "public_models.joblib"
    gallery_path = ROOT / "flask_game" / "private_data" / "enrollment_gallery.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Public model missing: {model_path}")
    if not gallery_path.is_file():
        raise FileNotFoundError(f"Private enrollment gallery missing: {gallery_path}")
    pkg = deployment.load_verifier(model_path)
    gallery = deployment.load_gallery(gallery_path)
    return pkg, gallery, public_research, extract_face_live


def score_one(pkg, public_research, vector, name, templates):
    return float(public_research.score_claim(
        pkg, {"image": vector}, name, "Image Only", templates
    ))


def report_scores(pkg, gallery, public_research, face_extract):
    threshold = float(pkg["models"]["Image Only"]["threshold"])
    templates = gallery.get("templates", {})
    print(f"FACE threshold (PUBLIC RAVDESS, not locally calibrated): {threshold:.6f}")
    print(f"Local enrollment gallery users: {len(templates)}")
    if not templates:
        print("No locally enrolled users found. Nothing to test.")
        return [], threshold
    rows = []
    for user in sorted(templates):
        directory = ROOT / "flask_game" / "private_data" / "users" / user / "faces"
        files = sorted(p for p in directory.glob("*")
                       if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            print(f"{user}: no saved face images at {directory}")
            continue
        good = []
        for path in files:
            try:
                feature = np.asarray(face_extract(path), dtype=np.float32)
                good.append((path.name, feature))
            except Exception as err:
                print(f"{user}: image not readable/detectable ({path.name}): {err}")
        if not good:
            continue
        actual = np.stack([v for _, v in good])
        stored = np.asarray(templates[user]["image"], dtype=np.float32)
        template_max_error = float(np.max(np.abs(stored - actual.mean(axis=0))))
        print(f"\nUSER {user}: {len(good)} readable saved faces; stored-template mean max-abs error={template_max_error:.7f}")
        if template_max_error > 1e-4:
            print("WARNING: saved media features do not match the stored template! Inspect enrollment pipeline / replaced images.")
        for i, (name, feature) in enumerate(good):
            all_scores = {other: score_one(pkg, public_research, feature, other, templates) for other in templates}
            own_score = all_scores[user]
            runner = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            best_user, best_score = runner[0]
            if len(good) > 1:
                loo = np.delete(actual, i, axis=0).mean(axis=0)
                custom = {user: {"image": loo}}
                leave_one_out_score = score_one(pkg, public_research, feature, user, custom)
            else:
                leave_one_out_score = float("nan")
            rows.append({
                "user": user,
                "sample": name,
                "own_public_score": round(own_score, 6),
                "public_threshold": round(threshold, 6),
                "own_pass": own_score >= threshold,
                "best_candidate": best_user,
                "best_candidate_score": round(best_score, 6),
                "correct_top_candidate": best_user == user,
                "loo_score_without_this_photo": round(leave_one_out_score, 6),
                "loo_pass": leave_one_out_score >= threshold,
                "stored_template_mean_error": round(template_max_error, 8),
            })
            print(f"  {name}: own={own_score:.4f}  LOO={leave_one_out_score:.4f}  "
                  f"threshold={threshold:.4f}  public_pass={'YES' if own_score >= threshold else 'NO'}  "
                  f"top={best_user}")
        own_pass = sum(bool(r["own_pass"]) for r in rows if r["user"] == user)
        print(f"  SUMMARY {user}: {own_pass}/{len(good)} saved faces accepted by public score gate")

    if len(templates) < 2:
        print("\nWARNING: Only one enrolled identity. You cannot assess unknown-user false acceptance from a single-user gallery.")
    if rows:
        out = ROOT / "flask_game" / "private_data" / "face_audit_scores.csv"
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print("\nLocal score-only report:", out)
        print("No raw images or biometric feature vectors have been exported.")
    return rows, threshold


def capture_probe(extract):
    import cv2
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError("Unable to open local camera. You can use --probe path/to/fresh_face.jpg instead.")
    print("Camera: press SPACE to evaluate a new frame, ESC to cancel. No photo will be saved.")
    frame = None
    try:
        while True:
            okay, image = cap.read()
            if not okay:
                raise RuntimeError("Camera opened but no frame could be read")
            cv2.imshow("Local BioGuard diagnosis — SPACE=capture ESC=cancel", image)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('q')):
                return None
            if key == 32:
                frame = image
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return extract(frame)


def run_probe(feature, pkg, gallery, research, threshold):
    rankings = []
    for user in gallery["templates"]:
        score = score_one(pkg, research, feature, user, gallery["templates"])
        rankings.append((user, score))
    rankings.sort(key=lambda pair: pair[1], reverse=True)
    print("\nFRESH FACE PROBE (not used for enrollment):")
    if not rankings:
        print("No registered users")
        return
    for user, score in rankings:
        print(f"  {user}: {score:.5f} vs {threshold:.5f} => {'PASS' if score >= threshold else 'REJECT'}")
    if rankings[0][1] < threshold:
        print("RESULT: Face gate would reject this frame; microphone will never be requested.")
    else:
        print("RESULT: Face stage might advance to voice; this says nothing about voice or unknown rejection safety.")



def report_voice_scores(pkg, gallery, research):
    """Audit stored voice samples against the actual public Voice Only model."""
    import bioguard_core as core
    threshold = float(pkg["models"]["Voice Only"]["threshold"])
    templates = gallery.get("templates", {})
    print(f"\nVOICE threshold (PUBLIC RAVDESS, not locally calibrated): {threshold:.6f}")
    rows = []
    for user in sorted(templates):
        directory = ROOT / "flask_game" / "private_data" / "users" / user / "voices"
        files = sorted(p for p in directory.glob("*") if p.is_file() and p.suffix.lower() == ".wav")
        if not files:
            print(f"{user}: no saved WAVs at {directory}")
            continue
        good = []
        for path in files:
            try:
                good.append((path.name, np.asarray(core.extract_voice(path), dtype=np.float32)))
            except Exception as err:
                print(f"{user}: audio invalid ({path.name}): {err}")
        if not good:
            continue
        actual = np.stack([v for _, v in good])
        stored = np.asarray(templates[user]["voice"], dtype=np.float32)
        mean_error = float(np.max(np.abs(stored - actual.mean(axis=0))))
        print(f"\nVOICE USER {user}: {len(good)} WAVs; stored template mean max-abs error={mean_error:.7f}")
        for i, (name, feature) in enumerate(good):
            scores = {other: float(research.score_claim(
                pkg, {"voice": feature}, other, "Voice Only", templates)) for other in templates}
            own_score = scores[user]
            best_user = max(scores, key=scores.get)
            if len(good) > 1:
                loo = np.delete(actual, i, axis=0).mean(axis=0)
                custom = {user: {"voice": loo}}
                loo_score = float(research.score_claim(pkg, {"voice": feature}, user, "Voice Only", custom))
            else:
                loo_score = float("nan")
            rows.append({"user": user, "sample": name, "own_score": round(own_score, 6),
                         "threshold": round(threshold, 6), "own_pass": own_score >= threshold,
                         "best_candidate": best_user, "top_correct": best_user == user,
                         "loo_score": round(loo_score, 6), "loo_pass": loo_score >= threshold,
                         "stored_template_mean_error": round(mean_error, 8)})
            print(f"  {name}: own={own_score:.4f} LOO={loo_score:.4f} "
                  f"threshold={threshold:.4f} pass={'YES' if own_score >= threshold else 'NO'}")
        own_pass = sum(bool(r["own_pass"]) for r in rows if r["user"] == user)
        print(f"  VOICE SUMMARY {user}: {own_pass}/{len(good)} saved recordings pass the public voice gate")
    if rows:
        out = ROOT / "flask_game" / "private_data" / "voice_audit_scores.csv"
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print("Local score-only voice report:", out)
    return threshold


def probe_voice(path, pkg, gallery, research, threshold):
    import bioguard_core as core
    vector = core.extract_voice(path)
    print("\nFRESH VOICE PROBE:")
    scores = []
    for user in gallery.get("templates", {}):
        score = float(research.score_claim(pkg, {"voice": vector}, user,
                                          "Voice Only", gallery["templates"]))
        scores.append((user, score))
    for user, score in sorted(scores, key=lambda pair: pair[1], reverse=True):
        print(f"  {user}: {score:.5f} vs {threshold:.5f} => {'PASS' if score >= threshold else 'REJECT'}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--probe", type=Path, help="Test a NEW face image; not an enrollment image")
    source.add_argument("--capture", action="store_true", help="Capture a temporary live webcam frame without saving it")
    parser.add_argument("--probe-voice", type=Path, help="Test a fresh 16kHz WAV; do not use enrollment audio")
    args = parser.parse_args()
    pkg, gallery, research, face_extract = load()
    _, threshold = report_scores(pkg, gallery, research, face_extract)
    voice_threshold = report_voice_scores(pkg, gallery, research)
    if args.probe_voice:
        probe_voice(args.probe_voice, pkg, gallery, research, voice_threshold)
    if args.probe:
        run_probe(face_extract(args.probe), pkg, gallery, research, threshold)
    elif args.capture:
        feature = capture_probe(face_extract)
        if feature is not None:
            run_probe(feature, pkg, gallery, research, threshold)


if __name__ == "__main__":
    main()
