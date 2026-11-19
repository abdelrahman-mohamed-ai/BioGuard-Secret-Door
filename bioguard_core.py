"""BioGuard Task 2 - all feature extractors & verification models use classical ML.
No pretrained face recognition / speech recognition / neural-network models.
This is an educational local demonstration, NOT production access control.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from dataclasses import dataclass

import cv2
import joblib
import librosa
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
SR = 16000
DURATION = 3
SPLITS = ("enroll", "train", "val", "test")
MODALITIES = ("face", "voice", "fingerprint")
FACE_DIM = 1764 + 10
VOICE_DIM = 82
FINGER_DIM = 29


def _gray(image):
    if image is None:
        raise ValueError("Image could not be loaded")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _lbp_hist(gray):
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    histogram = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10))[0].astype(np.float32)
    return histogram / max(histogram.sum(), 1.0)


def extract_face(image_or_path, require_face=True, already_cropped=False):
    """Face crop with classic Haar detector; HOG + LBP, grayscale 96x96.
    already_cropped=True for public images ALREADY cropped during ZIP preparation.
    """
    image = cv2.imread(str(image_or_path)) if isinstance(image_or_path, (str, Path)) else image_or_path
    gray = _gray(image)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("OpenCV Haar cascade is missing")
    faces = [] if already_cropped else detector.detectMultiScale(gray, 1.1, 4, minSize=(35, 35))
    if len(faces) == 0:
        if require_face and not already_cropped:
            raise ValueError("No frontal face detected: capture again with good lighting")
        crop = gray
    else:
        x, y, w, h = max(faces, key=lambda f: int(f[2])*int(f[3]))
        crop = gray[y:y+h, x:x+w]
    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
    crop = cv2.equalizeHist(crop)
    hog_vec = hog(crop, orientations=9, pixels_per_cell=(12, 12),
                  cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    result = np.concatenate((hog_vec, _lbp_hist(crop))).astype(np.float32)
    assert len(result) == FACE_DIM
    return result


def extract_voice(wav_path):
    """MFCC 13 + delta/delta2, aggregated mean & std, ZCR & RMS -> 82 values."""
    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < int(0.6 * sr):
        raise ValueError(f"Voice too short / silent: {wav_path}")
    y = librosa.util.fix_length(y, size=DURATION * sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=256)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)
    parts = []
    for arr in (mfcc, delta, delta2, zcr, rms):
        parts.extend((arr.mean(axis=1), arr.std(axis=1)))
    result = np.concatenate(parts).astype(np.float32)
    assert len(result) == VOICE_DIM
    return result


def extract_fingerprint(image_or_path):
    """Sobel horizontal/vertical gradient orientations + uniform LBP texture."""
    image = cv2.imread(str(image_or_path)) if isinstance(image_or_path, (str, Path)) else image_or_path
    gray = cv2.resize(_gray(image), (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    gx = cv2.Sobel(gray.astype(np.float32)/255, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32)/255, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    orientation = (np.arctan2(gy, gx) + np.pi) % np.pi
    orientation_hist, _ = np.histogram(orientation, bins=16, range=(0, np.pi), weights=magnitude)
    orientation_hist = orientation_hist / (orientation_hist.sum() + 1e-10)
    lbp_hist = _lbp_hist(gray)
    stats = np.array([magnitude.mean(), magnitude.std(), np.percentile(magnitude, 90)])
    result = np.concatenate((orientation_hist, lbp_hist, stats)).astype(np.float32)
    assert len(result) == FINGER_DIM
    return result


EXTRACTORS = {"face": extract_face, "voice": extract_voice, "fingerprint": extract_fingerprint}
EXTENSIONS = {"face": (".jpg", ".jpeg", ".png"), "voice": (".wav",), "fingerprint": (".png", ".jpg")}


def scan_dataset(root="data"):
    """Rows for paired face+voice; fingerprint optional. IDs must match across modalities."""
    base = Path(root)
    records = []
    if not base.exists():
        return pd.DataFrame(columns=["user", "split", "sample", *MODALITIES])
    for person in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for split in SPLITS:
            phase = person / split
            folder = phase / "face"
            if not folder.is_dir():
                continue
            for image in sorted(folder.iterdir()):
                if image.suffix.lower() not in EXTENSIONS["face"]:
                    continue
                sample = image.stem
                voice = phase / "voice" / (sample + ".wav")
                if not voice.exists():
                    continue
                finger = next((phase / "fingerprint" / (sample + ext) for ext in EXTENSIONS["fingerprint"]
                               if (phase / "fingerprint" / (sample + ext)).exists()), None)
                records.append({"user": person.name, "split": split, "sample": sample,
                                "face": str(image), "voice": str(voice),
                                "fingerprint": str(finger) if finger else None})
    return pd.DataFrame.from_records(records, columns=["user", "split", "sample", *MODALITIES])


def dataset_summary(data):
    if data.empty:
        return pd.DataFrame()
    return data.groupby(["user", "split"]).agg(pairs=("sample", "count"),
                      fingerprint_pairs=("fingerprint", lambda x: x.notna().sum())).reset_index()


def build_features(rows, use_fingerprint=False):
    rows = rows.copy()
    cache = {}
    for mod in ("face", "voice", "fingerprint"):
        if mod == "fingerprint" and not use_fingerprint:
            continue
        output = []
        for path in rows[mod]:
            if pd.isna(path):
                raise ValueError(f"Missing {mod} image/audio in a paired sample")
            if path not in cache:
                cache[path] = EXTRACTORS[mod](path)
            output.append(cache[path])
        rows[mod + "_vec"] = output
    return rows


def build_templates(rows, modalities):
    templates = {}
    enrolled = rows[rows.split == "enroll"]
    for user, part in enrolled.groupby("user"):
        templates[user] = {}
        for mod in modalities:
            vectors = np.stack(part[mod + "_vec"].values)
            if len(vectors) < 3:
                raise ValueError(f"Need at least 3 enrollment samples per user: {user}/{mod}")
            templates[user][mod] = np.mean(vectors, axis=0)
    return templates


def pair_difference(x, template):
    """Claim-conditioned features: absolute difference + normalized cosine similarity."""
    diff = np.abs(x - template)
    cos = np.dot(x, template)/(np.linalg.norm(x)*np.linalg.norm(template)+1e-9)
    return np.concatenate((diff, np.array([cos], dtype=np.float32))).astype(np.float32)


def make_claim_trials(rows, templates, modalities):
    """For each *real* sample test true claim and wrong claims, without reusing enrollment.
    All comparison modes get exactly the same claim trials and labels.
    """
    output = []
    users = sorted(templates)
    if len(users) < 2:
        raise ValueError("At least 2 distinct people are needed for genuine/impostor trials")
    for idx, sample in rows[rows.split != "enroll"].iterrows():
        for claim in users:
            record = {"row_idx": idx, "actual": sample.user, "claim": claim,
                      "split": sample.split, "sample": sample["sample"], "label": int(claim == sample.user)}
            for mod in modalities:
                record[mod] = pair_difference(sample[mod + "_vec"], templates[claim][mod])
            output.append(record)
    return pd.DataFrame(output)


def make_model(name):
    if name == "SVM":
        return Pipeline([("scaler", StandardScaler()),
                         ("model", SVC(kernel="rbf", C=2.0, probability=True,
                                        class_weight="balanced", random_state=SEED))])
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                      class_weight="balanced_subsample", random_state=SEED,
                                      n_jobs=-1)
    raise ValueError(name)


def score_model(model, x):
    return model.predict_proba(x)[:, list(model.classes_).index(1)]


def choose_threshold(y, scores, max_far=0.10):
    """Select on validation only: maximize genuine acceptance subject to validation FAR."""
    y, scores = np.asarray(y), np.asarray(scores)
    if len(np.unique(y)) != 2:
        raise ValueError("Validation must include genuine and impostor claims")
    sorted_scores = np.unique(scores)
    candidates = np.concatenate(([np.nextafter(sorted_scores[0], -np.inf)],
                                 (sorted_scores[1:] + sorted_scores[:-1])/2,
                                 [np.nextafter(sorted_scores[-1], np.inf)]))
    best = None
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        far = fp / max(1, fp+tn)
        tar = tp / max(1, tp+fn)
        balanced = (tar + tn/max(1, tn+fp))/2
        if far <= max_far + 1e-10:
            item = (tar, -far, balanced, float(threshold))
            if best is None or item > best:
                best = item
    if best is None:
        return float(np.nextafter(max(scores), np.inf))
    return best[3]


def measure(y_true, scores, threshold):
    y = np.asarray(y_true, dtype=int)
    p = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
    result = {"accuracy": accuracy_score(y,p), "balanced_accuracy": balanced_accuracy_score(y,p),
              "precision": precision_score(y,p,zero_division=0),
              "recall_TAR": recall_score(y,p,zero_division=0),
              "f1": f1_score(y,p,zero_division=0),
              "FAR": fp/max(1,fp+tn), "FRR": fn/max(1,fn+tp),
              "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
              "threshold": float(threshold)}
    try:
        result["AUC"] = float(roc_auc_score(y, scores))
    except ValueError:
        result["AUC"] = float("nan")
    return result


def _stack(trials, mod):
    return np.stack(trials[mod].to_numpy())


def train_and_evaluate(root="data", artifact="artifacts/models.joblib", target_far=0.10):
    manifest = scan_dataset(root)
    if manifest.empty:
        raise ValueError("No aligned face+voice samples. Collect data first (see README).")
    people = sorted(manifest.user.unique())
    if len(people) < 2:
        raise ValueError("Need at least two consenting people; 3-4 is preferred")
    counts = manifest.groupby(["user", "split"]).size().unstack(fill_value=0)
    for user in people:
        for split in SPLITS:
            if counts.loc[user, split] < (3 if split == "enroll" else 2):
                raise ValueError(f"Not enough samples for {user}/{split}; collect >=3 enroll and >=2 each other split")
    use_fingerprint = bool(manifest.fingerprint.notna().all())
    modalities = ("face", "voice", "fingerprint") if use_fingerprint else ("face", "voice")
    features = build_features(manifest, use_fingerprint=use_fingerprint)
    templates = build_templates(features, modalities)
    trials = make_claim_trials(features, templates, modalities)
    train, val, test = [trials[trials.split == part].reset_index(drop=True) for part in ("train", "val", "test")]
    if not len(train) or not len(val) or not len(test):
        raise ValueError("Train, val, test each need aligned samples")
    train_y, val_y, test_y = [phase.label.to_numpy(dtype=int) for phase in (train, val, test)]
    names = ["Face Only", "Voice Only", "Early Fusion"]
    columns = {"Face Only": ["face"], "Voice Only": ["voice"],
               "Early Fusion": ["face", "voice"]}
    if use_fingerprint:
        columns.update({"Fingerprint Only": ["fingerprint"],
                        "Early Fusion (3)": ["face", "voice", "fingerprint"]})
        names.extend(["Fingerprint Only", "Early Fusion (3)"])
    models, selection, scores = {}, [], {}
    for approach in names:
        components = columns[approach]
        arr_train = np.concatenate([_stack(train, col) for col in components], axis=1)
        arr_val = np.concatenate([_stack(val, col) for col in components], axis=1)
        arr_test = np.concatenate([_stack(test, col) for col in components], axis=1)
        possibilities = []
        for model_name in ("SVM", "Random Forest"):
            model = make_model(model_name)
            model.fit(arr_train, train_y)
            val_score = score_model(model, arr_val)
            threshold = choose_threshold(val_y, val_score, target_far)
            val_metrics = measure(val_y, val_score, threshold)
            selection.append({"approach": approach, "model": model_name,
                              "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                              "val_FAR": val_metrics["FAR"], "val_FRR": val_metrics["FRR"],
                              "threshold": threshold})
            possibilities.append((val_metrics["balanced_accuracy"], val_metrics["recall_TAR"],
                                  -val_metrics["FAR"], model_name, model, threshold))
        # Only validation data chooses model + threshold; NEVER use test to choose.
        best = max(possibilities, key=lambda i: i[:3])
        model_name, model, threshold = best[3:6]
        models[approach] = {"model": model, "model_name": model_name,
                            "threshold": threshold, "modalities": components}
        scores[approach] = {"val": score_model(model, arr_val),
                            "test": score_model(model, arr_test)}
    # Late fusion: decision-level averaging, validation-calibrated threshold.
    # Both face & voice decisions are computed independently first.
    for phase in ("val", "test"):
        scores.setdefault("Late Fusion", {})[phase] = 0.5*(scores["Face Only"][phase] + scores["Voice Only"][phase])
    late_threshold = choose_threshold(val_y, scores["Late Fusion"]["val"], target_far)
    # Hierarchical: PASS face THEN voice, hard AND; no score concatenation.
    for phase in ("val", "test"):
        accepted_face = scores["Face Only"][phase] >= models["Face Only"]["threshold"]
        accepted_voice = scores["Voice Only"][phase] >= models["Voice Only"]["threshold"]
        scores.setdefault("Hierarchical", {})[phase] = (accepted_face & accepted_voice).astype(float)
        if use_fingerprint:
            accepted_finger = scores["Fingerprint Only"][phase] >= models["Fingerprint Only"]["threshold"]
            scores.setdefault("Hierarchical (3)", {})[phase] = (accepted_face & accepted_voice & accepted_finger).astype(float)
    thresholds = {k: m["threshold"] for k,m in models.items()}
    thresholds["Late Fusion"] = late_threshold
    thresholds["Hierarchical"] = 0.5
    if use_fingerprint:
        thresholds["Hierarchical (3)"] = 0.5
    results = []
    for approach in scores:
        results.append({"approach": approach,
                        "model": models[approach]["model_name"] if approach in models else "Decision rule",
                        **measure(test_y, scores[approach]["test"], thresholds[approach])})
    report = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)
    # No test-set image/audio files go into the saved model artifact.
    package = {"models": models, "templates": templates, "users": people,
               "use_fingerprint": use_fingerprint, "late_threshold": late_threshold,
               "data_dimensions": {"face": FACE_DIM, "voice": VOICE_DIM, "fingerprint": FINGER_DIM},
               "target_validation_FAR": target_far, "reference_phrase": "BioGuard access",
               "notes": "Academic-only; uncalibrated model probabilities; no anti-spoof guarantee"}
    output = Path(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, output)
    report.to_csv(output.parent / "comparison.csv", index=False)
    pd.DataFrame(selection).to_csv(output.parent / "model_selection_validation.csv", index=False)
    return {"report": report, "selection": pd.DataFrame(selection), "artifact": str(output),
            "manifest": manifest, "trials": trials, "scores": scores,
            "y_test": test_y, "thresholds": thresholds, "use_fingerprint": use_fingerprint}


def verify_attempt(package, claimed_user, face_image=None, voice_wav=None,
                   fingerprint_image=None, mode="Hierarchical"):
    """Real 1:1 verification. Hierarchical gates truly short-circuit in capture order."""
    if claimed_user not in package["templates"]:
        return {"accepted": False, "reason": "Unknown claim", "stages": {}}
    templates = package["templates"][claimed_user]
    needs_face = mode not in ("Voice Only", "Fingerprint Only")
    needs_voice = mode not in ("Face Only", "Fingerprint Only")
    needs_fingerprint = mode in ("Fingerprint Only", "Early Fusion (3)", "Hierarchical (3)")
    hierarchical = mode in ("Hierarchical", "Hierarchical (3)")
    vectors = {}
    stages = {}

    def run(approach):
        entry = package["models"][approach]
        x = np.concatenate([vectors[mod] for mod in entry["modalities"]])[None, :]
        score = float(score_model(entry["model"], x)[0])
        passed = bool(score >= entry["threshold"])
        stages[approach] = {"score": score, "threshold": float(entry["threshold"]), "pass": passed}
        return score, passed

    if needs_face:
        try:
            if face_image is None:
                raise ValueError("Missing webcam photograph")
            face = extract_face(face_image, require_face=True)
            vectors["face"] = pair_difference(face, templates["face"])
        except Exception as exc:
            return {"accepted": False, "reason": f"Face capture invalid: {exc}", "stages": stages}
        if hierarchical and not run("Face Only")[1]:
            return {"accepted": False, "reason": "Face rejected", "stages": stages}

    if needs_voice:
        try:
            if voice_wav is None:
                raise ValueError("Missing microphone recording")
            voice = extract_voice(voice_wav)
            vectors["voice"] = pair_difference(voice, templates["voice"])
        except Exception as exc:
            return {"accepted": False, "reason": f"Voice capture invalid: {exc}", "stages": stages}
        if hierarchical and not run("Voice Only")[1]:
            return {"accepted": False, "reason": "Voice rejected", "stages": stages}

    if needs_fingerprint:
        if not package["use_fingerprint"] or fingerprint_image is None:
            return {"accepted": False, "reason": "Fingerprint scan required", "stages": stages}
        try:
            fingerprint = extract_fingerprint(fingerprint_image)
            vectors["fingerprint"] = pair_difference(fingerprint, templates["fingerprint"])
        except Exception as exc:
            return {"accepted": False, "reason": f"Fingerprint capture invalid: {exc}", "stages": stages}
        if hierarchical and not run("Fingerprint Only")[1]:
            return {"accepted": False, "reason": "Fingerprint rejected", "stages": stages}

    if hierarchical:
        return {"accepted": True, "reason": "All stages passed", "stages": stages}
    if mode == "Late Fusion":
        face_score, _ = run("Face Only")
        voice_score, _ = run("Voice Only")
        score = (face_score + voice_score)/2
        accepted = bool(score >= package["late_threshold"])
        return {"accepted": accepted, "reason": "Late fusion decision", "score": score,
                "threshold": float(package["late_threshold"]), "stages": stages}
    if mode not in package["models"]:
        raise ValueError(f"Unknown mode: {mode}")
    passed = run(mode)[1]
    return {"accepted": passed, "reason": mode, "stages": stages}


def command_permissions():
    """Demo permissions; customize per team member, not security-grade auth."""
    return {"default": {"status", "help"},
            "admin": {"open", "lock", "status", "help"}}


def scan_commands(root="data"):
    """data/<person>/commands/<train|val|test>/<command>/<sample>.wav"""
    records = []
    for person in sorted(Path(root).glob("*")):
        if not person.is_dir():
            continue
        for split in ("train", "val", "test"):
            phase = person / "commands" / split
            if not phase.exists():
                continue
            for cmd in phase.iterdir():
                if cmd.is_dir():
                    for wav in cmd.glob("*.wav"):
                        records.append({"user": person.name, "split": split,
                                        "command": cmd.name.lower(), "path": str(wav)})
    return pd.DataFrame(records, columns=["user", "split", "command", "path"])


def train_commands(root="data", artifact="artifacts/command_model.joblib"):
    df = scan_commands(root)
    if df.empty:
        raise ValueError("No command files; collect 'open', 'lock', 'status', 'help' recordings")
    if df.command.nunique() < 2:
        raise ValueError("Need at least two different commands")
    for split in ("train", "val", "test"):
        if df[df.split==split].command.nunique() < 2:
            raise ValueError(f"Need 2+ commands in {split}")
    feat = np.stack([extract_voice(p) for p in df.path])
    tr = (df.split=="train").to_numpy()
    va = (df.split=="val").to_numpy()
    te = (df.split=="test").to_numpy()
    models = []
    for name in ("SVM", "Random Forest"):
        model = make_model(name)
        model.fit(feat[tr], df.command[tr])
        accuracy = accuracy_score(df.command[va], model.predict(feat[va]))
        models.append((accuracy, name, model))
    winner = max(models, key=lambda x: x[0])
    model = winner[2]
    report = {"model": winner[1], "validation_accuracy": float(winner[0]),
              "test_accuracy": float(accuracy_score(df.command[te], model.predict(feat[te]))),
              "test_confusion_matrix": confusion_matrix(df.command[te],model.predict(feat[te]),labels=model.classes_).tolist(),
              "classes": model.classes_.tolist()}
    output = Path(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": report["classes"]}, output)
    (output.parent/"command_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def authorize_command(verification_result, voice_command_file, command_package,
                      claimed_user, role_by_user=None):
    """Never execute protected actions before a successful identity check."""
    if not verification_result.get("accepted", False):
        return {"allowed": False, "reason": "Identity not verified"}
    role_by_user = role_by_user or {}
    role = role_by_user.get(claimed_user, "default")
    command = str(command_package["model"].predict(extract_voice(voice_command_file)[None,:])[0])
    allowed = command in command_permissions().get(role, set())
    return {"allowed": bool(allowed), "command": command, "role": role,
            "reason": "Permitted" if allowed else "Not permitted"}
