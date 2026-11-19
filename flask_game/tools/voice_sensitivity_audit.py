"""Aggregate voice robustness diagnostics for the shared local matcher.

This reads the existing public feature cache and public WAV metadata only. It
does not retrain models, alter thresholds, or export raw audio/features.
RAVDESS has no microphone-condition label, so actor is reported only as a
capture/identity proxy and never as a microphone claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bioguard_matching as matching  # noqa: E402
from flask_game.tools.benchmark_local_matching import (  # noqa: E402
    _public_gallery,
    _score,
    benchmark_modality,
    load_public_features,
)


REPORT_PATH = ROOT / "flask_game" / "VOICE_SENSITIVITY_REPORT.md"


def _group_rows(records: list[dict], key: str) -> list[dict]:
    groups: dict[object, list[dict]] = {}
    for record in records:
        groups.setdefault(record[key], []).append(record)
    result = []
    for value, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        scores = np.asarray([row["score"] for row in rows], dtype=float)
        result.append({
            key: value,
            "n": len(rows),
            "mean_score": float(scores.mean()),
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
            "pass_rate": float(np.mean(scores >= rows[0]["threshold"])),
        })
    return result


def main() -> int:
    frame = load_public_features()
    rows, selected = benchmark_modality(frame, "voice")
    # The benchmark table remains a comparison.  The deployed matcher must
    # use the explicitly exported voice configuration rather than silently
    # inheriting whichever method wins a fresh re-run of the comparison.
    runtime = matching.load_reference_calibration(
        ROOT / "flask_game" / "local_matching_calibration.json", "voice"
    ) or selected
    chosen = runtime["method"]
    threshold = float(runtime["threshold"])
    gallery = _public_gallery(frame, "voice")
    records: list[dict] = []
    for _, row in frame[frame.split == "val"].iterrows():
        audio, _ = sf.read(str(row.voice), always_2d=False)
        audio = np.asarray(audio, dtype=float)
        records.append({
            "user": str(row.user),
            "statement": int(row.statement),
            "repetition": int(row.repetition),
            "score": _score(row.voice_vec, gallery[str(row.user)], "voice", chosen),
            "threshold": threshold,
            "duration": float(sf.info(str(row.voice)).duration),
            "rms": float(np.sqrt(np.mean(audio * audio))),
        })

    rms_values = np.asarray([row["rms"] for row in records], dtype=float)
    cut1, cut2 = np.quantile(rms_values, [1 / 3, 2 / 3])
    for row in records:
        row["volume_bin"] = "low" if row["rms"] <= cut1 else "mid" if row["rms"] <= cut2 else "high"

    by_volume = _group_rows(records, "volume_bin")
    by_statement = _group_rows(records, "statement")
    by_repetition = _group_rows(records, "repetition")
    actor_ids = sorted({row["user"] for row in records})
    actor_labels = {actor_id: f"capture_cohort_{index + 1}" for index, actor_id in enumerate(actor_ids)}
    for row in records:
        row["capture_cohort"] = actor_labels[row["user"]]
    by_actor = _group_rows(records, "capture_cohort")

    def table(rows_to_write: list[dict], key: str) -> list[str]:
        lines = [f"| {key} | n | mean score | min | max | pass rate |", "|---|---:|---:|---:|---:|---:|"]
        for row in rows_to_write:
            lines.append(
                f"| {row[key]} | {row['n']} | {row['mean_score']:.6f} | {row['min_score']:.6f} | {row['max_score']:.6f} | {row['pass_rate']:.4f} |"
            )
        return lines

    alternatives = {row["method"]: row for row in rows}
    comparison_methods = ["cosine_nn", "cosine_mean", "cosine_knn3", "zcos_nn", "zcos_knn3", "zdist_nn", "zdist_knn3", "voice_shape_cos_nn"]
    lines = [
        "# BioGuard Voice Sensitivity Report",
        "",
        "Generated from the existing public validation partition using the shared `bioguard_matching.py` score functions. No raw audio, feature vectors, or local private identifiers are written.",
        "",
        "## Preprocessing contract",
        "",
        "Enrollment and live verification both call the same adapter: browser/container audio is converted to mono 16 kHz WAV, `librosa.load` trims at `top_db=30`, the signal is fixed to 3 seconds, and `bioguard_core.extract_voice` computes MFCC(13) + delta + delta2 + ZCR + RMS means and standard deviations (82 values).",
        "",
        f"Selected runtime method: `{chosen}`. Versioned runtime threshold: `{threshold:.15f}` from `flask_game/local_matching_calibration.json`; target validation FAR is `{matching.TARGET_VALIDATION_FAR}`.",
        "",
        "## Validation genuine and unknown comparison",
        "",
        "| Method | Threshold | Validation genuine pass | Validation unknown rejection | Held-out known ID | Held-out unknown rejection |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in comparison_methods:
        row = alternatives[method]
        lines.append(
            f"| `{method}` | {row['threshold']:.6f} | {row['validation_known_id_rate']:.4f} | {row['validation_unknown_rejection_rate']:.4f} | {row['test_known_id_rate']:.4f} | {row['test_unknown_rejection_rate']:.4f} |"
        )
    lines += [
        "",
        f"`Actor 06` is the validation unknown partition; `Actor 07` is held out for final reporting. The deployed `{chosen}` configuration is selected from the existing validation evidence and is not tuned on the final test set.",
        "",
        "## Selected genuine sensitivity",
        "",
        "The selected validation set contains 25 genuine clips. Public RAVDESS statement IDs are used as a word/utterance proxy; they are not local user phrases.",
        "",
        *table(by_statement, "statement"),
        "",
        *table(by_repetition, "repetition"),
        "",
        "### Volume proxy",
        "",
        "The source dataset has no microphone-condition label. RMS is therefore only a recorded-volume proxy; it is not a controlled microphone experiment.",
        "",
        *table(by_volume, "volume_bin"),
        "",
        f"Raw source duration range before the shared 3-second fix-length step: `{min(row['duration'] for row in records):.4f}`–`{max(row['duration'] for row in records):.4f}` seconds. RMS range: `{min(row['rms'] for row in records):.6f}`–`{max(row['rms'] for row in records):.6f}`.",
        "",
        "### Actor/capture proxy",
        "",
        "Actor-level differences show capture/identity variability in the public data, but cannot identify a microphone effect.",
        "",
        *table(by_actor, "capture_cohort"),
        "",
        "## Deployment conclusion",
        "",
        "The current selected matcher is sensitive to recorded volume and utterance variation. The one-user private gallery has no independent local impostor population, so the current threshold remains provisional and uncertain local attempts stay denied. Additional independent local speakers, held-out phrases, and controlled microphone/volume captures are required before recalibrating a deployment threshold.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "report": str(REPORT_PATH),
        "method": chosen,
        "threshold": threshold,
        "validation_genuine": len(records),
        "validation_pass_rate": float(np.mean([row["score"] >= threshold for row in records])),
        "duration_range": [min(row["duration"] for row in records), max(row["duration"] for row in records)],
        "rms_range": [min(row["rms"] for row in records), max(row["rms"] for row in records)],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
