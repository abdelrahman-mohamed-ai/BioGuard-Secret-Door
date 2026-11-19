from __future__ import annotations

import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]


def test_public_research_files_are_present_and_parseable() -> None:
    notebooks = [
        ROOT / "01_BioGuard_Full_Master_Notebook.ipynb",
        ROOT / "02_Fingerprint_Bonus.ipynb",
        ROOT / "03_Speech_Commands_Bonus_QUALITY_FIXED.ipynb",
    ]
    assert all(path.is_file() for path in notebooks)
    for path in notebooks:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["cells"]

    for name in (
        "public_models.joblib",
        "fingerprint_model.joblib",
        "commands_model.joblib",
    ):
        assert (ROOT / "model" / name).is_file()


def test_reviewed_model_bundles_load_without_private_gallery_paths() -> None:
    public = joblib.load(ROOT / "model" / "public_models.joblib")
    fingerprint = joblib.load(ROOT / "model" / "fingerprint_model.joblib")
    commands = joblib.load(ROOT / "model" / "commands_model.joblib")

    assert set(public["models"]) >= {"Image Only", "Voice Only", "Video Only"}
    assert len(public["public_actors"]) == 5
    assert fingerprint["model_name"] == "SVM"
    assert fingerprint["threshold"] > 0
    assert len(fingerprint["demo_gallery"]) == 25
    assert commands["name"] == "SVM"
    assert set(commands["actions"]) == {"go", "stop"}


def test_private_runtime_directories_are_not_in_public_snapshot() -> None:
    forbidden = (
        ROOT / "flask_game" / "private_data",
        ROOT / "flask_game" / "backups",
        ROOT / "public_data",
        ROOT / "command_data",
        ROOT / "fingerprint_data",
    )
    assert all(not path.exists() for path in forbidden)
