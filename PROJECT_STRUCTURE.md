# Project structure

```text
BioGuard-Secret-Door/
├── bioguard_core.py                 # Shared HOG/LBP, MFCC/ZCR/RMS, and fingerprint features
├── bioguard_matching.py             # Shared local-gallery matching implementation
├── public_research.py               # Public RAVDESS experiments and evaluation helpers
├── deployment.py                    # Public deployment/model adapters
├── open_set.py                      # Unknown-identity evaluation helpers
├── fingerprint_research.py          # Sobel/LBP fingerprint research and upload adapter
├── speech_commands_research.py      # Mini Speech Commands preparation/inference
├── prepare_ravdess.py               # Local RAVDESS preparation
├── model/
│   ├── public_models.joblib         # Public research model bundle
│   ├── fingerprint_model.joblib     # Reviewed offline fingerprint model bundle
│   ├── commands_model.joblib        # Reviewed speech-command model bundle
│   ├── model_manifest.json          # Public model thresholds and feature descriptions
│   └── *_comparison.csv             # Saved evaluation summaries and predictions
├── flask_game/
│   ├── app.py                       # Flask application and protected routes
│   ├── auth.py                      # Admin/session protections
│   ├── inference.py                 # Flask adapter importing shared matching logic
│   ├── local_matching.py            # Local gallery storage/matching helpers
│   ├── fingerprint_inference.py     # Fingerprint-specific upload adapter
│   ├── local_matching_calibration.json
│   ├── templates/                   # HTML pages
│   ├── static/                      # CSS and browser JavaScript
│   ├── tests/                       # Flask integration/security tests
│   └── tools/                       # Local benchmark/audit utilities
├── tests/test_public_snapshot.py    # Dataset-free publication smoke test
├── outputs/                         # Public research figures only
├── 01_*.ipynb                       # Main multimodal notebook
├── 02_*.ipynb                       # Fingerprint notebook
├── 03_*.ipynb                       # Speech-command notebook
├── DATASETS.md
├── MODEL_CARD.md
├── SECURITY.md
├── SETUP.md
├── START_GAME.bat
└── .github/workflows/ci.yml
```

The following local paths are intentionally absent from the public tree and ignored by Git: `flask_game/private_data/`, `flask_game/backups/`, `public_data/`, `command_data/`, `fingerprint_data/`, `FINGERPRINT_DEMO/`, raw RAVDESS archives, virtual environments, caches, logs, and old notebook backups.
