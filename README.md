# BioGuard — Secret Door

BioGuard is a hierarchical multimodal biometric verification system built with traditional machine learning. It combines face, voice, fingerprint, speech-command, and Flask game components into an academic security-door demonstration.

The project deliberately separates public research artifacts from private local enrollment data. A fresh clone starts with an empty local enrollment gallery and asks the administrator to create the first local PIN.

Repository: <https://github.com/abdelrahman-mohamed-ai/BioGuard-Secret-Door>

## What it demonstrates

- Face verification with HOG and uniform LBP features.
- Speaker verification with MFCC, delta features, ZCR, and RMS.
- Video feature extraction from aligned face frames.
- Early and late multimodal fusion experiments.
- Hierarchical face → voice → permission door authentication.
- Dynamic local enrollment for new users using individual gallery templates.
- Voice commands (`go` and `stop`) gated by speaker identity and permissions.
- Offline fingerprint upload verification with Sobel/LBP preprocessing.
- A browser-based futuristic security-door game with explicit failure states.
- Three research notebooks, evaluation CSVs, confusion-matrix figures, and reproducible preparation scripts.

## Important privacy boundary

This public repository does not contain real enrolled users, face photographs, voice recordings, private biometric templates, admin PINs, session secrets, local databases, diagnostic logs, or raw datasets. Those files are created locally at runtime and are ignored by Git.

The included model artifacts are derived from public research data and were checked for local enrollment identifiers and private paths. They are still serialized `joblib`/pickle files: load them only from this trusted repository or another trusted source.

## Machine-learning pipeline

### Face and video

The shared feature code uses a 96×96 grayscale face crop, HOG, and uniform LBP. Video mode extracts the same face representation from multiple aligned frames and aggregates the frame features. The research artifacts include SVM and Random Forest comparisons.

### Voice

Voice features use MFCC plus first and second deltas, zero-crossing rate, and RMS statistics at 16 kHz. The Flask deployment imports the shared `bioguard_matching.py` implementation rather than maintaining a second extractor. Public RAVDESS pair verification and the local enrollment gallery are documented as separate deployment layers.

### Fingerprint

The offline lab uses grayscale preprocessing followed by Sobel and LBP features. The trained fingerprint artifact contains a 29-dimensional demo-template gallery and the validation-selected SVM threshold. Fingerprint upload verification is an academic bonus page and is not required for door entry.

### Speech commands

The command artifact is an SVM for eight isolated words: `down`, `go`, `left`, `no`, `right`, `stop`, `up`, and `yes`. Only `go` and `stop` map to protected actions. The command classifier never replaces speaker verification or permission checks.

## Research notebooks

1. `01_BioGuard_Full_Master_Notebook.ipynb` — face, voice, video, early/late fusion, open-set analysis, and hierarchical verification.
2. `02_Fingerprint_Bonus.ipynb` — Sobel/LBP fingerprint preprocessing, classifier comparison, thresholding, and confusion matrices.
3. `03_Speech_Commands_Bonus_QUALITY_FIXED.ipynb` — isolated-word command preparation, MFCC/ZCR/RMS features, model comparison, and command diagnostics.

The notebooks retain their research outputs. Their output-only local paths were sanitized in the publication snapshot so the notebooks do not disclose the original workstation path. Raw datasets are intentionally not embedded in the repository.

## Verified public evaluation results

The figures below are from the existing saved evaluation CSVs, not from live webcam or microphone tests. Percentages are rounded for readability.

| Mode / selected model | Test accuracy | Precision | Recall / TAR | F1 | FAR | FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image Only / SVM | 98.49% | 95.65% | 88.00% | 91.67% | 0.42% | 12.00% |
| Voice Only / Random Forest | 91.32% | 53.33% | 64.00% | 58.18% | 5.83% | 36.00% |
| Video Only / Random Forest | 99.25% | 100.00% | 92.00% | 95.83% | 0.00% | 8.00% |
| Early All Three / Random Forest | 99.25% | 100.00% | 92.00% | 95.83% | 0.00% | 8.00% |
| Late Fusion / decision rule | 98.49% | 92.00% | 92.00% | 92.00% | 0.83% | 8.00% |
| Hierarchical / decision rule | 95.85% | 100.00% | 56.00% | 71.79% | 0.00% | 44.00% |

Open-set results are also saved in `model/open_set_comparison.csv`. For example, Early All Three reports 92.00% known-identity accuracy, 100.00% unknown rejection, and 0.00% unknown false acceptance on that research split; the hierarchical rule reports 56.00% known-identity accuracy with 100.00% unknown rejection. These are dataset-split measurements, not a guarantee for newly enrolled people.

Fingerprint SVM evaluation reports 99.11% test accuracy, 98.65% F1, 0.00% FAR, and 2.67% FRR at the saved validation threshold. The isolated command comparison reports 66.94% test accuracy / 66.87% macro-F1 for SVM and 60.28% / 59.55% for Random Forest. The command results explain why the game still requires a confidence gate and performs no action for uncertain or unsupported words.

## Quick start on Windows

From PowerShell:

```powershell
git clone https://github.com/abdelrahman-mohamed-ai/BioGuard-Secret-Door.git
cd BioGuard-Secret-Door
Set-ExecutionPolicy -Scope Process Bypass
.\setup_venv.ps1
.\START_GAME.bat
```

The setup script creates `.venv`, installs `requirements.txt`, and registers a Jupyter kernel. `START_GAME.bat` then launches Flask on `http://127.0.0.1:5000` using the existing virtual environment.

For a manual setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python flask_game\app.py
```

The first browser launch follows this flow:

1. Open the admin setup page and create a local Admin PIN.
2. Log in to the Control Panel.
3. Add the first user with four face samples and four voice samples.
4. Save the enrollment and confirm that the user appears in the local gallery.
5. Choose **Enter Facility**, complete face verification, then voice verification.
6. Use Voice Command Mode only after the speaker check succeeds; `GO` opens and `STOP` closes when permission is granted.
7. Use Fingerprint Lab for upload-only academic verification.

No name or identity is hardcoded into the public game. If a clone has no local `flask_game/private_data/`, it should show first-run setup and an empty user gallery.

## Dataset preparation

Raw RAVDESS, SOCOFing, and Mini Speech Commands data are not distributed here. See [DATASETS.md](DATASETS.md) for official sources, license notes, expected folders, and preparation commands.

The checked-in `.joblib` bundles and evaluation CSVs allow the application and the dataset-free smoke checks to run without copying raw biometric media into Git.

## Remote classroom demo

For a temporary local tunnel, after starting Flask:

```powershell
cloudflared tunnel --url http://127.0.0.1:5000
```

The tunnel URL is temporary. Review authentication, consent, browser permissions, rate limits, and privacy before sharing it with anyone. Never expose a local private enrollment gallery or admin configuration through a public tunnel.

## Tests

Run the public snapshot checks:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q tests\test_public_snapshot.py
python -m pytest -q flask_game\tests\test_security.py -k "factory_starts_in_first_run_setup_without_secrets or first_run_setup_persists_hashed_admin_secret"
python -m compileall -q .
python -m pip check
```

Tests that require raw research datasets or private enrollment media remain available in the original working project, but are not part of the public CI path. The public workflow never needs a real user, a PIN, a webcam, a microphone, or an inaccessible dataset.

## Academic limitations

- Traditional ML only; this project does not claim deep-learning or production biometric performance.
- Public-dataset accuracy is not real-world accuracy.
- RAVDESS and SOCOFing are acted/research data with licensing and population limitations.
- No unsupported anti-spoofing or liveness guarantee is made.
- The local gallery needs independent samples to calibrate confidence for a real user population.
- Fingerprint verification is offline/upload-only and not a door requirement.
- Camera and microphone operation depends on browser permissions, device quality, codecs, lighting, and recording conditions.
- A public repository cannot contain real enrolled users; live enrollment must be performed locally.
- This is an academic prototype, not a production security system.

## License and attribution

No project-code license file is included in this snapshot because the author’s preferred code license and ownership grant were not independently established during packaging. Dataset licenses are separate and are summarized in [DATASETS.md](DATASETS.md). Do not assume that a dataset license grants commercial rights to the source recordings or to every downstream use.

## Further documentation

- [SETUP.md](SETUP.md) — installation and live demonstration checklist.
- [DATASETS.md](DATASETS.md) — dataset sources, terms, and preparation.
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — repository map.
- [MODEL_CARD.md](MODEL_CARD.md) — artifact inventory, provenance, and limitations.
- [SECURITY.md](SECURITY.md) — privacy boundary and issue-reporting guidance.
- [PUBLICATION_INVENTORY.md](PUBLICATION_INVENTORY.md) — what was included and excluded from the public snapshot.
- [CONTRIBUTING.md](CONTRIBUTING.md) — safe contribution workflow.
