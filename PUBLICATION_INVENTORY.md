# Publication inventory

This file records the clean publication boundary used for the initial GitHub snapshot.

## Included

- Python research and Flask application source.
- HTML templates, CSS, browser JavaScript, isolated Flask tests, and local matching configuration.
- The three requested notebooks, excluding the old timestamped backup notebook.
- The three reviewed `model/*.joblib` artifacts, `model_manifest.json`, and public evaluation CSVs.
- Public research plots under `outputs/`.
- Windows startup/setup scripts and the new documentation/CI files.

## Excluded deliberately

- `flask_game/private_data/`, including users, face images, voice recordings, enrollment templates, registry, admin configuration, audit logs, session secret, runtime files, staging files, and backups.
- `tmp_debug_enroll/` and all temporary/debug media.
- `flask_game/backups/` and the prior notebook backup.
- Raw `public_data/`, `command_data/`, `fingerprint_data/`, `FINGERPRINT_DEMO/`, and RAVDESS archives.
- `flask_game/USER_STORAGE_CONSISTENCY_REPORT.md`, because it contains local enrollment identifiers and storage details.
- The private-data-dependent `flask_game/tests/test_stop_command_real.py`, because the public snapshot cannot include its local user/media fixtures.
- Virtual environments, caches, logs, secrets, local databases, and temporary files.

## Review notes

- The model binaries were loaded locally and inspected as structured bundles. Their key names, model types, dimensions, public actor/fingerprint labels, and sizes were recorded without exporting raw arrays or private media.
- A byte-pattern scan found no local enrollment identifiers, private enrollment paths, admin secret files, or private-key markers in the reviewed model binaries.
- Notebook outputs containing the original workstation root were sanitized in the staging copy; source cells and research results were preserved.
- The repository has no inherited Git history because the original project directory was not a Git repository. The public snapshot is initialized with fresh history in this sibling staging directory.
- Dataset licenses and attribution requirements remain separate from the project code and model artifacts. See [DATASETS.md](DATASETS.md) and [MODEL_CARD.md](MODEL_CARD.md).
