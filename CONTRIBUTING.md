# Contributing safely

Contributions should preserve the separation between public research artifacts and private local enrollment data.

## Before opening a pull request

```powershell
python -m compileall -q .
python -m pytest -q tests\test_public_snapshot.py
python -m pip check
git status --short
```

If you have the required local datasets, run the relevant notebook or research test separately and document the exact dataset split and version. Do not add the raw files to Git.

## Pull-request expectations

- Do not retrain or overwrite checked-in artifacts without a documented experiment and provenance review.
- Do not change thresholds or claim improved biometric performance without independent validation evidence.
- Keep face, voice, commands, fingerprint, and enrollment changes scoped and tested.
- Add tests for new Flask routes and fail-closed behavior.
- Keep private data, credentials, logs, backups, and local paths out of source, notebooks, outputs, and commit history.
- Use a trusted source when loading serialized `joblib`/pickle files.

There is no project-code license file in this snapshot. Confirm ownership and choose a license before accepting external contributions or redistributing modified artifacts.
