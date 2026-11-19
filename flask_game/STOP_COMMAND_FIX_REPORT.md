# BioGuard STOP Command Fix Report

Generated 2026-11-19. This report uses aggregate command scores and labels only; it does not contain biometric media, secrets, or private enrollment templates.

## 1. Exact reason STOP was rejected

There are two separate causes:

1. The browser command recorder captured exactly 1,000 ms. A user who begins speaking slightly after clicking the button can lose the initial `/s/` or final `/p/` in `stop`. The command model was trained on short isolated words, so this timing difference matters.
2. The unchanged trained SVM itself has real STOP confusions. On the existing held-out STOP set, some samples are classified as `right`, `up`, `go`, `yes`, `down`, or `no`. These are genuine model decisions, not an endpoint or speaker-verification failure.

The server preprocessing was already aligned with the training pipeline:

`decode -> mono 16 kHz -> trim leading/trailing silence at top_db=38 -> reject silence/<0.13 s -> fix to 1 second -> MFCC(13), delta, delta2, ZCR, RMS -> 82 features`.

The trained artifact is an SVM pipeline with 82 input features and a confidence threshold of `0.55`. The artifact and feature extractor were not changed.

## 2. Real STOP diagnostics

| Input | Top prediction | Top-3 predictions | Confidence | Margin | Threshold | Result |
|---|---|---|---:|---:|---:|---|
| Genuine STOP, accepted sample | STOP | STOP 0.9951; DOWN 0.0021; UP 0.0011 | 0.9951 | 0.9930 | 0.55 | `close` |
| Genuine STOP, uncertain sample | DOWN | DOWN 0.2600; GO 0.2362; LEFT 0.1607 | 0.2600 | 0.0238 | 0.55 | `COMMAND_UNCERTAIN` |
| Genuine STOP, confused sample | RIGHT | RIGHT 0.3692; UP 0.2115; STOP 0.1880 | 0.3692 | 0.1577 | 0.55 | `COMMAND_UNCERTAIN` |
| Genuine STOP, confident wrong class | YES | YES 0.8986; STOP 0.0690; RIGHT 0.0229 | 0.8986 | 0.8296 | 0.55 | `COMMAND_NOT_ACTION` |

The final case is important: the server does not force a high-confidence non-action word into STOP.

## 3. Held-out command-model evaluation

The existing trained artifact was evaluated directly on 360 real held-out WAV files from all eight supported classes. All files decoded and produced 82-dimensional features.

### STOP

- Top-1 STOP predictions: `32/45` (`71.11%`).
- Correct STOP predictions above the confidence threshold: `30/45` (`66.67%`).
- Below-threshold/uncertain STOP attempts: `11/45`.
- Top-1 confusions: RIGHT `6`, UP `2`, GO `2`, YES `1`, DOWN `1`, NO `1`.
- Source duration range: `0.9288–1.0000 s`.
- Trimmed speech shorter than `0.6 s`: `11/45`.

### GO regression

- Top-1 GO predictions: `22/45` (`48.89%`).
- Correct GO predictions above the threshold: `13/45`.
- The real Flask endpoint still authorized a genuine GO sample and returned action `open`.

### Overall held-out result

- Accuracy: `0.6750`.
- Macro F1: `0.6730`.
- Confidence threshold: `0.55`.

Confusion matrix, label order `down, go, left, no, right, stop, up, yes`:

```text
[[30, 5, 2, 3, 3, 2, 0, 0],
 [ 4,22, 6, 5, 3, 2, 2, 1],
 [ 1, 3,29, 4, 3, 4, 1, 0],
 [ 4, 7, 2,25, 3, 1, 1, 2],
 [ 0, 0, 1, 4,38, 0, 1, 1],
 [ 1, 2, 0, 1, 6,32, 2, 1],
 [ 1, 1, 5, 0, 2, 7,28, 1],
 [ 2, 0, 0, 0, 4, 0, 0,39]]
```

This evidence shows that STOP is usable but not perfectly recognized by the existing model. Lowering the threshold or hardcoding STOP would increase unsafe false actions and was not done.

## 4. Correction implemented

The command-only browser capture window changed from `1,000 ms` to `1,500 ms`. This gives the user enough time to start speaking and preserve the complete isolated word, including STOP’s initial and final consonants.

The server still applies the exact trained command extractor and the exact artifact threshold. Speaker recording remains 3 seconds and was not changed. Face recognition, speaker verification, permissions, door authentication, enrollment, fingerprint verification, and the trained model artifact were not modified.

The command endpoint now returns the actual command failure category:

- `COMMAND_UNCERTAIN` when confidence is below `0.55`.
- `COMMAND_NOT_ACTION` when a non-action word such as YES is recognized.
- `PERMISSION_DENIED` when the recognized action is not permitted.
- Existing speaker-gate rejection remains separate.

Authenticated admin sessions also receive predicted command, confidence, threshold, margin, top three predictions, feature dimension, and command state. Public responses do not expose private speaker diagnostics.

## 5. Real Flask endpoint results

Using the current gallery, a real enrolled speaker WAV, and real command WAV files:

- STOP sample: HTTP 200, `accepted=true`, `command=stop`, `action=close`.
- GO sample: HTTP 200, `accepted=true`, `command=go`, `action=open`.
- Low-confidence STOP sample: HTTP 200, `error_code=COMMAND_UNCERTAIN`; no action.
- Confident STOP-to-YES confusion: HTTP 200, `error_code=COMMAND_NOT_ACTION`; no action.
- Generated WebM/Opus STOP payload through the production multipart route: HTTP 200, `command=stop`, `action=close`.
- Generated 1.5-second WebM/Opus STOP payload: HTTP 200, `command=stop`, `action=close`.

These are real model and endpoint calls. The WebM files were generated from existing WAVs to test container conversion; they are not fresh microphone recordings.

## 6. Original versus corrected performance

The trained model’s held-out confusion matrix is intentionally unchanged because no model, threshold, or feature extractor was altered. Existing 1-second dataset WAVs cannot measure whether a fresh browser user would have clipped STOP before the capture-window correction.

The measurable correction is in the live path: the browser now provides a 1.5-second command window, while Flask retains the same mono/16 kHz conversion and one-second isolated-word feature contract. A before/after physical microphone accuracy comparison is **NOT YET VERIFIED**.

## 7. Files changed

- `flask_game/static/js/game.js` — command-only recording window changed to 1.5 seconds and command prompt clarified.
- `speech_commands_research.py` — added non-invasive top-three/confidence/margin diagnostics around the existing inference path; no training code or artifact changed.
- `flask_game/app.py` — command-specific failure codes, admin diagnostics, and clearer command error handling only.
- `flask_game/tests/test_stop_command_real.py` — real artifact, real WAV, real endpoint, and capture-window tests.
- `flask_game/STOP_COMMAND_FIX_REPORT.md` — this report.

Backups created before editing:

- `flask_game/backups/stop_command_fix-20261119/game.js.before-stop-fix`
- `flask_game/backups/stop_command_fix-20261119/app.py.before-stop-fix`
- `flask_game/backups/stop_command_fix-20261119/speech_commands_research.py.before-stop-fix`

No notebook, model artifact, dataset, user, permission, speaker-verification threshold, face path, door-authentication path, or fingerprint path was changed.

## 8. Tests passed

- New real STOP command tests: **4 passed**.
- Command/security regression tests: **27 passed**.
- Full project suite excluding private-data traversal: **57 passed, 8 warnings**.
- Python compilation: **passed**.
- `pip check`: **passed — No broken requirements found**.
- Real held-out model evaluation: **360/360 command files decoded; 0 feature extraction errors**.
- Real GO/STOP endpoint tests: **passed**.

Node.js was not installed in the existing environment, so `node --check` could not be run. The command capture source is covered by the real test assertion and should be browser-tested during the live retest.

## 9. Remaining limitations

- The existing classifier is not perfect: STOP top-1 accuracy is `71.11%` on the available held-out set.
- No confidence threshold was lowered and no STOP prediction was hardcoded.
- A fresh physical microphone recording was not available, so the 1.5-second capture correction is not yet proven on the user’s microphone.
- The Mini Speech Commands test set is not a controlled browser/microphone experiment.
- No anti-spoofing or replay resistance is claimed.
- Improving the model’s genuine STOP confusion would require a separately evaluated retraining or calibration study, which was outside this strict fix scope.

## 10. Exact live test

1. Start the unchanged application:

   ```powershell
   cd "<PROJECT_ROOT>\Downloads\BioGuard_Task2_COMPLETE"
   .\START_GAME.bat
   ```

2. Open Voice Command Mode. Keep the camera disabled.
3. Click `Record Speaker Check`, speak the reference phrase, and wait for speaker verification and permissions to pass.
4. Click `Record Command`.
5. Say `STOP` once clearly during the 1.5-second recording window, including the final P sound.
6. Click submit. Expected result: `STOP authorized - close action permitted`, and an already-open door becomes closed.
7. Repeat with `GO`. Expected result: `GO authorized - open action permitted`.
8. If STOP is rejected, an authenticated admin can inspect the real predicted word, confidence, threshold, margin, and top three predictions. Do not treat `COMMAND_UNCERTAIN` or a non-action prediction as a successful STOP.

