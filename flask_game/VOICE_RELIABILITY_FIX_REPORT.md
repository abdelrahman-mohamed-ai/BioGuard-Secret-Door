# BioGuard Voice Reliability Fix Report

Generated 2026-11-19 from the existing project. This report contains aggregate biometric scores only; it does not contain raw audio, feature vectors, secrets, or private media.

## 1. Exact root cause

The recording and feature-extraction path was internally consistent. Flask converted uploaded audio with FFmpeg to mono 16 kHz WAV, and both enrollment and verification called the same `bioguard_matching.extract_voice()` adapter and the same `bioguard_core.extract_voice()` implementation:

`decode -> trim silence (top_db=30) -> fix to 3 seconds -> MFCC(13) + delta + delta2 + ZCR + RMS -> 82 features`.

The failure was the voice gate configuration:

1. The exported voice configuration selected raw `cosine_nn` with threshold `0.999384722048567`, calibrated on the public validation/unknown partition.
2. `calibration_for()` silently replaced that configuration when two or more private identities existed. It recalibrated raw cosine scores from private leave-one-out and cross-user enrollment pairs.
3. That made the runtime threshold depend on the current gallery: `0.998895976930414` for the earlier two-identity gallery and `0.9978263307130621` for the current three-identity gallery.
4. Those private enrollment pairs are useful diagnostics, but they are not independent validation data. The runtime was therefore both unstable and overly dependent on a raw-cosine scale whose genuine and impostor scores overlap.

The current gallery contains three identities with four individual voice templates each. The raw-cosine stored-template leave-one-out genuine scores were:

`min 0.991953, median 0.998101, max 0.998365`.

At the original public threshold, `0/12` stored leave-one-out genuine trials passed. At the old dynamically recalculated threshold, only `8/12` passed before the rank/ambiguity decision. The cross-user raw-cosine impostor scores reached `0.997577`, so lowering the raw threshold would not be a defensible fix.

## 2. Audio and identity-path audit

The two readable current-user media directories contained four WAV files each. They were mono, 16 kHz, approximately 2.995–3.015 seconds long, and all recomputed to the expected 82-dimensional feature shape. Recomputed vectors matched the stored enrollment vectors to floating-point precision. This rules out an enrollment-template serialization or extractor mismatch for those files.

The saved-WAV Flask endpoint test used the actual route, CSRF session, multipart field `voice_recording`, FFmpeg conversion, decoder, shared extractor, current gallery, and JSON response. It returned HTTP 200 with valid JSON.

The audit log intentionally stores only event/reason codes, not biometric scores. No fresh browser microphone file was retained, so the exact score of the user’s previous live recording cannot be recovered from the log. The saved-WAV endpoint and current-gallery diagnostics are the reproducible proxy; a new physical microphone attempt is still required for live confirmation.

## 3. Candidate matching comparison

The comparison used the existing MFCC/ZCR/RMS features and the shared `bioguard_matching.py` implementation. No model was retrained.

| Method | Validation threshold | Validation FAR | Validation FRR | Held-out unknown rejection | Current stored LOO genuine at public threshold |
|---|---:|---:|---:|---:|---:|
| `cosine_nn` | 0.999385 | 0.0500 | 0.5200 | 0.7500 | 0/12 |
| `cosine_mean` | 0.999003 | 0.0500 | 0.6800 | 0.7143 | 0/12 |
| `cosine_knn3` | 0.998874 | 0.0500 | 0.6400 | 0.6071 | 0/12 |
| `zcos_mean` | 0.390889 | 0.0417 | 0.6000 | 0.8571 | 11/12 |
| `zdist_mean` | 0.564257 | 0.0125 | 0.6800 | 1.0000 | 1/12 |
| `voice_shape_cos_mean` | 0.999931 | 0.0417 | 0.7200 | 0.8929 | 0/12 |

The selected deployment method is `zcos_mean`: standardized feature cosine similarity against the mean of the candidate’s individual templates. Its numeric threshold is not comparable to the old `0.999...` raw-cosine threshold; it is a different, versioned score scale. It was selected because it reduced validation false acceptance and improved current local leave-one-out stability. It does not make the public held-out genuine rate production-ready.

The selected public validation configuration is:

- Method: `zcos_mean`
- Threshold: `0.3908886485400538`
- Ambiguity margin: `0.05`
- Validation claim FAR: `0.041666666666666664`
- Validation claim FRR: `0.6`
- Validation unknown rejection: `0.8214285714285714`
- Held-out known identification rate: `0.28`
- Held-out unknown rejection: `0.8571428571428571`

The threshold was not tuned on the final held-out partition. The public data is a research reference, not a local employee population.

## 4. Correction implemented

The voice runtime now uses the versioned voice configuration from `flask_game/local_matching_calibration.json` verbatim. Private leave-one-out/cross-user calibration remains available for diagnostics but no longer silently replaces the deployed voice threshold. The existing face recalibration behavior was left unchanged.

The same shared extraction and scoring functions are used by the door voice stage and the voice-command speaker stage. No separate Flask-only speaker algorithm was introduced.

Admin diagnostics now distinguish the score for the face-selected voice claim from the strongest score anywhere in the gallery. A cross-user rejection therefore no longer displays the other speaker’s high score as if it were the selected candidate’s score.

## 5. Current-gallery real endpoint results

Using the actual current gallery and saved WAV files, without changing the gallery:

| Scenario | HTTP | Decision | Candidate score | Threshold | Result code |
|---|---:|---|---:|---:|---|
| Current identity A face + A voice | 200 | accepted | 0.595389 | 0.390889 | — |
| Current identity B face + B voice | 200 | accepted | 0.776446 | 0.390889 | — |
| A face + B voice | 200 | denied | -0.101211 | 0.390889 | `VOICE_MISMATCH` |
| B face + A voice | 200 | denied | 0.368831 | 0.390889 | `VOICE_MISMATCH` |

The stored-template leave-one-out diagnostic accepted `10/12` genuine trials after the selected ambiguity check and accepted `0/24` cross-user claims. This is an enrollment diagnostic, not independent accuracy evidence.

## 6. Command-mode regression

Using real command WAV files and a current enrolled speaker sample:

- GO: HTTP 200, authorized action `open`.
- STOP: HTTP 200, authorized action `close`.
- Public unknown-speaker sample + GO: HTTP 200, denied at `speaker_command_gate`.

The command classifier, command artifact, permissions, and actions were not modified.

## 7. Files changed

- `bioguard_matching.py` — voice calibration now honors the exported versioned configuration; face behavior is unchanged.
- `flask_game/local_matching_calibration.json` — voice deployment method/configuration changed to the validated standardized-cosine configuration.
- `flask_game/app.py` — admin diagnostics report the face-bound voice claim score and claim margin.
- `flask_game/tools/benchmark_local_matching.py` — rerunning the benchmark no longer silently overwrites an explicitly selected deployment voice configuration.
- `flask_game/tools/voice_sensitivity_audit.py` — sensitivity reports read the deployed voice configuration instead of assuming the comparison winner.
- `flask_game/tests/test_real_model_matching.py` — real voice calibration expectations and enrollment-template smoke-test wording updated.
- `flask_game/VOICE_RELIABILITY_FIX_REPORT.md` — this report.

Backup created before the configuration change:

`flask_game/backups/voice_reliability-20261119/local_matching_calibration.before-voice-fix.json`

No face thresholds, fingerprint code, command model, door animation, enrollment UI, notebooks, datasets, or model artifacts were changed.

## 8. Tests passed

- Real matching and saved-WAV endpoint tests: **10 passed**.
- Full project suite excluding the private-data directory: **53 passed, 8 warnings**.
- Python compilation: **passed**.
- `pip check`: **passed — No broken requirements found**.
- Current-gallery real API cases: **2 same-user accepts and 2 cross-user `VOICE_MISMATCH` denials**.
- Real GO/STOP command regression and unknown-speaker denial: **passed**.

The full suite warnings are existing dependency/notebook validation warnings; no test failure was observed.

## 9. What remains unverified

This is not a production speaker-recognition claim.

- A fresh held-out local recording from each enrolled user has not been collected.
- The current gallery’s local media are enrollment samples; leave-one-out results cannot establish unknown-person FAR.
- Public held-out genuine acceptance for the selected method was only `7/25` at the raw identification measurement used by the benchmark, and public held-out unknown rejection was `24/28`. This is evidence of remaining variability, not a success claim.
- A generated WebM/Opus container derived from a current WAV was accepted through the same upload/conversion route, confirming the browser-container decoder path. A fresh physical microphone recording was not available, so microphone-condition behavior remains unverified.
- No replay, presentation-attack, or anti-spoofing evaluation was performed.

Answers:

- Does the saved user pass a valid independent held-out local evaluation? **NOT YET VERIFIED.**
- Can this gallery reliably distinguish enrolled users from unknown people? **NOT YET VERIFIED.**
- Are mismatched face/voice identities rejected? **YES for the tested current-gallery cross-user endpoint cases; independent local generalization remains NOT YET VERIFIED.**

## 10. Exact live retest

1. Stop any old Flask process and start the project from PowerShell:

   ```powershell
   cd "C:\Users\Abdelrahman Mohamed\Downloads\BioGuard_Task2_COMPLETE"
   .\START_GAME.bat
   ```

2. Open the game, start the face scan, and wait for `FACE VERIFIED`.
3. Start voice verification and speak naturally for the full recording window, close to the microphone.
4. In the admin diagnostics panel, confirm that the voice method is `zcos_mean`, the threshold is `0.3908886485400538`, and the displayed voice score is the score for the same face-selected candidate.
5. Repeat with a fresh recording, not an enrollment file. Record the outcome and score; do not treat a saved enrollment recording as live proof.
6. Test a second enrolled speaker and the two cross-user combinations. The expected result for either mismatch is `VOICE_MISMATCH` and a locked door.
7. If a genuine fresh recording still fails, collect at least five new independent voice clips per enrolled user, including the intended live phrase and the actual microphone, plus unknown-speaker clips. Keep final test clips separate from threshold calibration.

## 11. Notebook boundary

No Jupyter notebook was modified in this voice-only fix. Flask now imports the shared matching implementation and the versioned configuration directly. If the master notebook must describe the newly selected `zcos_mean` deployment configuration, that is a separate documentation-only update; its existing research outputs and experiments were preserved here.

