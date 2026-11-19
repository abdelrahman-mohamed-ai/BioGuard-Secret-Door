# BioGuard Voice Path Report

## Observed failure

The existing private audit log shows successful face-candidate events followed by door denials with `best_local_voice_score_is_below_the_calibrated_threshold`. That is a genuine `VOICE_LOW_CONFIDENCE` decision after audio processing; it is not evidence that the browser failed to record. The previous endpoint/UI path did not expose a stable audio error taxonomy, so a post-processing failure could appear as a generic recording failure.

## Threshold sources and runtime selection

- `0.999384722048567` comes from `flask_game/local_matching_calibration.json`. It was calibrated from public validation genuine/impostor trials (`target FAR = 0.05`) for the selected `cosine_nn` matcher. **This is the active Flask threshold while the current gallery has one identity.**
- `0.998895976930414` is produced only when Flask sees at least two identities with individual templates. `bioguard_matching.calibration_for(...)` recalibrates the same `cosine_nn` method from private leave-one-out genuine and cross-user pairs. It is a provisional local cross-user threshold, not a replacement for the one-user runtime configuration.

The public research artifact thresholds (`0.297165` face and `0.533590` voice) are not used as the local voice gate.

## Corrected path

`MediaRecorder` now emits timed chunks, preserves a MIME-appropriate extension, rejects empty blobs, and attaches a stable browser error code. Flask stores the upload in a private temporary directory, converts it with the existing local FFmpeg binary to mono 16 kHz WAV, checks duration, rejects silence, runs the shared MFCC + delta + delta2 + ZCR + RMS extractor, verifies the speaker against the server-bound face candidate, and returns the result code to the browser.

The face matcher and all biometric thresholds were left unchanged.

## Current private-template diagnostic

The existing private gallery now contains four face vectors and four voice vectors. A read-only leave-one-out diagnostic using those saved vectors produced:

- Face: 2/4 above the unchanged `0.8924208` local threshold; final read-only audit of the current four stored templates produced score range `0.8167046–0.9373321`.
- Voice: 0/4 above the unchanged `0.9993847` local threshold; final read-only audit of the current four stored templates produced score range `0.9966691–0.9978466`. The earlier `0.9946398–0.9973075` range was from the prior enrollment state.

This confirms that the current user’s post-recording denial is `VOICE_LOW_CONFIDENCE` under the unchanged calibration, not `AUDIO_DECODE_FAILED`. It is enrollment-only evidence, not a held-out microphone accuracy result, and the threshold was not lowered or bypassed.

## Real saved-WAV endpoint result

The automated endpoint test used a real saved WAV from the existing public evaluation data and a server-bound accepted-face test session. It did not mock audio conversion, feature extraction, or speaker matching.

| Field | Result |
|---|---:|
| HTTP status | `200` |
| Decision | `accepted: true` |
| Voice score | `0.9989560145` |
| Voice threshold | `0.9988959769` |
| Error code | none |

The test does not claim that the current private employee’s microphone sample has the same score; the private media directory was not readable during this run.

## Real failure-code checks

The endpoint was also exercised with actual generated WAV/invalid inputs:

- Short WAV → HTTP `200`, `AUDIO_TOO_SHORT`
- Three-second silent WAV → HTTP `200`, `AUDIO_SILENT`
- Invalid WebM bytes → HTTP `200`, `AUDIO_DECODE_FAILED`
- Empty WAV upload → HTTP `200`, `AUDIO_UPLOAD_FAILED`
- Controlled unexpected server exception → HTTP `500`, `SERVER_ERROR`

Valid speaker decisions remain separate:

- `VOICE_LOW_CONFIDENCE` — processed audio score below threshold
- `VOICE_MISMATCH` — processed audio belongs to another speaker
- `PERMISSION_DENIED` — biometrics passed but door permission failed

## Validation command

```powershell
& .\.venv\Scripts\python.exe -m pytest -q .\flask_game\tests\test_voice_endpoint_real_audio.py
```

Result: 3 passed. The complete project suite reports 32 passed. Browser hardware behavior still requires a fresh physical microphone test; no production anti-spoofing claim is made.
