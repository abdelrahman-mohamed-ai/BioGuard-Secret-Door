# BioGuard Voice Sensitivity Report

Generated from the existing public validation partition using the shared `bioguard_matching.py` score functions. No raw audio, feature vectors, or local private identifiers are written.

## Preprocessing contract

Enrollment and live verification both call the same adapter: browser/container audio is converted to mono 16 kHz WAV, `librosa.load` trims at `top_db=30`, the signal is fixed to 3 seconds, and `bioguard_core.extract_voice` computes MFCC(13) + delta + delta2 + ZCR + RMS means and standard deviations (82 values).

Selected runtime method: `cosine_nn`. One-identity runtime threshold: `0.999384722048567` from `flask_game/local_matching_calibration.json`; target validation FAR is `0.05`.

## Validation genuine and unknown comparison

| Method | Threshold | Validation genuine pass | Validation unknown rejection | Held-out known ID | Held-out unknown rejection |
|---|---:|---:|---:|---:|---:|
| `cosine_nn` | 0.999385 | 0.4800 | 0.6429 | 0.4800 | 0.7500 |
| `cosine_mean` | 0.999003 | 0.2800 | 0.8571 | 0.2000 | 0.7143 |
| `cosine_knn3` | 0.998874 | 0.2800 | 0.8929 | 0.2000 | 0.6071 |
| `zcos_nn` | 0.562389 | 0.2800 | 0.6786 | 0.1600 | 0.9286 |
| `zcos_knn3` | 0.408122 | 0.2800 | 0.9286 | 0.2000 | 0.9643 |
| `zdist_nn` | 0.528939 | 0.2400 | 0.8571 | 0.1600 | 0.8929 |
| `zdist_knn3` | 0.516453 | 0.1600 | 0.9643 | 0.0800 | 0.9643 |
| `voice_shape_cos_nn` | 0.999951 | 0.3600 | 0.7143 | 0.4400 | 0.8929 |

`Actor 06` is the validation unknown partition; `Actor 07` is held out for final reporting. The selected `cosine_nn` is the validation harmonic-score winner, not a threshold tuned on the final test set.

## Selected genuine sensitivity

The selected validation set contains 25 genuine clips. Public RAVDESS statement IDs are used as a word/utterance proxy; they are not local user phrases.

| statement | n | mean score | min | max | pass rate |
|---|---:|---:|---:|---:|---:|
| 1 | 12 | 0.998254 | 0.990836 | 0.999598 | 0.3333 |
| 2 | 13 | 0.998548 | 0.990196 | 0.999843 | 0.6154 |

| repetition | n | mean score | min | max | pass rate |
|---|---:|---:|---:|---:|---:|
| 1 | 17 | 0.998090 | 0.990196 | 0.999843 | 0.4706 |
| 2 | 8 | 0.999078 | 0.997696 | 0.999645 | 0.5000 |

### Volume proxy

The source dataset has no microphone-condition label. RMS is therefore only a recorded-volume proxy; it is not a controlled microphone experiment.

| volume_bin | n | mean score | min | max | pass rate |
|---|---:|---:|---:|---:|---:|
| high | 8 | 0.996837 | 0.990196 | 0.999511 | 0.1250 |
| low | 9 | 0.999236 | 0.997640 | 0.999843 | 0.6667 |
| mid | 8 | 0.999042 | 0.997313 | 0.999645 | 0.6250 |

Raw source duration range before the shared 3-second fix-length step: `3.1031`–`4.4044` seconds. RMS range: `0.001295`–`0.108449`.

### Actor/capture proxy

Actor-level differences show capture/identity variability in the public data, but cannot identify a microphone effect.

| capture_cohort | n | mean score | min | max | pass rate |
|---|---:|---:|---:|---:|---:|
| capture_cohort_1 | 5 | 0.999317 | 0.998736 | 0.999843 | 0.6000 |
| capture_cohort_2 | 5 | 0.999524 | 0.999422 | 0.999645 | 1.0000 |
| capture_cohort_3 | 5 | 0.998864 | 0.997696 | 0.999477 | 0.2000 |
| capture_cohort_4 | 5 | 0.999254 | 0.998799 | 0.999609 | 0.4000 |
| capture_cohort_5 | 5 | 0.995074 | 0.990196 | 0.999385 | 0.2000 |

## Deployment conclusion

The current selected matcher is sensitive to recorded volume and utterance variation. The one-user private gallery has no independent local impostor population, so the current threshold remains provisional and uncertain local attempts stay denied. Additional independent local speakers, held-out phrases, and controlled microphone/volume captures are required before recalibrating a deployment threshold.
