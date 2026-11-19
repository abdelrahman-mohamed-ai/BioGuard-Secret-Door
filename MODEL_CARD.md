# Model card and artifact inventory

## Scope

These artifacts are traditional-ML research/deployment bundles. They support reproducible local experiments and an academic Flask demonstration. They do not provide a production biometric security guarantee.

## `model/public_models.joblib`

- Size: approximately 7.1 MB.
- Bundle version: `2.0`.
- Models: Image Only, Voice Only, Video Only, Early Image+Voice, Early Video+Voice, Early Image+Video, and Early All Three.
- Model families: the saved public experiments use SVM and Random Forest pipelines as described in the evaluation CSVs.
- Public template identities: five public RAVDESS actor labels (`actor_01` through `actor_05`). These are research-data identities, not BioGuard employees.
- Features: HOG/LBP image features, MFCC/ZCR/RMS voice features, and multi-frame video aggregates.
- Policy metadata records that the public research gallery is not the public test gallery and that deployment requires a face/voice gate.

## `model/fingerprint_model.joblib`

- Size: approximately 18.6 KB.
- Selected model: SVM.
- Feature vector: 29 dimensions for the stored fingerprint pair representation.
- Threshold: `0.8077645335329597`, selected by the existing validation procedure.
- Demo gallery: 25 public SOCOFing fingerprint labels/templates.
- Method: original fingerprint versus synthetic altered probe with subject-held-out splitting.
- Limitation: a synthetic altered probe is not an independent scanner capture; the artifact does not establish real-time fingerprint security.

## `model/commands_model.joblib`

- Size: approximately 1.0 MB.
- Selected model: SVM.
- Feature vector: 82 dimensions from MFCC, delta, delta2, ZCR, and RMS mean/std statistics on one-second isolated words.
- Words: `down`, `go`, `left`, `no`, `right`, `stop`, `up`, `yes`.
- Saved confidence threshold: `0.55`.
- Actions: only `go → open` and `stop → close`.
- Security rule: speaker verification and permission checks are separate gates; a command prediction alone never authorizes an action.

## Public versus local gallery

The public artifacts are frozen research bundles. The Flask game starts with an empty local gallery in a clean clone. Admin enrollment stores individual face and voice templates below ignored local storage and the Flask code uses the shared `bioguard_matching.py` module for local matching. The versioned local calibration file is explicitly provisional and requires local-gallery validation for a real population.

## Evaluation interpretation

The CSV reports and notebook figures use public dataset splits. They include validation thresholds, test metrics, open-set results, FAR/FRR, and confusion matrices. They must not be read as live microphone/camera accuracy, identity proof for a new employee, or resistance to replay/spoofing.

## Provenance and safe use

The RAVDESS-derived material is subject to the dataset’s CC BY-NC-SA terms, and SOCOFing is intended for non-commercial academic research. Review [DATASETS.md](DATASETS.md) and the upstream terms before redistribution or commercial use. Load these joblib files only from trusted sources.
