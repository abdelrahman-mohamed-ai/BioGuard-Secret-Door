# Dataset setup and licensing

The repository intentionally excludes raw biometric datasets. Download them from the official or author-provided source, review the current terms, and keep the files local. Do not upload the extracted media, private recordings, or local enrollment data to this repository.

## RAVDESS

- Official record: [Zenodo 10.5281/zenodo.1188976](https://zenodo.org/records/1188976)
- Project information: [RAVDESS at the SMART Lab](https://smartlaboratory.org/resources/speech-song-database-ravdess/)
- License: CC BY-NC-SA 4.0 for the research dataset; commercial licensing is handled separately by the rights holders.
- Use in BioGuard: public face, voice, video, fusion, open-set, and hierarchical research experiments.
- Required source archives for the current preparation script: `Video_Speech_Actor_01.zip` through `Video_Speech_Actor_07.zip`.

Place the seven archives in `raw_ravdess/` or in the user Downloads folder, then run:

```powershell
python prepare_ravdess.py --source raw_ravdess --output public_data --max-clips 28
```

The script extracts the full audio-visual speech modality, aligns face frames, extracts audio, and writes a local manifest/feature cache. Actors 01–05 are used as known research identities, actor 06 as validation unknown, and actor 07 as final unknown where the selected experiment supports that split. Do not confuse these public actors with BioGuard’s local enrolled employees.

## SOCOFing

- Author record: [Coventry University SOCOFing publication](https://pureportal.coventry.ac.uk/en/publications/sokoto-coventry-fingerprint-dataset/)
- Paper: [Sokoto Coventry Fingerprint Dataset](https://arxiv.org/abs/1807.10609)
- Access source referenced by the authors: [Kaggle SOCOFing](https://www.kaggle.com/datasets/ruizgara/socofing)
- Terms: intended for non-commercial academic research; verify the current provider terms before use or redistribution.
- Use in BioGuard: offline fingerprint preprocessing and subject-held-out research evaluation.

Place the downloaded archive in Downloads or a local `fingerprint_data/`/`datasets/` folder, then use the preparation functions in `fingerprint_research.py`. The expected source contains `Real/` and `Altered-*` image folders. The generated `fingerprint_data/` directory is ignored by Git.

The fingerprint artifact documents an important limit: synthetic altered probes are not independent new scanner captures. A high score on that benchmark is not proof of production fingerprint security.

## Mini Speech Commands

- Dataset announcement: [Google Research Speech Commands](https://research.google/blog/launching-the-speech-commands-dataset/)
- Archive used by the project: [mini_speech_commands.zip](https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip)
- License: the Speech Commands dataset is released under CC BY 4.0; follow attribution requirements for any local use.
- Use in BioGuard: isolated-word command recognition, not speaker identity.
- Supported words in the checked-in command artifact: `down`, `go`, `left`, `no`, `right`, `stop`, `up`, `yes`.

The preparation code keeps source speakers within one split and writes a local `command_data/mini_selected/` directory. That directory is ignored by Git. The model maps only `go` to `open` and `stop` to `close`; all other words produce no protected action.

## Reproducibility and attribution

Keep a local record of the dataset version, download date, source URL, and any checksum used for an experiment. Cite the dataset paper and provider in academic work. The evaluation CSVs and plots checked into this repository are derived summaries; they are not a substitute for the source datasets or their terms.
