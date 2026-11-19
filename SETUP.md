# Windows setup and live demonstration

This guide assumes Windows PowerShell and Python 3.12. The repository contains the reviewed model artifacts, so a dataset download is optional for launching the Flask game. Dataset downloads are required only to reproduce the full notebook experiments.

## 1. Clone and install

```powershell
git clone https://github.com/abdelrahman-mohamed-ai/BioGuard-Secret-Door.git
cd BioGuard-Secret-Door
Set-ExecutionPolicy -Scope Process Bypass
.\setup_venv.ps1
```

The equivalent manual commands are:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Start the game

```powershell
.\START_GAME.bat
```

Open `http://127.0.0.1:5000/`. The Flask process binds to loopback by default. To use a different local port, set `BIOGUARD_PORT` before starting; do not commit environment files containing secrets.

## 3. First-run admin setup

1. Choose the admin setup link shown by the application.
2. Create a local Admin PIN and confirm it.
3. Log in to the Control Panel.
4. Keep the PIN private. It is stored as a hash in the ignored local runtime directory, not in Git.

A clean public clone has no local users. If the browser shows an enrolled person on a clone, stop and inspect the local private directory before continuing.

## 4. Add the first local user

1. In the Control Panel, open **Add New User**.
2. Enter a name, employee identifier, role, and permissions.
3. Capture or upload four face samples. Replace only a failed slot if quality validation rejects it.
4. Record or upload four voice samples. Replace only a failed slot if duration, silence, or decoding validation rejects it.
5. Review the profile and select **Save & Enroll User**.
6. Confirm success only after the server responds that templates were saved.

The user’s images, recordings, templates, registry, audit log, and session secret remain under the local ignored `flask_game/private_data/` directory.

## 5. Test the Secret Door

1. Return to the home screen and choose **Enter Facility**.
2. Select **Start Face Scan** and allow camera access.
3. Wait for the face candidate and the explicit face verification result.
4. Select **Start Voice Verification** and allow microphone access.
5. Speak naturally using a fresh recording. The server checks voice against the same candidate identified by face.
6. The door opens only when face verification, speaker verification, and `open_door` permission all pass.
7. On failure, read the reason (`FACE_UNKNOWN`, `VOICE_MISMATCH`, `VOICE_LOW_CONFIDENCE`, `PERMISSION_DENIED`, or a technical error) and retry; no failure should open the door.

## 6. Add a second user and test mismatch

Add another local user through the same admin flow. Then test:

- user A face + user A voice: accepted only if both real checks pass and permission is present;
- user B face + user B voice: accepted under the same conditions;
- user A face + user B voice: denied as a mismatch;
- an unregistered visitor: denied as unknown;
- a registered user without door permission: denied at the permission stage.

The game never asks the visitor to select or type an identity.

## 7. Voice Command Mode

1. Open Voice Command Mode; the camera is not required.
2. Record the speaker check and wait for identity and permission verification.
3. Record an isolated `go` or `stop` command.
4. `go` opens and `stop` closes only for a verified speaker with the corresponding command permission.
5. Silence, unsupported words, low confidence, and unknown speakers produce no protected action.

## 8. Fingerprint Lab

Fingerprint Lab accepts a local PNG, JPG, JPEG, or BMP upload and runs the existing offline Sobel/LBP model. It displays the source image, preprocessing views, score, threshold, and decision. It is separate from door access and does not require a live scanner.

## 9. Tests and diagnostics

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q tests\test_public_snapshot.py
python -m pytest -q flask_game\tests\test_security.py -k "factory_starts_in_first_run_setup_without_secrets or first_run_setup_persists_hashed_admin_secret"
python -m compileall -q .
python -m pip check
```

Full research tests need the raw datasets. Live camera/microphone behavior also needs a real browser and hardware; a dataset-free CI run cannot prove those paths.

## 10. Optional temporary tunnel

```powershell
cloudflared tunnel --url http://127.0.0.1:5000
```

Treat the URL as public. Use it only for a controlled demonstration after reviewing browser permissions, authentication, privacy, and rate limits.
