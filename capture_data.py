"""Collect consented webcam + microphone samples locally on Windows/macOS/Linux.
Examples:
  python capture_data.py paired --user ahmed --split train --count 15
  python capture_data.py command --user ahmed --split train --word open --count 10
Note: use SEPARATE recording sessions/days for train/val/test.
"""
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from bioguard_core import SR, DURATION, SPLITS


def capture_face():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Camera not available; close other camera apps and check Windows privacy")
    print("Camera: press SPACE to capture, Q or Esc to cancel")
    frame_saved = None
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                raise RuntimeError("Failed to read camera frame")
            preview = frame.copy()
            cv2.putText(preview, "SPACE: capture | Q: cancel", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            cv2.imshow("BioGuard - Face Capture", preview)
            key = cv2.waitKey(10) & 0xFF
            if key == 32:
                frame_saved = frame.copy()
                break
            if key in (27, ord('q')):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
    if frame_saved is None:
        raise KeyboardInterrupt("Capture cancelled")
    return frame_saved


def record_voice(seconds=DURATION):
    input(f"Press Enter, then say 'BioGuard access' for {seconds} seconds... ")
    print("Recording...")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    if float(np.sqrt(np.mean(audio**2))) < 0.002:
        raise ValueError("Recording too quiet. Check mic selection/permission and retry.")
    return audio[:, 0]


def next_index(folder):
    used = {p.stem for p in folder.glob("*") if p.is_file()}
    n = 1
    while f"{n:03d}" in used:
        n += 1
    return f"{n:03d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["paired", "command"])
    parser.add_argument("--user", required=True, help="User identifier, e.g. ahmed")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--word", choices=["open", "lock", "status", "help"])
    parser.add_argument("--fingerprint", help="Optional path to genuine fingerprint scan to attach to ONE paired sample")
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    if not args.user.replace('_', '').replace('-', '').isalnum():
        parser.error("--user should contain only letters/numbers/_/-")
    if args.count < 1:
        parser.error("--count must be positive")
    if args.action == "command" and (not args.word or args.split == "enroll"):
        parser.error("For command: --word is required and split must be train/val/test")
    base = Path(args.root)/args.user
    if args.action == "paired":
        face_dir = base/args.split/"face"
        voice_dir = base/args.split/"voice"
        face_dir.mkdir(parents=True, exist_ok=True)
        voice_dir.mkdir(parents=True, exist_ok=True)
        if args.fingerprint and args.count > 1:
            parser.error("Provide individual fingerprint scans per sample; --count must be 1 with --fingerprint")
        for i in range(args.count):
            print(f"\nPair {i+1}/{args.count} - {args.user} {args.split}")
            sample = next_index(face_dir)
            face = capture_face()
            audio = record_voice()
            photo_path = face_dir/(sample+".jpg")
            voice_path = voice_dir/(sample+".wav")
            if not cv2.imwrite(str(photo_path), face):
                raise IOError(f"Could not save {photo_path}")
            sf.write(str(voice_path), audio, SR)
            print("Saved:", photo_path, voice_path)
            if args.fingerprint:
                input("Consent required. Press Enter to attach the fingerprint scan to this sample: ")
                target = base/args.split/"fingerprint"/(sample + ".png")
                target.parent.mkdir(parents=True, exist_ok=True)
                img = cv2.imread(args.fingerprint, cv2.IMREAD_GRAYSCALE)
                if img is None or not cv2.imwrite(str(target), img):
                    raise ValueError("Invalid fingerprint image or save failed")
                print("Fingerprint saved:", target)
    else:
        target_dir = base/"commands"/args.split/args.word
        target_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.count):
            sample = next_index(target_dir)
            input(f"Press Enter then clearly speak '{args.word}' ... ")
            print("Recording command...")
            data = sd.rec(DURATION*SR, samplerate=SR, channels=1, dtype="float32")
            sd.wait()
            sf.write(str(target_dir/(sample+".wav")), data[:, 0], SR)
            print("Saved:", target_dir/(sample+".wav"))

if __name__ == "__main__":
    main()
