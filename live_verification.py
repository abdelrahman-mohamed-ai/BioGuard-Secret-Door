"""Local webcam + mic demo using ONLY enrolled users and exported classical verifiers.
Run after training notebook and enrollment: python live_verification.py
Not a hardened production security system.
"""
from __future__ import annotations
import argparse
import tempfile
from pathlib import Path
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import deployment


def capture():
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Webcam unavailable. Check Windows camera permissions.')
    frame=None
    try:
        print('Webcam: SPACE to capture / Q to cancel.')
        while True:
            ok,img=cap.read()
            if not ok:
                raise RuntimeError('Unable to read camera')
            cv2.imshow('BioGuard – Secret Door',img)
            key=cv2.waitKey(25)&255
            if key==32:
                frame=img.copy();break
            if key in (27,ord('q')):
                break
    finally:
        cap.release();cv2.destroyAllWindows()
    if frame is None:raise RuntimeError('Cancelled')
    return frame


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seconds',type=int,default=3)
    a=parser.parse_args()
    model=deployment.load_verifier()
    gallery=deployment.load_gallery()
    if not gallery['templates']:
        print('Please enroll users first: see README / deployment.enroll_user');return
    frame=capture()
    input('Press ENTER to record your voice for 3 seconds ...')
    signal=sd.rec(a.seconds*16000,samplerate=16000,channels=1,dtype='float32')
    sd.wait()
    with tempfile.TemporaryDirectory() as root:
        image=Path(root)/'attempt.png';voice=Path(root)/'attempt.wav'
        if not cv2.imwrite(str(image),frame):raise RuntimeError('Failed to save camera capture')
        sf.write(str(voice),signal[:,0],16000)
        decision=deployment.verify_new_attempt(image,voice,package=model,gallery=gallery)
    print('\nACCESS GRANTED' if decision['accepted'] else '\nACCESS DENIED')
    print(decision)

if __name__=='__main__':main()
