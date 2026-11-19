"""Prepare matched image, speech and multi-frame video from 7 RAVDESS ZIPs.
Read archives directly from Downloads, without unpacking all 7 ZIP files.
Only full AV modality (01), speech (01); do NOT mix video-only duplicates.
Original dataset: https://zenodo.org/records/1188976 (CC BY-NC-SA 4.0).
"""
from __future__ import annotations
import argparse
import json
import random
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

PATTERN = re.compile(r'^(01)-(01)-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.mp4$', re.I)
KNOWN = (1, 2, 3, 4, 5)
UNKNOWN_VAL = (6,)
UNKNOWN_TEST = (7,)


def locate_archives(source=None):
    """Find Video_Speech_Actor_01..07.zip in project/raw_ravdess or Downloads."""
    roots = []
    if source:
        roots.append(Path(source).expanduser())
    roots += [Path('raw_ravdess'), Path.home() / 'Downloads', Path.home() / 'downloads', Path.cwd()]
    found = {}
    missing = []
    for actor in range(1, 8):
        name = f'Video_Speech_Actor_{actor:02d}.zip'
        path = next((root / name for root in roots if (root / name).is_file()), None)
        if path is None:
            missing.append(name)
        else:
            found[actor] = path.resolve()
    if missing:
        raise FileNotFoundError('Missing: ' + ', '.join(missing) + '\nLooked in: ' + ', '.join(str(x) for x in roots))
    return found


def extract_frames(mp4, desired=7):
    cap = cv2.VideoCapture(str(mp4))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if not cap.isOpened() or cascade.empty():
        raise RuntimeError('MP4/Haar cascade cannot be opened (OpenCV).')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 10:
        cap.release()
        raise RuntimeError('Video has too few frames.')
    faces_out = []
    try:
        for at in np.linspace(int(total * .13), int(total * .85), desired * 5, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(at))
            valid, frame = cap.read()
            if not valid:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            regions = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(45,45))
            if len(regions) == 0:
                continue
            x,y,w,h = max(regions, key=lambda a: int(a[2])*int(a[3]))
            crop = cv2.resize(gray[y:y+h, x:x+w], (96,96), interpolation=cv2.INTER_AREA)
            faces_out.append(cv2.equalizeHist(crop))
            if len(faces_out) == desired:
                break
    finally:
        cap.release()
    if len(faces_out) < 5:
        raise RuntimeError('Fewer than five usable face frames; check video/codec/lighting')
    return faces_out[0], np.stack(faces_out[1:]).astype(np.uint8)


def extract_audio(mp4, wav):
    """Audio from THE SAME audio-video MP4; no separate audio dataset needed."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
           '-i', str(mp4), '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(wav)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not wav.exists() or wav.stat().st_size < 5000:
        raise RuntimeError('FFmpeg audio extraction failed: ' + proc.stderr[-1200:])


def split_members(members, actor, seed=42, limit=28):
    """Group emotion+statement together to reduce repeated-utterance leakage.
    Exhaustive group split then take a balanced quota from each split.
    """
    if actor in (6,7):
        candidates=list(members)
        random.Random(seed+actor).shuffle(candidates)
        label='val_unknown' if actor==6 else 'test_unknown'
        return [(x,label) for x in candidates[:limit]]
    groups = {}
    for member in members:
        bits = PATTERN.match(Path(member).name)
        if bits:
            key = (bits.group(3), bits.group(5))  # emotion + statement; both repetitions/intensities grouped
            groups.setdefault(key, []).append(member)
    keys = sorted(groups)
    if len(keys) < 10:
        raise ValueError('Too few independent groups for actor '+str(actor))
    rng = random.Random(seed + actor)
    rng.shuffle(keys)
    n = len(keys)
    e = max(3, round(n*.25)); t = max(3, round(n*.35)); v = max(2, round(n*.20))
    # e=25%, train=35%, val=20%, test=remainder.
    groups_by_split = {'enroll': keys[:e], 'train':keys[e:e+t], 'val':keys[e+t:e+t+v], 'test':keys[e+t+v:]}
    if any(not selected for selected in groups_by_split.values()):
        raise ValueError('Split empty for actor '+str(actor))
    # Limit per actor while preserving all four splits.
    weights = {'enroll':.24,'train':.40,'val':.18,'test':.18}
    quotas = {k:max(3, int(limit*frac)) for k,frac in weights.items()}
    selected = []
    for split, subgroups in groups_by_split.items():
        candidates = [name for group in subgroups for name in groups[group]]
        rng.shuffle(candidates)
        selected += [(name,split) for name in candidates[:quotas[split]]]
    rng.shuffle(selected)
    return selected


def prepare(source=None, output='public_data', max_clips=28, recreate=False):
    if max_clips < 16:
        raise ValueError('max_clips should be at least 16 for 4 stable splits.')
    found = locate_archives(source)
    base = Path(output).resolve()
    base.mkdir(parents=True, exist_ok=True)
    manifest_path = base / 'manifest.csv'
    # Per-clip data can be skipped on reruns; always rebuild manifest to keep reproducible splits.
    entries=[]
    skipped=[]
    for actor, archive in found.items():
        print(f'Actor {actor:02d} — {archive.name}', flush=True)
        if not zipfile.is_zipfile(archive):
            raise ValueError(f'Incomplete or corrupt ZIP: {archive}')
        with zipfile.ZipFile(archive) as container:
            members=sorted(x for x in container.namelist() if PATTERN.match(Path(x).name)
                and int(PATTERN.match(Path(x).name).group(7)) == actor)
            if not members:
                raise ValueError(f'No modality 01 audio+video speech MP4s in {archive}. Check ZIP name/content.')
            chosen=split_members(members, actor, limit=max_clips)
            for item_no,(member,split) in enumerate(chosen,1):
                stem=Path(member).stem
                folder=base/f'actor_{actor:02d}'/split
                img=folder/'image'/f'{stem}.png'
                vid=folder/'video'/f'{stem}.npz'
                wav=folder/'voice'/f'{stem}.wav'
                if recreate or not (img.exists() and vid.exists() and wav.exists()):
                    for p in (img,vid,wav):
                        p.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory() as t:
                        extracted=Path(t)/f'{stem}.mp4'
                        with container.open(member) as src, extracted.open('wb') as dst:
                            shutil.copyfileobj(src, dst, length=2**20)
                        try:
                            still,frames=extract_frames(extracted)
                            extract_audio(extracted,wav)
                            if not cv2.imwrite(str(img),still):
                                raise RuntimeError('Cannot write still image')
                            np.savez_compressed(vid,frames=frames)
                        except Exception as exc:
                            for p in (img,vid,wav):
                                p.unlink(missing_ok=True)
                            skipped.append({'actor':actor,'stem':stem,'reason':str(exc)})
                            print(f'  SKIP: {stem}: {exc}', flush=True)
                            continue
                entries.append({'user':f'actor_{actor:02d}', 'actor':actor,'split':split,
                    'sample':stem, 'image':str(img),'voice':str(wav),'video':str(vid),
                    'emotion':int(stem.split('-')[2]),'statement':int(stem.split('-')[4]),
                    'repetition':int(stem.split('-')[5])})
                if item_no % 8 == 0:
                    print(f'  {item_no}/{len(chosen)} processed',flush=True)
    df=pd.DataFrame(entries)
    if df.empty:
        raise ValueError('No valid records created. Inspect skipped.json and test MP4 decoding.')
    df.to_csv(manifest_path,index=False)
    (base/'skipped.json').write_text(json.dumps(skipped,indent=2),encoding='utf-8')
    print('\nPrepared',len(df),'real matched audio/image/video clips')
    print(df.groupby(['user','split']).size().to_string())
    required={'enroll','train','val','test','val_unknown','test_unknown'}
    if not required <= set(df.split):
        raise ValueError('Missing split(s): '+str(required-set(df.split))+'; increase --max-clips or inspect skipped clips')
    for actor in KNOWN:
        counts=df[df.actor==actor].groupby('split').size()
        for part in ('enroll','train','val','test'):
            if counts.get(part,0)<2:
                raise ValueError(f'Too few valid clips for actor {actor} / {part}')
    return df


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',default=None,help='Folder containing seven ZIPs (default searches Downloads automatically)')
    p.add_argument('--output',default='public_data')
    p.add_argument('--max-clips',type=int,default=28)
    p.add_argument('--recreate',action='store_true')
    args=p.parse_args()
    prepare(args.source,args.output,args.max_clips,args.recreate)
