"""SOCOFing fingerprint verification experiment, traditional ML ONLY.

One physical finger (subject + side + finger) is an identity. Original scan
is gallery reference, altered scans are probes, and TRAIN/VAL/TEST are disjoint
by source subject (all 10 fingers from a person share one split). Synthetic alterations are NOT independent scan sessions, so this
benchmark cannot establish real scanner accuracy.

Download: https://www.kaggle.com/datasets/ruizgara/socofing
Place the downloaded ZIP in Downloads (any filename with 'socofing') OR extract
it to ~/Downloads/SOCOFing/. Script auto-detects either.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from itertools import groupby
import re, zipfile, hashlib
import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import bioguard_core as core
import public_research as research

IMG_EXTS={'.bmp','.png','.jpg','.jpeg','.tif','.tiff'}
# Synthetic alteration suffixes used by the SOCOFing authors.
SUFFIX = re.compile(r'_(?:cr|obl|zcut)$', re.I)


def finger_id(name):
    stem=Path(name).stem
    return SUFFIX.sub('',stem).lower()


def _source(source=None):
    if source:
        p=Path(source).expanduser()
        if not p.exists():raise FileNotFoundError(p)
        return p
    roots=[Path.home()/'Downloads', Path.cwd()/'fingerprint_data',Path.cwd()/'datasets']
    for base in roots:
        if not base.exists():continue
        for p in sorted(base.iterdir()):
            if 'socofing' in p.name.lower() and (p.is_dir() or p.suffix.lower()=='.zip'):
                return p
    raise FileNotFoundError('Download SOCOFing ZIP from https://www.kaggle.com/datasets/ruizgara/socofing and place it in Downloads')


def _catalog(source):
    """Return per-finger original & altered entries; zip supported, no full extraction."""
    source=Path(source)
    files = [x.filename for x in zipfile.ZipFile(source).infolist() if not x.is_dir()] if source.is_file() else [str(p.relative_to(source)).replace('\\','/') for p in source.rglob('*') if p.is_file()]
    groups=defaultdict(lambda:{'real':None,'altered':[]})
    for path in files:
        if Path(path).suffix.lower() not in IMG_EXTS:continue
        slash='/'+path.replace('\\','/').lower()+'/'
        if '/real/' in slash: kind='real'
        elif '/altered-' in slash:kind='altered'
        else:continue
        key=finger_id(path)
        if kind=='real':groups[key]['real']=path
        else:groups[key]['altered'].append(path)
    return {k:v for k,v in groups.items() if v['real'] and v['altered']}


def prepare_socofing(source=None, max_fingers=120, max_alterations=3,
                     output='fingerprint_data/selected', seed=42):
    """Extract a small, reproducible subset, with safe generated paths."""
    src=_source(source)
    groups=_catalog(src)
    if len(groups)<35:raise ValueError(f'Found only {len(groups)} fingers with originals + altered scans; need >=35')
    keys=np.random.default_rng(seed).permutation(sorted(groups))[:max_fingers].tolist()
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    z=zipfile.ZipFile(src) if src.is_file() else None
    rows=[]
    try:
        for i,key in enumerate(keys):
            group=groups[key]
            originals=[group['real']]
            variations=sorted(group['altered'])[:max_alterations]
            for kind, paths in [('real',originals),('altered',variations)]:
                for j,inner in enumerate(paths):
                    data=z.read(inner) if z else (src/inner).read_bytes()
                    im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
                    if im is None:continue
                    name=hashlib.sha256(key.encode()).hexdigest()[:12]
                    target=dest/kind/(name+f'_{j:02d}.png')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    cv2.imwrite(str(target), im)
                    rows.append({'finger':key,'kind':kind,'path':str(target),
                                 'source':str(inner)})
    finally:
        if z:z.close()
    result=pd.DataFrame(rows)
    result.to_csv(dest/'manifest.csv',index=False)
    print(f'SOCOFing: {result.finger.nunique()} fingers, {len(result)} images; source={src}')
    return result


def _trial_set(keys, feats, refs, rng, per_probe_impostors=2):
    pairs=[]
    keys=sorted(keys)
    for k in keys:
        for feature in feats[k]:
            pairs.append((k,k,1,core.pair_difference(feature,refs[k])))
            others=[x for x in keys if x!=k]
            for wrong in rng.choice(others,size=min(len(others),per_probe_impostors),replace=False):
                pairs.append((k,str(wrong),0,core.pair_difference(feature,refs[str(wrong)])))
    return pairs


def train_socofing(source=None,max_fingers=120,max_alterations=3,
                    output='model/fingerprint_model.joblib',seed=42):
    manifest=prepare_socofing(source,max_fingers,max_alterations)
    refs={}; altered={}
    for k,grp in manifest.groupby('finger'):
        real=grp[grp.kind=='real'].path.to_list()
        probes=grp[grp.kind=='altered'].path.to_list()
        if not real or not probes:continue
        refs[k]=core.extract_fingerprint(real[0])
        altered[k]=[core.extract_fingerprint(p) for p in probes]
    keys=sorted(refs)
    if len(keys)<35:raise ValueError('Insufficient fingerprint identities')
    # Do not allow one person's OTHER fingers to leak into another split.
    by_person=defaultdict(list)
    for key in keys:
        person=key.split('__')[0]
        by_person[person].append(key)
    people=sorted(by_person)
    if len(people)<15:
        raise ValueError(f'Need at least 15 different subjects, found {len(people)}')
    train_people,remaining=train_test_split(people,random_state=seed,test_size=.40)
    val_people,test_people=train_test_split(remaining,random_state=seed,test_size=.50)
    train_keys=[finger for person in train_people for finger in by_person[person]]
    val_keys=[finger for person in val_people for finger in by_person[person]]
    test_keys=[finger for person in test_people for finger in by_person[person]]
    assert not(set(train_people)&set(val_people)|set(train_people)&set(test_people)|set(val_people)&set(test_people))
    rng=np.random.default_rng(seed)
    splits={'train':_trial_set(train_keys,altered,refs,rng),
            'val':_trial_set(val_keys,altered,refs,rng),
            'test':_trial_set(test_keys,altered,refs,rng)}
    X={s:np.stack([r[-1] for r in vals]) for s,vals in splits.items()}
    y={s:np.array([r[2] for r in vals]) for s,vals in splits.items()}
    reports=[]; fitted={}; selected=[]
    for name in ['SVM','Random Forest']:
        model=research.make_model(name)
        model.fit(X['train'],y['train'])
        val_score=research.proba(model,X['val'])
        test_score=research.proba(model,X['test'])
        threshold=float(research.pick_threshold(y['val'],val_score,far_max=.05))
        val_metrics=research.evaluate_binary(y['val'],val_score,threshold)
        test_metrics=research.evaluate_binary(y['test'],test_score,threshold)
        reports.append({'model':name,'threshold':threshold,**{'val_'+k:v for k,v in val_metrics.items()},
                        **{'test_'+k:v for k,v in test_metrics.items()}})
        fitted[name]=model
        selected.append((val_metrics['balanced_accuracy'],-val_metrics['FAR'],name,threshold))
    _,_,best,threshold=max(selected)
    result=pd.DataFrame(reports)
    artifact=Path(output);artifact.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':fitted[best],'model_name':best,'threshold':threshold,
                 'demo_gallery':{key:refs[key] for key in test_keys},
                 'dataset':'SOCOFing','method':'Original print vs synthetic altered probe; subject-held-out',
                 'limits':'Synthetic altered probe is NOT an independent capture; no real-time scanner'},artifact)
    result.to_csv(artifact.parent/'fingerprint_comparison.csv',index=False)
    detail=pd.DataFrame([{'finger':q[0],'claim':q[1],'label':q[2],
                **{f'score_{model_name}':float(research.proba(fitted[model_name],q[-1][None,:])[0])
                   for model_name in fitted}} for q in splits['test']])
    detail.to_csv(artifact.parent/'fingerprint_test_predictions.csv',index=False)
    print('Fingerprint select based on VALIDATION only:',best,'Threshold:',threshold)
    print(result[['model','val_FAR','val_FRR','test_FAR','test_FRR','test_balanced_accuracy']])
    return {'comparison':result,'predictions':detail,'manifest':manifest,
            'split_fingers':{s:len(ids) for s,ids in zip(['train','val','test'],[train_keys,val_keys,test_keys])},
            'best':best,'threshold':threshold}


def verify_upload(image_file, claimed_finger, model_path='model/fingerprint_model.joblib'):
    """Offline image-upload 1:1 fingerprint demonstration, NOT a door input."""
    pack=joblib.load(model_path)  # Only load trusted, locally generated model files.
    ref=pack['demo_gallery'].get(claimed_finger)
    if ref is None:return {'match':False,'reason':'Unknown fingerprint ID'}
    query=core.extract_fingerprint(image_file)
    score=float(research.proba(pack['model'],core.pair_difference(query,ref)[None,:])[0])
    return {'match':score>=pack['threshold'],'score':score,'threshold':pack['threshold']}

def identify_upload(image_file, model_path='model/fingerprint_model.joblib'):
    """Optional academic 1:N gallery search for a single uploaded BMP/PNG/JPG.

    A fingerprint database ID is returned only when score beats the validation
    threshold; these are SOCOFing FINGER identities, NOT people in the game.
    Unknown/ambiguous comparisons may still produce false accepts.
    """
    pack=joblib.load(model_path)  # load ONLY our own generated local file
    gallery=pack.get('demo_gallery',{})
    if not gallery:
        return {'match':False,'finger':'Unknown','reason':'No reference fingerprints'}
    query=core.extract_fingerprint(image_file)
    scores={k:float(research.proba(pack['model'],core.pair_difference(query,ref)[None,:])[0])
            for k,ref in gallery.items()}
    best=max(scores,key=scores.get)
    match=bool(scores[best]>=pack['threshold'])
    return {'match':match,'finger':best if match else 'Unknown',
            'score':scores[best],'threshold':float(pack['threshold']),
            'compared_with':len(scores)}


# Backwards-compatible callable used in the old Notebook; no invented results.
def run(root=None,output='model/fingerprint_model.joblib'):
    try:return train_socofing(root,output=output)['comparison']
    except FileNotFoundError as exc:
        print('FINGERPRINT DATA NOT DOWNLOADED:',exc);return None

if __name__=='__main__':run()
