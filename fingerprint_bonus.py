"""Optional REAL fingerprint verification experiment (Sobel + LBP + SVM/RF).
Expected: fingerprint_data/<person>/<scan01>.png, >=4 images per person, >=3 persons.
Never fuse RAVDESS faces/audio with unrelated fingerprint identities.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import bioguard_core as core
import public_research as research


def run(root='fingerprint_data',output='model/fingerprint_model.joblib'):
    root=Path(root)
    rows=[]
    for person in sorted(root.iterdir()) if root.exists() else []:
        if not person.is_dir():continue
        files=sorted(p for p in person.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'))
        if len(files)<5:continue
        for p in files:
            rows.append((person.name,p))
    actors=sorted(set(p for p,_ in rows))
    if len(actors)<3:
        print('BONUS PENDING: add >=5 real fingerprint images per >=3 consenting people in fingerprint_data/<user>/')
        return None
    refs={}; trials=[]
    for name in actors:
        imgs=[p for n,p in rows if n==name]
        feats=[core.extract_fingerprint(p) for p in imgs]
        refs[name]=np.stack(feats[:2]).mean(axis=0)
        for j,feat in enumerate(feats[2:]):
            for claim in actors:
                trials.append((name,claim,j, int(name==claim),core.pair_difference(feat,refs[claim])))
    split_groups={name: [t for t in trials if t[0]==name] for name in actors}
    tr=[]; va=[]; te=[]
    for person,part in split_groups.items():
        unique_j=sorted(set(t[2] for t in part))
        if len(unique_j)<3:raise ValueError('Need >=5 separate scans per user')
        for t in part:
            (tr if t[2] < unique_j[-2] else va if t[2]==unique_j[-2] else te).append(t)
    X=lambda data:np.stack([r[-1] for r in data]);y=lambda data:np.array([r[-2] for r in data])
    records=[]; selected=[]
    for name in ('SVM','Random Forest'):
        mdl=research.make_model(name)
        mdl.fit(X(tr),y(tr))
        pva=research.proba(mdl,X(va)); pte=research.proba(mdl,X(te))
        th=research.pick_threshold(y(va),pva,.10)
        mv=research.evaluate_binary(y(va),pva,th);mt=research.evaluate_binary(y(te),pte,th)
        records.append({'model':name,'threshold':th,**{'val_'+k:v for k,v in mv.items()},
                **{'test_'+k:v for k,v in mt.items()}})
        selected.append((mv['balanced_accuracy'],name,mdl,th))
    best=max(selected,key=lambda x:x[0])
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':best[2],'threshold':best[3],'templates':refs},output)
    table=pd.DataFrame(records)
    table.to_csv(Path(output).parent/'fingerprint_comparison.csv',index=False)
    print(table[['model','val_FAR','val_FRR','test_FAR','test_FRR']].to_string(index=False))
    return table

if __name__=='__main__':run()
