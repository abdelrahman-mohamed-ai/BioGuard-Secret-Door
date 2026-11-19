"""Classical multimodal verification research; public RAVDESS data only.
Unknown speakers for validation and test are isolated by actor ID.
Image: one still; video: multiple other face frames; voice: aligned WAV.
Video is an additional temporal FACE representation, not a separate physical biometric.
"""
from __future__ import annotations
import itertools
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, balanced_accuracy_score)
import bioguard_core as core

MODES = {
    'Image Only': ('image',),
    'Voice Only': ('voice',),
    'Video Only': ('video',),
    'Early Image+Voice': ('image','voice'),
    'Early Video+Voice': ('video','voice'),
    'Early Image+Video': ('image','video'),
    'Early All Three': ('image','voice','video'),
}
SEED = 42


def video_vector(file):
    """Aggregate HOG+LBP features of 5-6 video frames with mean+std."""
    with np.load(str(file), allow_pickle=False) as data:
        frames = data['frames']
    if len(frames) < 4:
        raise ValueError('Video needs at least 4 face frames')
    feats = np.stack([core.extract_face(frame, require_face=False, already_cropped=True) for frame in frames])
    return np.concatenate([feats.mean(axis=0), feats.std(axis=0)]).astype(np.float32)


def read_manifest(path='public_data/manifest.csv'):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{path} missing. Run download_prepare_ravdess.py first')
    df = pd.read_csv(path)
    cols = ['user','actor','split','sample','image','voice','video']
    if not all(c in df.columns for c in cols):
        raise ValueError('Incomplete manifest; required columns: '+str(cols))
    if not {'enroll','train','val','test','val_unknown','test_unknown'} <= set(df.split):
        raise ValueError('Missing one or more splits; download more clips/actors')
    if any(df[['user','sample']].duplicated()):
        raise ValueError('Repeated source clips across splits! Leakage risk')
    return df


def compute_features(manifest, cache='public_data/feature_cache.joblib'):
    path = Path(cache)
    if path.exists():
        obj = joblib.load(path)
        if (obj.get('samples') == manifest[['user','split','sample']].to_dict('records')
            and obj.get('extractor_version') == 'public_cropped_v2'):
            return obj['frame']
    df = manifest.copy()
    for name, fn in [('image', lambda f:core.extract_face(f,already_cropped=True)),('voice',core.extract_voice),('video',video_vector)]:
        values = []
        for i, filepath in enumerate(df[name]):
            values.append(fn(filepath))
            if (i+1)%20==0: print(f'{name}: {i+1}/{len(df)}')
        df[name+'_vec'] = values
    path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'samples':manifest[['user','split','sample']].to_dict('records'),
                 'extractor_version':'public_cropped_v2', 'frame':df},path)
    return df


def build_templates(df):
    templates = {}
    enrolled = df[df.split=='enroll']
    for person,group in enrolled.groupby('user'):
        if len(group)<3: raise ValueError(f'Need 3+ enrollment clips for {person}')
        templates[person] = {mod:np.stack(group[mod+'_vec']).mean(axis=0)
                             for mod in ('image','voice','video')}
    if len(templates)<3:
        raise ValueError('Need 3+ known public actors to train verification pairs')
    return templates


def pair_features(vecs, template, mods):
    return np.concatenate([core.pair_difference(vecs[m],template[m]) for m in mods])


def make_trials(df, templates, split):
    """Each sample paired with every registered identity; unknown has NO positive claim."""
    records=[]
    for row in df[df.split == split].itertuples():
        vecs={m:getattr(row,m+'_vec') for m in ('image','voice','video')}
        for claim,ref in templates.items():
            features={mode:pair_features(vecs, ref, mod) for mode,mod in MODES.items()}
            records.append({'true':row.user,'claim':claim,'label':int(row.user==claim),
                            'sample':row.sample,'x':features})
    return records


def make_model(name):
    if name=='SVM':
        return Pipeline([('scale',StandardScaler()),('svm',SVC(C=2,kernel='rbf',
                                    probability=True,class_weight='balanced',random_state=SEED))])
    if name=='Random Forest':
        return RandomForestClassifier(n_estimators=130,min_samples_leaf=2,
                 class_weight='balanced',n_jobs=-1,random_state=SEED)
    raise ValueError(name)


def proba(model, X):
    return np.asarray(model.predict_proba(X)[:,list(model.classes_).index(1)],dtype=float)


def pick_threshold(labels, scores, far_max=.05):
    y=np.asarray(labels,dtype=int)
    p=np.asarray(scores,dtype=float)
    thresholds=np.unique(np.concatenate((p,[0,1,np.nextafter(float(p.max()),np.inf)])))
    candidates=[]
    for th in thresholds:
        pred=p>=th
        neg=(y==0); pos=(y==1)
        far=float(pred[neg].mean()) if neg.any() else 0.0
        tar=float(pred[pos].mean()) if pos.any() else 0.0
        if far <= far_max + 1e-12:
            candidates.append((tar,-far,-float(th),float(th)))
    if not candidates: return float(np.nextafter(p.max(),np.inf))
    return max(candidates)[-1]


def evaluate_binary(labels,score,threshold):
    y=np.asarray(labels,dtype=int); pred=(np.asarray(score)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall_TAR':float(recall_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0)),
            'FAR':fp/(fp+tn) if fp+tn else np.nan,
            'FRR':fn/(fn+tp) if fn+tp else np.nan,
            'ROC_AUC':float(roc_auc_score(y,score)) if len(set(y))==2 else np.nan,
            'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}


def train_public(df, artifact='model/public_models.joblib',target_far=.05):
    templates=build_templates(df)
    trains=make_trials(df,templates,'train')
    vals=make_trials(df,templates,'val')+make_trials(df,templates,'val_unknown')
    tests=make_trials(df,templates,'test')+make_trials(df,templates,'test_unknown')
    if min(len(trains),len(vals),len(tests))==0:
        raise ValueError('Empty train/val/test trials')
    ytrain=np.array([t['label'] for t in trains]); yval=np.array([t['label'] for t in vals]); ytest=np.array([t['label'] for t in tests])
    chosen={}; model_rows=[]; result_rows=[]; scores={}
    for mode, mods in MODES.items():
        Xtrain=np.stack([t['x'][mode] for t in trains]); Xval=np.stack([t['x'][mode] for t in vals]); Xtest=np.stack([t['x'][mode] for t in tests]);
        possibilities=[]
        for model_name in ('SVM','Random Forest'):
            model=make_model(model_name)
            model.fit(Xtrain,ytrain)
            pv=proba(model,Xval); pt=proba(model,Xtest)
            th=pick_threshold(yval,pv,target_far)
            mv=evaluate_binary(yval,pv,th)
            mt=evaluate_binary(ytest,pt,th)
            model_rows.append({'mode':mode,'model':model_name,'validation_threshold':th,
                               **{'val_'+k:v for k,v in mv.items()},
                               **{'test_'+k:v for k,v in mt.items()}})
            # Match project priority: target FAR constraint, then validation TAR / balanced accuracy.
            possibilities.append((mv['balanced_accuracy'], mv['recall_TAR'], -mv['FAR'],model_name, model, th,pv,pt))
        best=max(possibilities,key=lambda x:x[:3]); selected_name,best_model,threshold,pv,pt=best[3:]
        chosen[mode]={'model':best_model,'model_name':selected_name,'threshold':float(threshold), 'modalities':mods}
        scores[mode]={'val':pv,'test':pt}
        result_rows.append({'mode':mode,'selected_model':selected_name,'threshold':threshold,
                            **evaluate_binary(ytest,pt,threshold)})
        print(f'{mode:25} {selected_name:14} val threshold={threshold:.4f}')
    # Use SAME samples/claims for late and hierarchical trials.
    for split in ('val','test'):
        scores.setdefault('Late Fusion',{})[split]=(scores['Image Only'][split]+scores['Voice Only'][split])/2
        a=scores['Image Only'][split]>=chosen['Image Only']['threshold']
        b=scores['Voice Only'][split]>=chosen['Voice Only']['threshold']
        scores.setdefault('Hierarchical',{})[split]=(a & b).astype(float)
    late=pick_threshold(yval,scores['Late Fusion']['val'],target_far)
    for mode,threshold in [('Late Fusion',late),('Hierarchical',.5)]:
        result_rows.append({'mode':mode,'selected_model':'Decision rule','threshold':threshold,
                            **evaluate_binary(ytest,scores[mode]['test'],threshold)})
    out=Path(artifact);out.parent.mkdir(parents=True,exist_ok=True)
    package={'models':chosen,'templates':templates,'late_threshold':late,
             'policy':{'target_validation_FAR':target_far,
                       'deployment_requires_face_voice_gate':True},
             'public_actors':sorted(templates),'version':'2.0',
             'feature_config':{'image':'HOG 9/12x12/2x2 + uniform LBP 8,1 on 96x96',
                              'voice':'MFCC13+delta+delta2+ZCR+RMS mean/std at 16kHz 3s',
                              'video':'6 cropped frames HOG+LBP mean/std'},
             'enrolled_gallery_is_not_the_public_test_gallery':True}
    joblib.dump(package,out)
    model_report=pd.DataFrame(model_rows)
    summary=pd.DataFrame(result_rows)
    summary.to_csv(out.parent/'public_mode_comparison.csv',index=False)
    model_report.to_csv(out.parent/'public_model_comparison.csv',index=False)
    open_set_details=[]
    for mode in list(MODES)+['Late Fusion','Hierarchical']:
        for split in ('test', 'test_unknown'):
            for row in df[df.split==split].itertuples():
                vecs={m:getattr(row,m+'_vec') for m in ('image','voice','video')}
                decision=identify_from_vectors(package,vecs,mode=mode,templates=templates)
                expected=row.user if row.user in templates else 'Unknown'
                open_set_details.append({'mode':mode,'actual':expected,'predicted':decision['identity'],
                                         'accepted':decision['accepted'],'sample':row.sample})
    details=pd.DataFrame(open_set_details)
    details.to_csv(out.parent/'open_set_test_predictions.csv',index=False)
    open_summary=[]
    for mode,part in details.groupby('mode'):
        unk=part[part.actual=='Unknown']; known=part[part.actual!='Unknown']
        open_summary.append({'mode':mode,'known_id_accuracy':(known.predicted==known.actual).mean(),
                             'unknown_rejection_rate':(unk.predicted=='Unknown').mean(),
                             'unknown_false_accept_rate':(unk.predicted!='Unknown').mean(),
                             'overall_accuracy':(part.predicted==part.actual).mean()})
    opens=pd.DataFrame(open_summary)
    opens.to_csv(out.parent/'open_set_comparison.csv',index=False)
    return {'models':model_report,'summary':summary,'open_summary':opens,'details':details,
            'package':package,'test_labels':ytest,'test_scores':scores,'test_trials':tests}


def score_claim(package, vecs, claim, mode, templates):
    if mode in MODES:
        entry=package['models'][mode]
        X=pair_features(vecs,templates[claim],entry['modalities'])[None,:]
        return float(proba(entry['model'],X)[0])
    if mode=='Late Fusion':
        return .5*(score_claim(package,vecs,claim,'Image Only',templates)+
                   score_claim(package,vecs,claim,'Voice Only',templates))
    raise ValueError(mode)


def identify_from_vectors(package,vecs,mode='Hierarchical',templates=None,permissions=None):
    templates=templates if templates is not None else package['templates']
    if not templates: return {'accepted':False,'identity':'Unknown','reason':'No users enrolled'}
    if mode=='Hierarchical':
        candidate=max(templates,key=lambda u:score_claim(package,vecs,u,'Image Only',templates))
        p_face=score_claim(package,vecs,candidate,'Image Only',templates)
        p_voice=score_claim(package,vecs,candidate,'Voice Only',templates)
        allowed=(p_face>=package['models']['Image Only']['threshold'] and
                 p_voice>=package['models']['Voice Only']['threshold'])
        if permissions is not None: allowed=allowed and bool(permissions.get(candidate,False))
        return {'accepted':bool(allowed),'identity':candidate if allowed else 'Unknown',
                'reason':'Both face and voice accepted' if allowed else 'Rejected',
                'scores':{'face':p_face,'voice':p_voice}}
    if mode not in MODES and mode!='Late Fusion':
        raise ValueError(mode)
    winner=max(templates,key=lambda u:score_claim(package,vecs,u,mode,templates))
    score=score_claim(package,vecs,winner,mode,templates)
    threshold=package['late_threshold'] if mode=='Late Fusion' else package['models'][mode]['threshold']
    allowed=score>=threshold
    if permissions is not None:allowed=allowed and bool(permissions.get(winner,False))
    return {'accepted':bool(allowed),'identity':winner if allowed else 'Unknown',
            'reason':'Accepted' if allowed else 'Rejected','score':score,'threshold':threshold}


if __name__=='__main__':
    manifest=read_manifest()
    features=compute_features(manifest)
    result=train_public(features)
    print('\nTEST SET RESULTS (not validation):\n',result['summary'].to_string(index=False))
    print('\nUNKNOWN REJECTION:\n',result['open_summary'].to_string(index=False))
