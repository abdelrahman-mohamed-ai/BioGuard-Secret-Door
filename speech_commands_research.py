"""BioGuard classical ML speech-command bonus.

Mini Speech Commands contains eight ISOLATED WORDS, not sentence transcripts.
Speech command recognition answers WHAT was said; public RAVDESS pairwise
voice verifier / enrolled gallery answers WHO spoke. Neither replaces the other.
The game must require BOTH a verified speaker and an allowed action.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import zipfile
import io

import soundfile as sf

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import bioguard_core as core
import public_research as research

WORDS = ('down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes')
ACTIONS = {'go': 'open', 'stop': 'close'}  # no other word triggers a protected action
URL = 'https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip'
SR = 16000
QUALITY_FIX_VERSION = 'quality-skip-v2'


def find_mini(source=None):
    if source:
        p = Path(source).expanduser()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for parent in (Path.home()/'Downloads', Path.cwd()/'datasets', Path.cwd()/'command_data'):
        for name in ('mini_speech_commands.zip','mini_speech_commands'):
            p = parent/name
            if p.exists():
                return p
    raise FileNotFoundError('Download mini_speech_commands.zip from '+URL+' and place it in Downloads')


def speaker_split(filename):
    # Same source speaker ID MUST remain within one of train/val/test.
    speaker = Path(filename).name.split('_nohash_')[0]
    n = int(hashlib.sha256(speaker.encode()).hexdigest()[:8],16)%100
    return 'train' if n<70 else ('val' if n<85 else 'test')


def extract_command_features(audio_file):
    """Handcrafted 82D: MFCC(13), delta, delta2, ZCR, RMS, each mean+std.

    Uses 1.0s clips (not the 3s speaker verification extractor).
    Quiet/missing clips are rejected rather than classified as commands.
    """
    y, sr = librosa.load(str(audio_file),sr=SR,mono=True)
    if not len(y):
        raise ValueError('Empty command audio')
    # Trim only leading/trailing silence, but don't require a 0.6s *spoken* word.
    y,_=librosa.effects.trim(y,top_db=38)
    if len(y)<int(.13*sr) or float(np.sqrt(np.mean(y**2)))<.002:
        raise ValueError('Silent or too-short command audio')
    # Align length to official dataset (one-second isolated word).
    y=librosa.util.fix_length(y,size=SR)
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13,n_fft=512,hop_length=160)
    delta=librosa.feature.delta(mfcc)
    delta2=librosa.feature.delta(mfcc,order=2)
    zcr=librosa.feature.zero_crossing_rate(y,frame_length=512,hop_length=160)
    rms=librosa.feature.rms(y=y,frame_length=512,hop_length=160)
    vec=np.concatenate([s for arr in (mfcc,delta,delta2,zcr,rms)
                        for s in (arr.mean(axis=1),arr.std(axis=1))]).astype('float32')
    if vec.shape!=(82,) or not np.isfinite(vec).all():
        raise ValueError('Invalid command feature vector')
    return vec


def _valid_pcm_wav(raw):
    """Validate both RIFF/WAVE signature and actual audio decoding.

    Some earlier cached `mini_selected/*.wav` files can be incomplete or invalid.
    Validating before reusing them prevents LibsndfileError halfway through training.
    """
    if len(raw) < 44 or raw[:4] not in (b'RIFF', b'RF64') or raw[8:12] != b'WAVE':
        return False
    try:
        with sf.SoundFile(io.BytesIO(raw), mode='r') as f:
            if f.samplerate < 8000 or f.frames < max(1, int(.13 * f.samplerate)):
                return False
            # Actually decode samples, rather than trusting a file extension.
            samples = f.read(frames=min(f.frames, f.samplerate), dtype='float32')
            return len(samples) > 0 and np.isfinite(samples).all()
    except (OSError, ValueError, RuntimeError):
        return False


def prepare_mini(source=None,out='command_data/mini_selected',max_per_word_train=180,
                 max_per_word_eval=45):
    """Select speaker-disjoint examples and repair invalid/stale cache files.

    Uses the original ZIP in Downloads; no re-download or extra dependency.
    If archive members themselves are invalid, reports them and tries further
    eligible samples before failing with a helpful error.
    """
    src=find_mini(source)
    output=Path(out)
    output.mkdir(parents=True,exist_ok=True)
    if src.is_file():
        try:
            archive=zipfile.ZipFile(src)
            members=[n for n in archive.namelist() if n.lower().endswith('.wav')]
        except zipfile.BadZipFile as exc:
            raise ValueError(f'Not a valid ZIP archive: {src}. Please re-download the official Mini Speech Commands ZIP.') from exc
    else:
        archive=None
        members=[str(p.relative_to(src)).replace('\\','/') for p in src.rglob('*.wav')]
    rows=[]
    issues=[]
    repaired=0
    try:
        for label in WORDS:
            files=[p for p in members if len(Path(p).parts)>=2 and Path(p).parts[-2].lower()==label]
            for split in ('train','val','test'):
                subset=sorted(p for p in files if speaker_split(p)==split)
                limit=max_per_word_train if split=='train' else max_per_word_eval
                dest=output/label/split
                dest.mkdir(parents=True,exist_ok=True)
                count=0
                for p in subset:
                    if count>=limit:
                        break
                    safe=dest/(hashlib.sha256(p.encode()).hexdigest()[:16]+'.wav')
                    try:
                        if safe.exists() and _valid_pcm_wav(safe.read_bytes()):
                            pass  # Valid cache: safe to reuse.
                        else:
                            raw=archive.read(p) if archive else (src/p).read_bytes()
                            if not _valid_pcm_wav(raw):
                                issues.append(f'{p}: invalid WAV in source (first 16 bytes={raw[:16]!r})')
                                continue
                            if safe.exists():
                                repaired+=1
                            temp=safe.with_suffix('.wav.tmp')
                            temp.write_bytes(raw)
                            temp.replace(safe)
                        rows.append({'word':label,'split':split,
                                     'speaker':Path(p).name.split('_nohash_')[0],
                                     'path':str(safe)})
                        count+=1
                    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                        issues.append(f'{p}: {type(exc).__name__}: {exc}')
                        continue
                if count<8:
                    examples='\n'.join(issues[:5]) if issues else '(no readable .wav members)'
                    raise ValueError(
                        f'Only {count} valid recordings for {label}/{split} in {src}. '
                        f'Check whether the ZIP is the official Mini Speech Commands archive.\n'
                        f'Sample invalid files:\n{examples}'
                    )
    finally:
        if archive:archive.close()
    frame=pd.DataFrame(rows)
    assert all(not(set(frame[frame.split==a].speaker)&set(frame[frame.split==b].speaker))
               for a,b in [('train','val'),('train','test'),('val','test')]),'Speaker leakage'
    frame.to_csv(output/'manifest.csv',index=False)
    print('Mini Speech Commands:',pd.crosstab(frame.word,frame.split))
    print(f'Audio validation: {len(frame)} readable WAV files; repaired {repaired} corrupted cached files; skipped {len(issues)} invalid source files.')
    if issues:
        print('First invalid source:',issues[0])
    return frame


def train_mini(source=None,output='model/commands_model.joblib', max_per_word_train=180,
               max_per_word_eval=45):
    """Train on valid isolated-word WAVs, skipping damaged/silent source clips.

    Every skipped recording is logged to a CSV, and held-out test speakers remain
    disjoint from training speakers. Runtime audio validation remains strict.
    """
    frame=prepare_mini(source,max_per_word_train=max_per_word_train,max_per_word_eval=max_per_word_eval)
    artifact=Path(output)
    artifact.parent.mkdir(parents=True,exist_ok=True)

    feature_rows=[]
    audit=[]
    for item in frame.itertuples(index=False):
        try:
            features=extract_command_features(item.path)
            feature_rows.append((item.word,item.split,item.speaker,item.path,features))
            audit.append({'word':item.word,'split':item.split,'speaker':item.speaker,
                          'path':item.path,'status':'valid','reason':''})
        except (ValueError,OSError,RuntimeError,EOFError) as exc:
            # Never turn silence into a valid command; exclude bad TRAINING samples only.
            audit.append({'word':item.word,'split':item.split,'speaker':item.speaker,
                          'path':item.path,'status':'skipped',
                          'reason':f'{type(exc).__name__}: {exc}'})

    audit_df=pd.DataFrame(audit)
    audit_path=artifact.parent/'commands_quality_audit.csv'
    audit_df.to_csv(audit_path,index=False)
    rejected=audit_df[audit_df.status=='skipped']
    print(f'Feature audit: {len(feature_rows)} valid, {len(rejected)} skipped; report: {audit_path}')
    if len(rejected):
        print('Examples of skipped low-quality/broken recordings:')
        print(rejected[['word','split','path','reason']].head(6).to_string(index=False))

    # Keep only audio whose features were successfully computed.
    clean=pd.DataFrame([{'word':word,'split':split,'speaker':speaker,'path':path}
                        for word,split,speaker,path,_ in feature_rows])
    if clean.empty:
        raise ValueError(f'All command WAVs failed quality checks. See {audit_path}')
    counts=pd.crosstab(clean.word,clean.split).reindex(index=WORDS,
                         columns=['train','val','test'],fill_value=0)
    print('Usable recordings by label and split:\n',counts)
    counts.to_csv(artifact.parent/'commands_clean_split_counts.csv')
    clean.to_csv(artifact.parent/'commands_clean_manifest.csv',index=False)
    if (counts<8).any().any():
        raise ValueError(f'Too few valid samples after rejecting low-quality audio. '
                         f'At least 8 per word/split needed. See {audit_path} '
                         f'and commands_clean_split_counts.csv')

    assert all(not(set(clean[clean.split==a].speaker) & set(clean[clean.split==b].speaker))
               for a,b in [('train','val'),('train','test'),('val','test')]),'Speaker leakage'

    X={}; y={}
    for split in ('train','val','test'):
        part=[row for row in feature_rows if row[1]==split]
        X[split]=np.stack([row[4] for row in part])
        y[split]=np.asarray([row[0] for row in part])

    models={
        'SVM':Pipeline([('scale',StandardScaler()),('model',SVC(kernel='rbf',C=2,
                     probability=True,random_state=42))]),
        'Random Forest':RandomForestClassifier(n_estimators=160,random_state=42,n_jobs=-1),
    }
    results=[]; detail={}; fitted={}
    for name,mdl in models.items():
        mdl.fit(X['train'],y['train']);fitted[name]=mdl
        val=mdl.predict(X['val']);test=mdl.predict(X['test'])
        results.append({'model':name,'val_accuracy':accuracy_score(y['val'],val),
                        'val_macro_F1':f1_score(y['val'],val,average='macro',zero_division=0),
                        'test_accuracy':accuracy_score(y['test'],test),
                        'test_macro_F1':f1_score(y['test'],test,average='macro',zero_division=0)})
        detail[name]={'val':val,'test':test}
    df=pd.DataFrame(results)
    best=str(df.sort_values(['val_macro_F1','val_accuracy'],ascending=False).iloc[0]['model'])
    scores=np.max(fitted[best].predict_proba(X['val']),axis=1)
    correct=detail[best]['val']==y['val']
    # Validation-only rule, not a guarantee of rejecting unseen phrases.
    reject_threshold=min(.99,max(.55,float(np.quantile(scores[correct],.10)))) if correct.any() else .75
    joblib.dump({'model':fitted[best],'name':best,'words':list(WORDS),'actions':ACTIONS,
                 'confidence_threshold':float(reject_threshold),
                 'feature_extractor':'speech_commands_research.extract_command_features',
                 'notes':'Eight isolated English words, not sentences. Check speaker & permission too.'},artifact)
    df.to_csv(artifact.parent/'commands_comparison.csv',index=False)
    detail_rows=[]
    for name in models:
        for true,pred in zip(y['test'],detail[name]['test']):
            detail_rows.append({'model':name,'true':true,'predicted':pred})
    pd.DataFrame(detail_rows).to_csv(artifact.parent/'commands_test_predictions.csv',index=False)
    print('Selected by validation:',best,'command threshold:',reject_threshold)
    print(df.to_string(index=False))
    return {'comparison':df,'true':y['test'],'predicted':detail[best]['test'],
            'models':fitted,'manifest':clean,'best':best,'threshold':reject_threshold,
            'quality_audit':audit_df,'quality_audit_path':str(audit_path)}


def predict_command_diagnostics(audio_file,path='model/commands_model.joblib'):
    """Run the unchanged trained command model and expose safe diagnostics.

    The feature extractor, model artifact, class decision, and confidence
    threshold are unchanged.  The additional top-three and margin fields are
    for command troubleshooting and admin diagnostics; they are not an
    authorization decision.
    """
    pack=joblib.load(path) # Only trusted local files
    features=extract_command_features(audio_file)
    scores=pack['model'].predict_proba(features[None,:])[0]
    classes=[str(value) for value in pack['model'].classes_]
    order=np.argsort(scores)[::-1]
    top=[{'word':classes[int(index)],'confidence':float(scores[int(index)])}
         for index in order[:3]]
    label=top[0]['word']; confidence=float(top[0]['confidence'])
    threshold=float(pack['confidence_threshold'])
    margin=float(top[0]['confidence']-top[1]['confidence']) if len(top)>1 else confidence
    recognized=bool(confidence>=threshold)
    return {
        'recognized':recognized,
        'word':label if recognized else None,
        'predicted_word':label,
        'action':pack.get('actions',ACTIONS).get(label) if recognized else None,
        'score':confidence,
        'confidence':confidence,
        'threshold':threshold,
        'margin':margin,
        'top_predictions':top,
        'feature_dim':int(features.shape[0]),
    }


def predict_command(audio_file,path='model/commands_model.joblib'):
    return predict_command_diagnostics(audio_file,path)


def verify_voice_only_command(command_audio_file, public_package, gallery, command_package=None,
                              policy=None, speaker_audio_file=None, reject_margin=0.):
    """Server-side gate. Independent identity verification, keyword, permission.

    `speaker_audio_file` can be a distinct 2-3s live speaker challenge; if None,
    same recording is used (short 1s words may be too short/unreliable to identify).
    No replay/liveness protection is claimed.
    """
    if not gallery.get('templates'):
        return {'accepted':False,'identity':'Unknown','reason':'Empty enrollment gallery'}
    speaker_audio_file=speaker_audio_file or command_audio_file
    try:
        voice=core.extract_voice(speaker_audio_file)
    except (ValueError,RuntimeError) as e:
        return {'accepted':False,'identity':'Unknown','reason':'Speaker audio insufficient'}
    entry=public_package['models']['Voice Only']
    candidates={}
    for name,temps in gallery['templates'].items():
        if 'voice' in temps:
            pair=core.pair_difference(voice,temps['voice'])[None,:]
            candidates[name]=float(research.proba(entry['model'],pair)[0])
    if not candidates:
        return {'accepted':False,'identity':'Unknown','reason':'No voice enrollment'}
    candidate=max(candidates,key=candidates.get)
    if candidates[candidate]<float(entry['threshold'])+reject_margin:
        return {'accepted':False,'identity':'Unknown','reason':'Speaker unverified'}
    if not gallery.get('permissions',{}).get(candidate,False):
        return {'accepted':False,'identity':'Unknown','reason':'Speaker unauthorized'}
    pack=command_package or joblib.load('model/commands_model.joblib')
    try:
        feat=extract_command_features(command_audio_file)[None,:]
    except ValueError:
        return {'accepted':False,'identity':'Unknown','reason':'Command audio insufficient'}
    scores=pack['model'].predict_proba(feat)[0]
    label=str(pack['model'].classes_[int(scores.argmax())])
    if float(scores.max())<pack['confidence_threshold']:
        return {'accepted':False,'identity':'Unknown','reason':'Command uncertain'}
    action=pack['actions'].get(label)
    if not action:
        return {'accepted':False,'identity':'Unknown','reason':'Non-action word'}
    if action not in (policy or {}).get(candidate,set()):
        return {'accepted':False,'identity':'Unknown','reason':'Missing action permission'}
    return {'accepted':True,'identity':candidate,'command':label,'action':action,
            'reason':'Verified speaker + allowed word'}


def train_commands(root=None,output='model/commands_model.joblib'):
    try:return train_mini(source=root,output=output)['comparison']
    except FileNotFoundError as exc:
        print('COMMAND DATA NOT DOWNLOADED:',exc);return None


def authorized_command(identity,command,verified,access_policy):
    return bool(verified and identity in access_policy and command in access_policy[identity])

if __name__=='__main__':train_commands()
