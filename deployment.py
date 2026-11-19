"""Exported verifier + live enrollment gallery. Does NOT retrain global SVM for new users.
Traditional ML pair scoring stays fixed, and identity templates are added on enrollment.
All local media is sensitive; opt-in only. Not certified for real physical security.
"""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import public_research as research
import bioguard_core as core
import bioguard_matching as matching


def load_verifier(path='model/public_models.joblib'):
    model=Path(path)
    if not model.exists():
        raise FileNotFoundError(f'{model} not created yet. Run notebook training and export first.')
    obj=joblib.load(model)  # Load ONLY locally generated, trusted joblib files.
    if 'models' not in obj or 'Image Only' not in obj['models']:
        raise ValueError('Invalid verifier package')
    return obj


def gallery_path():
    return Path('model/enrollment_gallery.joblib')


def load_gallery(path=None):
    path=Path(path or gallery_path())
    if not path.is_file():
        return {'templates':{},'permissions':{},'meta':{}}
    data=joblib.load(path)   # Never accept untrusted uploads as joblib/pickle.
    if not all(x in data for x in ('templates','permissions','meta')):
        raise ValueError('Malformed enrollment gallery')
    # Version 1 galleries only stored means.  Keep them readable and let the
    # local matcher fall back to that mean until a controlled re-enrollment can
    # add individual sample vectors.
    data.setdefault('sample_vectors', {})
    return data


def validate_new_username(name):
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_\-]{2,40}',name):
        raise ValueError('Username: 2-40 Latin letters/digits/_/- only, e.g. ziad')
    return name


def enroll_user(name, image_paths, audio_paths, allowed=True, gallery_file=None, video_paths=None,
                fingerprint_paths=None, face_feature_extractor=None):
    """Enroll with >=4 real faces and >=4 real audio recordings (4 recommended).
    Adds a new template immediately; retains the public-trained verifier.
    Audio and image need not be frame-synchronized at enrollment, as both belong to same consented user.
    """
    name=validate_new_username(name)
    if len(image_paths)<4 or len(audio_paths)<4:
        raise ValueError('At least 4 distinct face photos AND 4 voice recordings required')
    # Every new face MUST be detected in un-cropped media.
    face_extractor = face_feature_extractor or matching.extract_face_live
    face=np.stack([face_extractor(Path(p)) for p in image_paths])
    voice=np.stack([matching.extract_voice(Path(p)) for p in audio_paths])
    templates={'image':face.mean(axis=0).astype('float32'),
               'voice':voice.mean(axis=0).astype('float32')}
    if video_paths:
        video=np.stack([research.video_vector(p) for p in video_paths])
        templates['video']=video.mean(axis=0).astype('float32')
    if fingerprint_paths:
        if len(fingerprint_paths)<4:
            raise ValueError('Fingerprint bonus requires >=4 separate scanner images')
        fp=np.stack([core.extract_fingerprint(p) for p in fingerprint_paths])
        templates['fingerprint']=fp.mean(axis=0).astype('float32')
    gallery_file=Path(gallery_file or gallery_path())
    gallery_file.parent.mkdir(parents=True,exist_ok=True)
    gallery=load_gallery(gallery_file)
    if name in gallery['templates']:
        raise ValueError('User already enrolled; use controlled re-enrollment separately')
    gallery['templates'][name]=templates
    # Preserve every validated enrollment vector.  The mean remains for
    # backwards compatibility with the public research/command helpers, while
    # Flask local matching uses these individual samples when present.
    gallery['sample_vectors'][name] = {
        'image': face.astype('float32'),
        'voice': voice.astype('float32'),
    }
    gallery['permissions'][name]=bool(allowed)
    gallery['meta'][name]={'created_utc':datetime.now(timezone.utc).isoformat(),
                          'photos':len(image_paths),'recordings':len(audio_paths),
                          'face_samples':int(len(face)), 'voice_samples':int(len(voice)),
                          'video_clips':len(video_paths or []),
                          'fingerprint_scans':len(fingerprint_paths or []), 'authorized':bool(allowed)}
    joblib.dump(gallery,gallery_file)
    return gallery['meta'][name]


def verify_new_attempt(face_image, voice_file, package=None, gallery=None,
                       fingerprint_image=None, require_fingerprint=False,
                       fingerprint_model_path='model/fingerprint_model.joblib',
                       face_feature_extractor=None):
    """Identity not manually selected. Face candidate, THEN voice match SAME candidate.
    Permissions checked after biometrics. Fail closed if gallery empty/missing sample.
    """
    package=package if package is not None else load_verifier()
    gallery=gallery if gallery is not None else load_gallery()
    if not gallery['templates']:
        return {'accepted':False,'identity':'Unknown','reason':'No users enrolled'}
    try:
        face_extractor = face_feature_extractor or matching.extract_face_live
        features={'image':face_extractor(face_image),
                  'voice':matching.extract_voice(voice_file)}
    except Exception as exc:
        return {'accepted':False,'identity':'Unknown','reason':'Invalid live face/voice: '+str(exc)}
    decision=research.identify_from_vectors(package,features,mode='Hierarchical',
            templates=gallery['templates'],permissions=gallery['permissions'])
    if not require_fingerprint or not decision['accepted']:
        return decision
    # BONUS fingerprint gate: requires both a real scan of the same registered
    # person AND a separately trained genuine/impostor traditional ML classifier.
    identity=decision['identity']
    if fingerprint_image is None or 'fingerprint' not in gallery['templates'].get(identity, {}):
        return {'accepted':False,'identity':'Unknown','reason':'Fingerprint required but not enrolled/scanned'}
    finger_path=Path(fingerprint_model_path)
    if not finger_path.is_file():
        return {'accepted':False,'identity':'Unknown','reason':'Fingerprint traditional ML model not trained'}
    try:
        finger_bundle=joblib.load(finger_path)  # locally trained trusted file only
        observed=core.extract_fingerprint(fingerprint_image)
        reference=gallery['templates'][identity]['fingerprint']
        pair=core.pair_difference(observed,reference)[None,:]
        score=float(research.proba(finger_bundle['model'],pair)[0])
        if score < float(finger_bundle['threshold']):
            return {'accepted':False,'identity':'Unknown','reason':'Fingerprint mismatch','fingerprint_score':score}
        decision['fingerprint_score']=score
        decision['reason']='Face, voice, and fingerprint accepted'
        return decision
    except Exception as exc:
        return {'accepted':False,'identity':'Unknown','reason':'Invalid fingerprint sample: '+str(exc)}


def export_summary(path='model/model_manifest.json'):
    model=load_verifier()
    summary={'version':model.get('version'),'models':{},'feature_config':model.get('feature_config'),
             'note':'Use model/public_models.joblib locally; do not upload untrusted pickle/joblib.'}
    for mode,data in model['models'].items():
        summary['models'][mode]={'model':data['model_name'],'threshold':data['threshold'],
                                  'modalities':list(data['modalities'])}
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary
