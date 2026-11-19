"""Automatic open-set recognition + independent identity verification.

Education only. In UI there is NO manually claimed name. A candidate is proposed
from biometric evidence then checked against the reference identity templates.
No deep learning or pretrained recognition embeddings are used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score
import bioguard_core as bg


def _claim_scores(package, vectors, claim):
    """Each claim uses the training-time pair difference & trained binary models."""
    comparisons = {m: bg.pair_difference(vec, package['templates'][claim][m])
                   for m, vec in vectors.items() if m in package['templates'][claim]}
    out = {}
    for approach in ('Face Only', 'Voice Only', 'Early Fusion', 'Fingerprint Only', 'Early Fusion (3)'):
        if approach not in package['models']:
            continue
        entry = package['models'][approach]
        if not all(m in comparisons for m in entry['modalities']):
            continue
        features = np.concatenate([comparisons[m] for m in entry['modalities']])[None, :]
        out[approach] = float(bg.score_model(entry['model'], features)[0])
    if 'Face Only' in out and 'Voice Only' in out:
        out['Late Fusion'] = .5 * (out['Face Only'] + out['Voice Only'])
    return out


def rank_face_candidates(package, face_image=None, face_features=None):
    """Rank enrolled identities for live webcam. Not an authentication decision."""
    if face_features is None:
        if face_image is None:
            raise ValueError('Face image is required')
        face_features = bg.extract_face(face_image)
    vectors = {'face': face_features}
    ranked = []
    for name in sorted(package['templates']):
        score = _claim_scores(package, vectors, name)['Face Only']
        ranked.append((name, float(score)))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    threshold = float(package['models']['Face Only']['threshold'])
    best = ranked[0]
    return {'candidate': best[0] if best[1] >= threshold else None,
            'face_score': best[1], 'face_threshold': threshold,
            'face_pass': bool(best[1] >= threshold), 'ranking': ranked}


def identify_and_verify(package, face_image=None, voice_wav=None, fingerprint_image=None,
                        mode='Hierarchical', face_features=None, voice_features=None,
                        fingerprint_features=None, authorized_users=None):
    """No user chooses their name. Output is a known name or Unknown.

    Hierarchical: face picks a candidate and must pass, then voice MUST pass for
    that *same candidate*. Early/Late Fusion: compute score for each registered
    claim, pick best, and apply validation-calibrated binary threshold.
    A good score is NOT a security guarantee on unseen outsiders.
    """
    try:
        if face_features is None:
            face_features = bg.extract_face(face_image)
        if voice_features is None:
            voice_features = bg.extract_voice(voice_wav)
        vectors = {'face': face_features, 'voice': voice_features}
        if mode in ('Hierarchical (3)', 'Early Fusion (3)'):
            if not package.get('use_fingerprint'):
                return {'accepted': False, 'identity': 'Unknown', 'reason': 'Fingerprint models not trained'}
            if fingerprint_features is None:
                if fingerprint_image is None:
                    return {'accepted': False, 'identity': 'Unknown', 'reason': 'Fingerprint scan missing'}
                fingerprint_features = bg.extract_fingerprint(fingerprint_image)
            vectors['fingerprint'] = fingerprint_features
    except Exception as exc:
        return {'accepted': False, 'identity': 'Unknown', 'reason': f'Invalid sample: {exc}'}

    all_scores = {name: _claim_scores(package, vectors, name) for name in sorted(package['templates'])}
    face_sorted = sorted(all_scores, key=lambda name: all_scores[name]['Face Only'], reverse=True)
    name = face_sorted[0]
    stages = {}
    def passed(claim, approach):
        score = float(all_scores[claim][approach])
        threshold = (float(package['late_threshold']) if approach == 'Late Fusion'
                     else float(package['models'][approach]['threshold']))
        stages[approach] = {'score': score, 'threshold': threshold,
                            'pass': bool(score >= threshold)}
        return score >= threshold

    if mode in ('Hierarchical', 'Hierarchical (3)'):
        if not passed(name, 'Face Only'):
            return {'accepted': False, 'identity': 'Unknown', 'reason': 'Face not recognized', 'stages': stages}
        if not passed(name, 'Voice Only'):
            return {'accepted': False, 'identity': 'Unknown', 'reason': 'Voice does not match face candidate', 'stages': stages}
        if mode == 'Hierarchical (3)' and not passed(name, 'Fingerprint Only'):
            return {'accepted': False, 'identity': 'Unknown', 'reason': 'Fingerprint mismatch', 'stages': stages}
    elif mode in ('Early Fusion', 'Late Fusion', 'Early Fusion (3)'):
        # The fused score, not face score, chooses the identity in these modes.
        name = max(all_scores, key=lambda claim: all_scores[claim][mode])
        if not passed(name, mode):
            return {'accepted': False, 'identity': 'Unknown', 'reason': f'{mode} rejected', 'stages': stages}
    else:
        raise ValueError(f'Unknown mode: {mode}')
    # Permissions are configured by operator, not a client-side dropdown.
    if authorized_users is not None and name not in authorized_users:
        return {'accepted': False, 'identity': 'Unknown', 'reason': 'Identified but not authorized', 'stages': stages}
    return {'accepted': True, 'identity': name, 'reason': 'Registered and verified', 'stages': stages}


def evaluate_open_set(package, known_root='data', unknown_root='data_unknown',
                      modes=('Hierarchical', 'Early Fusion', 'Late Fusion')):
    """Compare known subjects (test split only) vs totally unregistered visitors.

    The unknown test subjects must not be enrolled or trained and must not be
    used for model/threshold selection. Results appear ONLY after real data.
    """
    known = bg.scan_dataset(known_root)
    known = known[(known.split == 'test') & (known.user.isin(package['templates']))]
    unknown = bg.scan_dataset(unknown_root)
    unknown = unknown[unknown.split == 'test']
    if known.empty:
        raise ValueError('No registered test samples found in data/<user>/test/')
    if unknown.empty:
        raise ValueError('No UNKNOWN test samples: collect real outsiders under data_unknown/<user>/test/')
    if any(person in package['templates'] for person in unknown.user):
        raise ValueError('Unknown test subjects overlap with registered users!')
    rows = []
    # Test on the identical images and recordings for fair comparison.
    for row in pd.concat([known, unknown], ignore_index=True).itertuples(index=False):
        truth = row.user if row.user in package['templates'] else 'Unknown'
        # Compute features only once per attempt, reuse across all modes.
        try:
            face = bg.extract_face(row.face)
            voice = bg.extract_voice(row.voice)
        except Exception as exc:
            for mode in modes:
                rows.append({'mode': mode, 'person': row.user, 'truth': truth,
                             'predicted': 'Unknown', 'accepted': False,
                             'reason': f'Capture invalid: {exc}'})
            continue
        for mode in modes:
            pred = identify_and_verify(package, mode=mode,
                                       face_features=face, voice_features=voice)
            rows.append({'mode': mode, 'person': row.user, 'truth': truth,
                         'predicted': pred['identity'], 'accepted': pred['accepted'],
                         'reason': pred['reason']})
    details = pd.DataFrame(rows)
    metrics = []
    for mode, part in details.groupby('mode', sort=False):
        known_part = part[part.truth != 'Unknown']
        unknown_part = part[part.truth == 'Unknown']
        labels = sorted(package['templates']) + ['Unknown']
        metrics.append({'mode': mode,
                        'known_correct_id_rate': float((known_part.predicted == known_part.truth).mean()),
                        'known_reject_rate': float((known_part.predicted == 'Unknown').mean()),
                        'unknown_reject_rate': float((unknown_part.predicted == 'Unknown').mean()),
                        'unknown_false_accept_rate': float((unknown_part.predicted != 'Unknown').mean()),
                        'overall_open_set_accuracy': float(accuracy_score(part.truth, part.predicted)),
                        'known_test_attempts': len(known_part),
                        'unknown_test_attempts': len(unknown_part)})
    return details, pd.DataFrame(metrics)
