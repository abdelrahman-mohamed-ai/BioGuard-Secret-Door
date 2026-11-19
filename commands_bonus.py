"""Traditional isolated-word command recognition: MFCC/ZCR/RMS + SVM/RF.
Folder: commands/<word>/train|val|test/*.wav (>=5 examples for train, >=2 val/test).
Command prediction NEVER gives authorization; identity & permissions checked first.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import bioguard_core as core


def train_commands(root='commands',output='model/commands_model.joblib'):
    root=Path(root)
    words=sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if len(words)<2:
        print('BONUS PENDING: need >=2 words in commands/<word>/train|val|test/*.wav')
        return None
    arrays={}
    for split in ('train','val','test'):
        xx,yy=[],[]
        for word in words:
            files=sorted((root/word/split).glob('*.wav'))
            minimum=5 if split=='train' else 2
            if len(files)<minimum:
                print(f'BONUS PENDING: {root/word/split} needs {minimum} WAV recordings')
                return None
            xx += [core.extract_voice(f) for f in files]
            yy += [word]*len(files)
        arrays[split]=(np.stack(xx),np.asarray(yy))
    models={'SVM':Pipeline([('scale',StandardScaler()),('model',SVC(C=1.5,kernel='rbf'))]),
            'Random Forest':RandomForestClassifier(n_estimators=120,random_state=42)}
    rows=[]; fitted={}
    for name,model in models.items():
        model.fit(*arrays['train']);fitted[name]=model
        val=accuracy_score(arrays['val'][1],model.predict(arrays['val'][0]))
        test=accuracy_score(arrays['test'][1],model.predict(arrays['test'][0]))
        rows.append({'model':name,'validation_accuracy':val,'test_accuracy':test})
    df=pd.DataFrame(rows)
    best=df.sort_values(['validation_accuracy','model'],ascending=[False,True]).iloc[0]['model']
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':fitted[best],'commands':words},output)
    df.to_csv(Path(output).parent/'commands_comparison.csv',index=False)
    print('Selection based on validation, test untouched for selection.\n',df.to_string(index=False))
    return df


def authorized_command(identity,command,verified,access_policy):
    """The action must be checked SERVER SIDE, after biometric verification."""
    return bool(verified and identity in access_policy and command in access_policy[identity])

if __name__=='__main__':train_commands()
