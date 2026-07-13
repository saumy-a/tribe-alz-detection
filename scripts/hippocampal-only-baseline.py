import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Load data (mirrors 03_classify_structural.py setup) ────────────────────
FEATURES_DIR = os.path.expanduser('~/oasis-scripts/features')

hipp_df   = pd.read_csv(os.path.join(FEATURES_DIR, 'hippocampal_features.csv'))
label_map = {'HC': 0, 'AD': 1}

X_hipp = hipp_df[['hipp_L_norm', 'hipp_R_norm', 'hipp_total_norm']].values
y      = np.array([label_map[l] for l in hipp_df['label']])

print(f"Loaded {len(y)} subjects | HC={sum(y==0)}, AD={sum(y==1)}\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, aucs = [], []

for tr, te in skf.split(X_hipp, y):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_hipp[tr])
    Xte = sc.transform(X_hipp[te])
    
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(Xtr, y[tr])
    
    pred = lda.predict(Xte)
    prob = lda.predict_proba(Xte)[:, 1]
    
    accs.append(accuracy_score(y[te], pred))
    aucs.append(roc_auc_score(y[te], prob))

print(f"Hippocampal volume ONLY — LDA")
print(f"  Acc={np.mean(accs):.3f}±{np.std(accs):.3f}")
print(f"  AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")