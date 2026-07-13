# Run this diagnostic to check for leakage
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ── Load data (mirrors 03_classify_structural.py setup) ────────────────────
TIMESERIES_DIR = os.path.expanduser('~/oasis-scripts/processed_timeseries')
FEATURES_DIR   = os.path.expanduser('~/oasis-scripts/features')

hipp_df   = pd.read_csv(os.path.join(FEATURES_DIR, 'hippocampal_features.csv'))
subj_df   = pd.read_csv(os.path.join(FEATURES_DIR, 'subject_index.csv'))
label_map = {'HC': 0, 'AD': 1}

binary_df = subj_df[subj_df['label'].isin(['HC', 'AD'])].reset_index(drop=True)
binary_df = binary_df[binary_df['session_id'].isin(hipp_df['session_id'])].reset_index(drop=True)

ts_list, labels, sids = [], [], []
for _, row in binary_df.iterrows():
    path = os.path.join(TIMESERIES_DIR, f"{row['session_id']}_timeseries.npy")
    if os.path.exists(path):
        ts = np.load(path)
        ts_list.append(ts)
        labels.append(label_map[row['label']])
        sids.append(row['session_id'])

min_p = min(ts.shape[1] for ts in ts_list)
y     = np.array(labels)

def compute_fc(ts):
    c = np.corrcoef(ts[:, :min_p].T)
    np.fill_diagonal(c, 0)
    return c

fc_all = np.array([compute_fc(ts) for ts in ts_list])
triu   = np.triu_indices(min_p, k=1)
X_fc   = np.array([fc[triu] for fc in fc_all])

subject_order = pd.DataFrame({'session_id': sids})
hipp_matched  = subject_order.merge(hipp_df, on='session_id', how='left')
X_hipp        = hipp_matched[['hipp_L_norm', 'hipp_R_norm', 'hipp_total_norm']].values

print(f"Loaded {len(y)} subjects | HC={sum(y==0)}, AD={sum(y==1)}")
print(f"X_fc: {X_fc.shape}, X_hipp: {X_hipp.shape}\n")

# ── Diagnostic ─────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr, te) in enumerate(skf.split(X_fc, y)):
    Xtr_hipp, Xte_hipp = X_hipp[tr], X_hipp[te]
    ytr, yte = y[tr], y[te]
    
    # Check hippocampal distributions per fold
    hc_mean = Xtr_hipp[ytr==0, 2].mean()  # total hippocampal, train HC
    ad_mean = Xtr_hipp[ytr==1, 2].mean()  # total hippocampal, train AD
    te_hc_mean = Xte_hipp[yte==0, 2].mean()
    te_ad_mean = Xte_hipp[yte==1, 2].mean()
    
    print(f"Fold {fold+1}:")
    print(f"  Train HC: {hc_mean:.6f}, Train AD: {ad_mean:.6f}")
    print(f"  Test HC:  {te_hc_mean:.6f}, Test AD:  {te_ad_mean:.6f}")
    print(f"  Train gap: {hc_mean - ad_mean:.6f}")
    print(f"  Test gap:  {te_hc_mean - te_ad_mean:.6f}")
    print()