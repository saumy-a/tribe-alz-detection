"""
Feature Extraction Pipeline — v3
==================================
Fix: use intersection of parcels present in ALL subjects
instead of requiring exactly 100 parcels.
This recovers subjects with partial brain coverage.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# CONFIG
# ============================================================
TIMESERIES_DIR = os.path.expanduser('~/oasis-scripts/processed_timeseries')
LABELS_CSV     = os.path.expanduser('~/oasis-scripts/pilot_90_subjects.csv')
OUTPUT_DIR     = os.path.expanduser('~/oasis-scripts/features')

os.makedirs(OUTPUT_DIR, exist_ok=True)

label_map = {'HC': 0, 'MCI': 1, 'AD': 2}

# ============================================================
# STEP 1 — Load labels
# ============================================================
labels_df = pd.read_csv(LABELS_CSV)
labels_df.columns = labels_df.columns.str.strip()

records = []
for _, row in labels_df.iterrows():
    session_id = row['session_id']
    npy_path   = os.path.join(TIMESERIES_DIR, f'{session_id}_timeseries.npy')
    if os.path.exists(npy_path) and row['label'] in label_map:
        records.append({
            'session_id': session_id,
            'subject_id': row['OASISID'],
            'label':      row['label'],
            'label_int':  label_map[row['label']],
            'npy_path':   npy_path
        })

df = pd.DataFrame(records).reset_index(drop=True)
print(f"Found {len(df)} subjects with timeseries + labels")

# ============================================================
# STEP 2 — Find minimum parcel count across all subjects
# ============================================================
parcel_counts = []
for _, row in df.iterrows():
    ts = np.load(row['npy_path'])
    parcel_counts.append(ts.shape[1])

min_parcels = min(parcel_counts)
print(f"Parcel counts: min={min_parcels}, max={max(parcel_counts)}, "
      f"most common={pd.Series(parcel_counts).mode()[0]}")
print(f"Using first {min_parcels} parcels for all subjects (common coverage)\n")

# ============================================================
# STEP 3 — Load all subjects using min_parcels columns
# ============================================================
def compute_fc(timeseries):
    corr = np.corrcoef(timeseries.T)
    np.fill_diagonal(corr, 0)
    return corr

print("Computing FC matrices...")
fc_matrices   = []
valid_records = []

for _, row in df.iterrows():
    ts = np.load(row['npy_path'])
    ts = ts[:, :min_parcels]          # trim to common parcel count
    fc = compute_fc(ts)
    fc_matrices.append(fc)
    valid_records.append(row)

df          = pd.DataFrame(valid_records).reset_index(drop=True)
fc_matrices = np.array(fc_matrices)

print(f"All {len(df)} subjects loaded")
print(df['label'].value_counts())
print(f"FC matrices shape: {fc_matrices.shape}")

# ============================================================
# STEP 4 — Healthy brain template
# ============================================================
hc_idx      = df[df['label'] == 'HC'].index.tolist()
hc_matrices = fc_matrices[hc_idx]
hc_mean     = np.mean(hc_matrices, axis=0)
hc_std      = np.std(hc_matrices,  axis=0)
hc_std      = np.where(hc_std < 1e-8, 1e-8, hc_std)

print(f"\nHealthy template built from {len(hc_idx)} HC subjects")
np.save(os.path.join(OUTPUT_DIR, 'hc_mean_fc.npy'), hc_mean)
np.save(os.path.join(OUTPUT_DIR, 'hc_std_fc.npy'),  hc_std)

# ============================================================
# STEP 5 — Extract features
# ============================================================
n_parcels = min_parcels
triu_idx  = np.triu_indices(n_parcels, k=1)
n_features = len(triu_idx[0])
print(f"Upper triangle features: {n_features} per subject")

X_baseline  = []
X_deviation = []
X_combined  = []

for fc in fc_matrices:
    deviation = (fc - hc_mean) / hc_std
    fc_vec    = fc[triu_idx]
    dev_vec   = deviation[triu_idx]
    X_baseline.append(fc_vec)
    X_deviation.append(dev_vec)
    X_combined.append(np.concatenate([fc_vec, dev_vec]))

X_baseline  = np.array(X_baseline)
X_deviation = np.array(X_deviation)
X_combined  = np.array(X_combined)
y           = df['label_int'].values

print(f"\nFeature shapes:")
print(f"  Baseline  (FC only):        {X_baseline.shape}")
print(f"  Deviation (TRIBE-inspired): {X_deviation.shape}")
print(f"  Combined  (FC + Dev):       {X_combined.shape}")
print(f"  Labels: HC={sum(y==0)}, MCI={sum(y==1)}, AD={sum(y==2)}")

# ============================================================
# STEP 6 — PCA
# ============================================================
def reduce_features(X, n_components=0.95, label=''):
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA(n_components=n_components, svd_solver='full', random_state=42)
    X_red  = pca.fit_transform(X_sc)
    print(f"  {label}: {X.shape[1]} → {X_red.shape[1]} components")
    return X_red

print("\nPCA (95% variance):")
X_base_pca = reduce_features(X_baseline,  label='Baseline')
X_dev_pca  = reduce_features(X_deviation, label='Deviation')
X_comb_pca = reduce_features(X_combined,  label='Combined')

# ============================================================
# STEP 7 — Save
# ============================================================
np.save(os.path.join(OUTPUT_DIR, 'X_baseline.npy'),      X_base_pca)
np.save(os.path.join(OUTPUT_DIR, 'X_deviation.npy'),     X_dev_pca)
np.save(os.path.join(OUTPUT_DIR, 'X_combined.npy'),      X_comb_pca)
np.save(os.path.join(OUTPUT_DIR, 'X_baseline_raw.npy'),  X_baseline)
np.save(os.path.join(OUTPUT_DIR, 'X_deviation_raw.npy'), X_deviation)
np.save(os.path.join(OUTPUT_DIR, 'y_labels.npy'),        y)
np.save(os.path.join(OUTPUT_DIR, 'n_parcels_used.npy'),  np.array([n_parcels]))

df[['session_id','subject_id','label','label_int']].to_csv(
    os.path.join(OUTPUT_DIR, 'subject_index.csv'), index=False)

print(f"\n✅ Done. {len(df)} subjects, {n_parcels} parcels, {n_features} connections")
print(f"Files saved to: {OUTPUT_DIR}")