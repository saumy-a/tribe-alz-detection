import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
FEATURES_DIR = os.path.expanduser('~/oasis-scripts/features')
FS_CSV_PATH = os.path.expanduser('~/oasis-scripts/download_freesurfer/oasis3_freesurfer_m.csv')  # UPDATE THIS

# ============================================================
# STEP 1: Load your 56 subjects
# ============================================================
subj_df = pd.read_csv(os.path.join(FEATURES_DIR, 'subject_index.csv'))
label_map = {'HC': 0, 'AD': 1}
binary_df = subj_df[subj_df['label'].isin(['HC', 'AD'])].reset_index(drop=True)
binary_df['freesurfer_id'] = binary_df['session_id'].str.replace('_MR_', '_Freesurfer53_')

print(f"Subjects to match: {len(binary_df)}")

# ============================================================
# STEP 2: Load FreeSurfer data
# ============================================================
fs_df = pd.read_csv(FS_CSV_PATH)

# Rename FS CSV's 'label' column to avoid conflict
if 'label' in fs_df.columns:
    fs_df = fs_df.rename(columns={'label': 'fs_label'})

# Merge
merged = binary_df.merge(fs_df, left_on='freesurfer_id', right_on='FS_FSDATA ID', how='left')

print(f"Matched subjects: {merged['Left-Hippocampus_volume'].notna().sum()}")
print(f"Unmatched: {merged['Left-Hippocampus_volume'].isna().sum()}")

# Show unmatched subjects
unmatched = merged[merged['Left-Hippocampus_volume'].isna()]
if len(unmatched) > 0:
    print("\nUnmatched subjects (will be excluded):")
    print(unmatched[['session_id', 'freesurfer_id', 'label']])

# Keep only matched subjects
matched = merged[merged['Left-Hippocampus_volume'].notna()].copy()
print(f"\nFinal matched subjects: {len(matched)}")

# ============================================================
# STEP 3: Compute normalized hippocampal features
# ============================================================
matched['hipp_L_norm'] = matched['Left-Hippocampus_volume'] / matched['IntraCranialVol']
matched['hipp_R_norm'] = matched['Right-Hippocampus_volume'] / matched['IntraCranialVol']
matched['hipp_total_norm'] = (matched['Left-Hippocampus_volume'] + matched['Right-Hippocampus_volume']) / matched['IntraCranialVol']

hipp_features = matched[['hipp_L_norm', 'hipp_R_norm', 'hipp_total_norm']].values

print(f"\nHippocampal features shape: {hipp_features.shape}")
print(f"HC mean total hippocampus: {matched[matched['label']=='HC']['hipp_total_norm'].mean():.6f}")
print(f"AD mean total hippocampus: {matched[matched['label']=='AD']['hipp_total_norm'].mean():.6f}")

# ============================================================
# STEP 4: Save
# ============================================================
hipp_df = matched[['session_id', 'label', 'hipp_L_norm', 'hipp_R_norm', 'hipp_total_norm']].copy()
hipp_df.to_csv(os.path.join(FEATURES_DIR, 'hippocampal_features.csv'), index=False)
print(f"\nSaved to: {os.path.join(FEATURES_DIR, 'hippocampal_features.csv')}")