"""
OASIS-3 Resting-State fMRI Preprocessing Pipeline
====================================================
TR = 2.2s, Schaefer-100 atlas, BIDS structure
Fixed: padlen error + mismatched run shapes
"""

import os
import glob
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BOLD_DIR   = os.path.expanduser('~/oasis-scripts/downloaded_bold')
OUTPUT_DIR = os.path.expanduser('~/oasis-scripts/processed_timeseries')
SESSIONS_CSV = os.path.expanduser('~/oasis-scripts/pilot_90_sessions.csv')
TR = 2.2
N_PARCELS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1 — Atlas
# ============================================================
print("Fetching Schaefer-100 atlas...")
atlas = fetch_atlas_schaefer_2018(n_rois=N_PARCELS, yeo_networks=7, resolution_mm=2)
atlas_img    = atlas['maps']
print(f"Atlas loaded: {N_PARCELS} parcels\n")

# ============================================================
# STEP 2 — Masker WITHOUT bandpass (we apply it manually per-run)
# This avoids the padlen error for short scans
# ============================================================
masker = NiftiLabelsMasker(
    labels_img=atlas_img,
    standardize=True,
    detrend=True,
    t_r=TR,
    memory='nilearn_cache',
    verbose=0
)

def apply_bandpass(ts, tr, low_pass=0.1, high_pass=0.01):
    """Apply bandpass filter only if scan is long enough."""
    from nilearn.signal import clean
    min_timepoints = 50  # safe minimum for filtering
    if ts.shape[0] < min_timepoints:
        return None  # too short, reject this run
    try:
        filtered = clean(ts, t_r=tr, low_pass=low_pass,
                         high_pass=high_pass, standardize=False)
        return filtered
    except Exception:
        return None  # filter failed, reject

# ============================================================
# STEP 3 — Process all sessions
# ============================================================
sessions_df = pd.read_csv(SESSIONS_CSV)
session_ids = sessions_df['session_id'].tolist()

print(f"Processing {len(session_ids)} sessions...\n")

qc_log = []
success_count = 0
fail_count = 0

for i, session_id in enumerate(session_ids):
    session_path = os.path.join(BOLD_DIR, session_id)

    if not os.path.isdir(session_path):
        print(f"[{i+1}/{len(session_ids)}] SKIP — missing folder: {session_id}")
        qc_log.append({'session_id': session_id, 'status': 'missing_folder', 'n_timepoints': None})
        fail_count += 1
        continue

    bold_files = sorted(glob.glob(
        os.path.join(session_path, '**', '*task-rest*_bold.nii.gz'), recursive=True))

    if len(bold_files) == 0:
        print(f"[{i+1}/{len(session_ids)}] SKIP — no bold files: {session_id}")
        qc_log.append({'session_id': session_id, 'status': 'no_bold_files', 'n_timepoints': None})
        fail_count += 1
        continue

    try:
        # Process each run independently, filter, then keep valid ones
        valid_runs = []
        for bf in bold_files:
            ts_raw = masker.fit_transform(bf)   # extract parcel signals
            ts_filt = apply_bandpass(ts_raw, TR) # bandpass filter
            if ts_filt is not None:
                valid_runs.append(ts_filt)

        if len(valid_runs) == 0:
            print(f"[{i+1}/{len(session_ids)}] FAIL — all runs too short: {session_id}")
            qc_log.append({'session_id': session_id, 'status': 'all_runs_too_short', 'n_timepoints': 0})
            fail_count += 1
            continue

        # Concatenate valid runs along time axis
        # Only concatenate runs with same number of parcels
        parcel_counts = [r.shape[1] for r in valid_runs]
        most_common = max(set(parcel_counts), key=parcel_counts.count)
        valid_runs = [r for r in valid_runs if r.shape[1] == most_common]

        full_ts = np.concatenate(valid_runs, axis=0)

        # QC checks
        n_timepoints = full_ts.shape[0]
        has_nan = np.isnan(full_ts).any()
        is_flat = full_ts.std() < 0.01

        if n_timepoints < 100 or has_nan or is_flat:
            print(f"[{i+1}/{len(session_ids)}] FAIL QC — {session_id} "
                  f"(tp={n_timepoints}, nan={has_nan}, flat={is_flat})")
            qc_log.append({'session_id': session_id, 'status': 'failed_qc', 'n_timepoints': n_timepoints})
            fail_count += 1
            continue

        # Save
        out_path = os.path.join(OUTPUT_DIR, f'{session_id}_timeseries.npy')
        np.save(out_path, full_ts)

        print(f"[{i+1}/{len(session_ids)}] OK — {session_id} "
              f"(shape={full_ts.shape}, valid_runs={len(valid_runs)})")
        qc_log.append({'session_id': session_id, 'status': 'success', 'n_timepoints': n_timepoints})
        success_count += 1

    except Exception as e:
        print(f"[{i+1}/{len(session_ids)}] ERROR — {session_id}: {str(e)}")
        qc_log.append({'session_id': session_id, 'status': f'error: {str(e)}', 'n_timepoints': None})
        fail_count += 1

# ============================================================
# STEP 4 — Save QC log
# ============================================================
qc_df = pd.DataFrame(qc_log)
qc_df.to_csv(os.path.join(OUTPUT_DIR, 'qc_log.csv'), index=False)

print("\n" + "="*60)
print(f"DONE. {success_count} succeeded, {fail_count} failed.")
print(f"Files saved to: {OUTPUT_DIR}")
print("="*60)