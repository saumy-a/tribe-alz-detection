"""
Binary Classification: HC vs AD — v5 (with Structural Features)
===============================================================
- 52 subjects (4 excluded due to missing FreeSurfer data)
- Features: FC (2485) + Hippocampal volume (3)
- Classifiers: LDA, SVM-linear, LR, RF
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
TIMESERIES_DIR = os.path.expanduser('~/oasis-scripts/processed_timeseries')
FEATURES_DIR   = os.path.expanduser('~/oasis-scripts/features')
RESULTS_DIR    = os.path.expanduser('~/oasis-scripts/results')
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS     = 5
label_names  = ['HC', 'AD']

# ============================================================
# STEP 1 — Load hippocampal features (52 subjects)
# ============================================================
hipp_df = pd.read_csv(os.path.join(FEATURES_DIR, 'hippocampal_features.csv'))
print(f"Hippocampal features loaded: {len(hipp_df)} subjects")
print(f"HC={sum(hipp_df['label']=='HC')}, AD={sum(hipp_df['label']=='AD')}")

# ============================================================
# STEP 2 — Load timeseries, filter to matched subjects only
# ============================================================
subj_df   = pd.read_csv(os.path.join(FEATURES_DIR, 'subject_index.csv'))
label_map = {'HC': 0, 'AD': 1}

# Filter to HC and AD, then to matched subjects only
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

# Align parcel count
min_p = min(ts.shape[1] for ts in ts_list)
y     = np.array(labels)

def compute_fc(ts):
    c = np.corrcoef(ts[:, :min_p].T)
    np.fill_diagonal(c, 0)
    return c

fc_all = np.array([compute_fc(ts) for ts in ts_list])
triu   = np.triu_indices(min_p, k=1)
X_fc   = np.array([fc[triu] for fc in fc_all])

print(f"\nSubjects loaded: {len(ts_list)}, parcels: {min_p}")
print(f"HC={sum(y==0)}, AD={sum(y==1)}")
print(f"FC features: {X_fc.shape[1]}")

# ============================================================
# STEP 3 — Match hippocampal features to loaded subjects
# ============================================================
# Create dataframe with session_id in same order as ts_list
subject_order = pd.DataFrame({'session_id': sids})
hipp_matched = subject_order.merge(hipp_df, on='session_id', how='left')
X_hipp = hipp_matched[['hipp_L_norm', 'hipp_R_norm', 'hipp_total_norm']].values

print(f"Structural features: {X_hipp.shape[1]}")
print(f"Total features per subject: {X_fc.shape[1] + X_hipp.shape[1]}\n")

# ============================================================
# STEP 4 — Per-fold feature builder (with structural)
# ============================================================
def get_features(Xtr_fc, Xte_fc, Xtr_hipp, Xte_hipp, ytr, mode):
    """
    mode: 'baseline' = FC + structural (no deviation)
          'deviation' = deviation FC + structural
          'combined' = FC + deviation FC + structural
    """
    # Standardize structural features
    sc_hipp = StandardScaler()
    Xtr_hipp_s = sc_hipp.fit_transform(Xtr_hipp)
    Xte_hipp_s = sc_hipp.transform(Xte_hipp)
    
    if mode == 'baseline':
        # FC only (standardized) + structural
        sc_fc = StandardScaler()
        Xtr_fc_s = sc_fc.fit_transform(Xtr_fc)
        Xte_fc_s = sc_fc.transform(Xte_fc)
        return np.c_[Xtr_fc_s, Xtr_hipp_s], np.c_[Xte_fc_s, Xte_hipp_s]
    
    # Compute deviation on RAW FC
    hc = Xtr_fc[ytr == 0]
    mu = hc.mean(axis=0)
    sigma = hc.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1e-8, sigma)
    
    dtr = (Xtr_fc - mu) / sigma
    dte = (Xte_fc - mu) / sigma
    
    if mode == 'deviation':
        # Deviation FC (NOT re-standardized) + structural
        return np.c_[dtr, Xtr_hipp_s], np.c_[dte, Xte_hipp_s]
    
    # Combined: standardized FC + deviation + structural
    sc_fc = StandardScaler()
    Xtr_fc_s = sc_fc.fit_transform(Xtr_fc)
    Xte_fc_s = sc_fc.transform(Xte_fc)
    return np.c_[Xtr_fc_s, dtr, Xtr_hipp_s], np.c_[Xte_fc_s, dte, Xte_hipp_s]

def run_cv(mode, clf_template):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s, aucs = [], [], []
    all_true, all_pred, all_prob = [], [], []

    for tr, te in skf.split(X_fc, y):
        Xtr, Xte = get_features(X_fc[tr], X_fc[te], X_hipp[tr], X_hipp[te], y[tr], mode)
        ytr, yte = y[tr], y[te]

        # PCA on FC portion only (not structural)
        fc_dim = X_fc.shape[1]
        if mode == 'combined':
            fc_dim = X_fc.shape[1] * 2  # FC + deviation
        
        # Conservative PCA
        max_comp = max(5, len(ytr) // 5)
        n_comp = min(max_comp, fc_dim)
        
        if n_comp < fc_dim:
            pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
            Xtr_fc_pca = pca.fit_transform(Xtr[:, :fc_dim])
            Xte_fc_pca = pca.transform(Xte[:, :fc_dim])
            # Concatenate with structural features
            Xtr_final = np.c_[Xtr_fc_pca, Xtr[:, fc_dim:]]
            Xte_final = np.c_[Xte_fc_pca, Xte[:, fc_dim:]]
        else:
            Xtr_final, Xte_final = Xtr, Xte

        clf = copy.deepcopy(clf_template)
        clf.fit(Xtr_final, ytr)
        pred = clf.predict(Xte_final)
        prob = clf.predict_proba(Xte_final)[:, 1]

        accs.append(accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, zero_division=0))
        aucs.append(roc_auc_score(yte, prob))
        all_true.extend(yte)
        all_pred.extend(pred)
        all_prob.extend(prob)

    return {
        'acc_mean': np.mean(accs), 'acc_std': np.std(accs),
        'f1_mean':  np.mean(f1s),  'f1_std':  np.std(f1s),
        'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
        'y_true': np.array(all_true),
        'y_pred': np.array(all_pred),
        'y_prob': np.array(all_prob),
    }

# ============================================================
# STEP 5 — Run experiments
# ============================================================
experiments = {
    'Baseline (FC only)':         'baseline',
    'Deviation (TRIBE-inspired)': 'deviation',
    'Combined FC+Dev (Ours)':     'combined',
}

classifiers = {
    'LDA':   LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    'SVM':   SVC(kernel='linear', C=0.001, probability=True, random_state=RANDOM_STATE),
    'LR':    LogisticRegression(C=0.001, max_iter=2000, random_state=RANDOM_STATE),
    'RF':    RandomForestClassifier(n_estimators=100, max_depth=3,
                                     random_state=RANDOM_STATE),
}

all_results = {}
print("Running experiments (FC + Hippocampal Volume)...\n")

for feat_name, feat_mode in experiments.items():
    for clf_name, clf in classifiers.items():
        key = f"{feat_name} | {clf_name}"
        res = run_cv(feat_mode, clf)
        all_results[key] = res
        print(f"  {feat_name} | {clf_name}")
        print(f"    Acc={res['acc_mean']:.3f}±{res['acc_std']:.3f}  "
              f"F1={res['f1_mean']:.3f}±{res['f1_std']:.3f}  "
              f"AUC={res['auc_mean']:.3f}±{res['auc_std']:.3f}")

# ============================================================
# STEP 6 — Results table
# ============================================================
rows = []
for key, res in all_results.items():
    feat, clf = key.split(' | ')
    rows.append({
        'Features': feat, 'Classifier': clf,
        'Accuracy': f"{res['acc_mean']:.3f} ± {res['acc_std']:.3f}",
        'F1':       f"{res['f1_mean']:.3f} ± {res['f1_std']:.3f}",
        'AUC':      f"{res['auc_mean']:.3f} ± {res['auc_std']:.3f}",
        'acc_val':  res['acc_mean'],
        'f1_val':   res['f1_mean'],
        'auc_val':  res['auc_mean'],
    })

results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(RESULTS_DIR, 'binary_results_structural.csv'), index=False)

print("\n" + "="*75)
print("RESULTS TABLE (FC + Hippocampal Volume):")
print("="*75)
print(results_df[['Features','Classifier','Accuracy','F1','AUC']].to_string(index=False))

best = results_df.loc[results_df['auc_val'].idxmax()]
print(f"\n★ Best: {best['Features']} | {best['Classifier']}")
print(f"  Acc={best['Accuracy']}  F1={best['F1']}  AUC={best['AUC']}")

# ============================================================
# STEP 7 — Figures
# ============================================================
best_clf = best['Classifier']

# Fig 1 — Confusion matrix
best_key = f"{best['Features']} | {best_clf}"
br = all_results[best_key]
cm = confusion_matrix(br['y_true'], br['y_pred'])
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names, ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
ax.set_title(f'Confusion Matrix — {best_key}', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig1_confusion_matrix.pdf'), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, 'fig1_confusion_matrix.png'), dpi=300)
plt.close()

# Fig 2 — ROC curves
fig, ax = plt.subplots(figsize=(7,5))
colors = ['#4C72B0','#DD8452','#55A868']
for (feat_name, _), color in zip(experiments.items(), colors):
    key = f"{feat_name} | {best_clf}"
    res = all_results[key]
    fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{feat_name} (AUC={res['auc_mean']:.3f})")
ax.plot([0,1],[0,1],'k--',alpha=0.4,label='Chance')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'ROC Curves — HC vs AD ({best_clf})', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig2_roc_curves.pdf'), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, 'fig2_roc_curves.png'), dpi=300)
plt.close()

# Fig 3 — Bar chart
feat_labels = ['Baseline\n(FC+Hipp)', 'Deviation\n(TRIBE+Hipp)', 'Combined\n(Ours+Hipp)']
sub_df = results_df[results_df['Classifier']==best_clf].copy()
x, w = np.arange(3), 0.25
fig, ax = plt.subplots(figsize=(9,5))
for i,(m,c,ml) in enumerate(zip(
        ['acc_val','f1_val','auc_val'],
        ['#4C72B0','#DD8452','#55A868'],
        ['Accuracy','F1','AUC'])):
    ax.bar(x+i*w, sub_df[m].values, w, label=ml, color=c, alpha=0.85)
ax.set_xticks(x+w)
ax.set_xticklabels(feat_labels, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0,1.05)
ax.axhline(0.5, color='grey', linestyle='--', alpha=0.5, label='Chance')
ax.set_title(f'HC vs AD Classification ({best_clf}, 5-fold CV)', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig4_bar_chart.pdf'), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, 'fig4_bar_chart.png'), dpi=300)
plt.close()

print(f"\n✅ All saved to {RESULTS_DIR}")