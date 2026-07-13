"""
Final Ablation Study + Definitive Results
==========================================
Three ablation conditions:
  A. Structural only    — hippocampal volume alone
  B. FC only            — resting-state FC alone (no structural)
  C. FC + Structural    — multimodal baseline
  D. Deviation + Structural — TRIBE-inspired + hippocampal (your method)
  E. Combined + Structural  — FC + deviation + hippocampal

This tells you exactly what each component contributes.
"""

import os, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings('ignore')

TIMESERIES_DIR = os.path.expanduser('~/oasis-scripts/processed_timeseries')
FEATURES_DIR   = os.path.expanduser('~/oasis-scripts/features')
RESULTS_DIR    = os.path.expanduser('~/oasis-scripts/results')
os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 42
label_names  = ['HC', 'AD']

# ============================================================
# STEP 1 — Load data
# ============================================================
hipp_df   = pd.read_csv(os.path.join(FEATURES_DIR, 'hippocampal_features.csv'))
subj_df   = pd.read_csv(os.path.join(FEATURES_DIR, 'subject_index.csv'))
label_map = {'HC': 0, 'AD': 1}

binary_df = subj_df[subj_df['label'].isin(['HC','AD'])].reset_index(drop=True)
binary_df = binary_df[binary_df['session_id'].isin(hipp_df['session_id'])].reset_index(drop=True)

ts_list, labels, sids = [], [], []
for _, row in binary_df.iterrows():
    path = os.path.join(TIMESERIES_DIR, f"{row['session_id']}_timeseries.npy")
    if os.path.exists(path):
        ts_list.append(np.load(path))
        labels.append(label_map[row['label']])
        sids.append(row['session_id'])

min_p  = min(t.shape[1] for t in ts_list)
y      = np.array(labels)

def compute_fc(ts):
    c = np.corrcoef(ts[:, :min_p].T)
    np.fill_diagonal(c, 0)
    return c

fc_all = np.array([compute_fc(t) for t in ts_list])
triu   = np.triu_indices(min_p, k=1)
X_fc   = np.array([f[triu] for f in fc_all])

subject_order = pd.DataFrame({'session_id': sids})
hipp_matched  = subject_order.merge(hipp_df, on='session_id', how='left')
X_hipp        = hipp_matched[['hipp_L_norm','hipp_R_norm','hipp_total_norm']].values

print(f"Subjects: HC={sum(y==0)}, AD={sum(y==1)}")
print(f"FC features: {X_fc.shape[1]}, Structural features: {X_hipp.shape[1]}\n")

# ============================================================
# STEP 2 — Build feature sets per fold
# ============================================================
def build_fold_features(X_fc_tr, X_fc_te, X_hipp_tr, X_hipp_te, y_tr, mode):
    # Structural: always standardize
    sc_h = StandardScaler()
    H_tr = sc_h.fit_transform(X_hipp_tr)
    H_te = sc_h.transform(X_hipp_te)

    # HC template for deviation
    hc    = X_fc_tr[y_tr == 0]
    mu    = hc.mean(axis=0)
    sigma = hc.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1e-8, sigma)
    D_tr  = (X_fc_tr - mu) / sigma
    D_te  = (X_fc_te  - mu) / sigma

    # Standardized FC
    sc_f  = StandardScaler()
    F_tr  = sc_f.fit_transform(X_fc_tr)
    F_te  = sc_f.transform(X_fc_te)

    if mode == 'structural_only':
        return H_tr, H_te
    elif mode == 'fc_only':
        return F_tr, F_te
    elif mode == 'fc_structural':
        return np.c_[F_tr, H_tr], np.c_[F_te, H_te]
    elif mode == 'deviation_structural':
        return np.c_[D_tr, H_tr], np.c_[D_te, H_te]
    elif mode == 'combined_structural':
        return np.c_[F_tr, D_tr, H_tr], np.c_[F_te, D_te, H_te]

def run_cv(mode, clf_template, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s, aucs = [], [], []
    all_true, all_pred, all_prob = [], [], []

    for tr, te in skf.split(X_fc, y):
        Xtr, Xte = build_fold_features(
            X_fc[tr], X_fc[te], X_hipp[tr], X_hipp[te], y[tr], mode)
        ytr, yte = y[tr], y[te]

        clf = copy.deepcopy(clf_template)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        prob = clf.predict_proba(Xte)[:, 1]

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
# STEP 3 — Run ablation (LDA only — most honest for small N)
# ============================================================
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

ablation_modes = {
    'A. Structural only':              'structural_only',
    'B. FC only':                      'fc_only',
    'C. FC + Structural':              'fc_structural',
    'D. Deviation + Structural':       'deviation_structural',
    'E. Combined + Structural (Ours)': 'combined_structural',
}

print("="*65)
print("ABLATION STUDY (LDA, 5-fold CV, HC vs AD)")
print("="*65)

ablation_results = {}
for name, mode in ablation_modes.items():
    res = run_cv(mode, lda)
    ablation_results[name] = res
    print(f"{name}")
    print(f"  Acc={res['acc_mean']:.3f}±{res['acc_std']:.3f}  "
          f"F1={res['f1_mean']:.3f}±{res['f1_std']:.3f}  "
          f"AUC={res['auc_mean']:.3f}±{res['auc_std']:.3f}\n")

# ============================================================
# STEP 4 — Save ablation table
# ============================================================
rows = []
for name, res in ablation_results.items():
    rows.append({
        'Condition': name,
        'Accuracy':  f"{res['acc_mean']:.3f} ± {res['acc_std']:.3f}",
        'F1':        f"{res['f1_mean']:.3f} ± {res['f1_std']:.3f}",
        'AUC':       f"{res['auc_mean']:.3f} ± {res['auc_std']:.3f}",
        'auc_val':   res['auc_mean'],
        'acc_val':   res['acc_mean'],
    })
abl_df = pd.DataFrame(rows)
abl_df.to_csv(os.path.join(RESULTS_DIR, 'ablation_results.csv'), index=False)

# ============================================================
# STEP 5 — Figure: ROC curves for all ablation conditions
# ============================================================
colors = ['#888888','#4C72B0','#9467BD','#DD8452','#2CA02C']
fig, ax = plt.subplots(figsize=(8, 6))
for (name, mode), color in zip(ablation_modes.items(), colors):
    res = ablation_results[name]
    fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name} (AUC={res['auc_mean']:.3f})")
ax.plot([0,1],[0,1],'k--', alpha=0.4, label='Chance')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Ablation Study ROC Curves — HC vs AD (LDA)', fontsize=12)
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_ablation_roc.pdf'), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, 'fig_ablation_roc.png'), dpi=300)
plt.close()
print("Figure saved: ablation ROC curves")

# ============================================================
# STEP 6 — Figure: Bar chart ablation
# ============================================================
short_labels = ['Structural\nonly','FC\nonly','FC+\nStruct','Dev+\nStruct','Combined\n(Ours)']
fig, ax = plt.subplots(figsize=(10, 5))
x, w = np.arange(5), 0.25
for i,(m,c,ml) in enumerate(zip(
        ['acc_val','f1_val','auc_val'],
        ['#4C72B0','#DD8452','#55A868'],
        ['Accuracy','F1','AUC'])):
    ax.bar(x+i*w, abl_df[m.replace("f1_val","f1_mean").replace("acc_val","acc_mean").replace("auc_val","auc_mean")].values, w, label=ml, color=c, alpha=0.85)
ax.set_xticks(x+w)
ax.set_xticklabels(short_labels, fontsize=10)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color='grey', linestyle='--', alpha=0.5, label='Chance')
ax.set_title('Ablation Study — Feature Contribution (LDA, 5-fold CV)', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_ablation_bar.pdf'), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, 'fig_ablation_bar.png'), dpi=300)
plt.close()
print("Figure saved: ablation bar chart")

# ============================================================
# STEP 7 — Confusion matrix for best condition
# ============================================================
best_name = abl_df.loc[abl_df['auc_val'].idxmax(), 'Condition']
best_res  = ablation_results[best_name]
cm = confusion_matrix(best_res['y_true'], best_res['y_pred'])
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names, ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
ax.set_title(f'Confusion Matrix\n{best_name}', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_best_confusion.pdf'), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, 'fig_best_confusion.png'), dpi=300)
plt.close()
print(f"Figure saved: confusion matrix ({best_name})")

print(f"\n✅ All ablation results saved to: {RESULTS_DIR}")
print(f"\nKey insight for paper:")
print(f"  FC only AUC:           {ablation_results['B. FC only']['auc_mean']:.3f}")
print(f"  Structural only AUC:   {ablation_results['A. Structural only']['auc_mean']:.3f}")
print(f"  FC + Structural AUC:   {ablation_results['C. FC + Structural']['auc_mean']:.3f}")
print(f"  Dev + Structural AUC:  {ablation_results['D. Deviation + Structural']['auc_mean']:.3f}")
print(f"  Combined AUC:          {ablation_results['E. Combined + Structural (Ours)']['auc_mean']:.3f}")