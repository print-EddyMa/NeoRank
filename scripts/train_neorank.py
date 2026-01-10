"""
=====================================
NeoRank: Training Script
=====================================
Author: Eddy Ma
=====================================
"""

import argparse
import pandas as pd
import numpy as np
import os
import subprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, brier_score_loss)
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import tempfile
from pathlib import Path

warnings.filterwarnings('ignore')


def build_config(args):
    repo_root = Path(__file__).resolve().parents[1]
    default_folder = repo_root / 'data' / 'example'
    default_netmhc_dir = repo_root / 'tools'
    default_netmhc_path = default_netmhc_dir / 'netMHCpan-4.2' / 'Darwin_x86_64' / 'bin' / 'netMHCpan-4.2'

    class Config:
        FOLDER_PATH = str(args.folder) if args.folder else str(default_folder)
        EPITOPE_FILE = args.epitope if args.epitope else 'epitope.tsv'
        TCELL_FILE = args.tcell if args.tcell else 'tcell.tsv'
        NETMHCPAN_PATH = args.netmhcpan_path if args.netmhcpan_path else str(default_netmhc_path)
        NETMHCPAN_DIR = args.netmhcpan_dir if args.netmhcpan_dir else str(default_netmhc_dir)
        RANDOM_SEED = args.seed
        N_CV_FOLDS = args.n_folds
        N_ESTIMATORS = args.n_estimators

    return Config()


def main():
    parser = argparse.ArgumentParser(description='Train NeoRank model')
    parser.add_argument('--folder', help='Folder containing epitope and tcell files')
    parser.add_argument('--epitope', help='Epitope file name (default: epitope.tsv)')
    parser.add_argument('--tcell', help='Tcell file name (default: tcell.tsv)')
    parser.add_argument('--netmhcpan-path', help='Path to netMHCpan binary')
    parser.add_argument('--netmhcpan-dir', help='Directory with netMHCpan tool files')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-folds', type=int, default=10)
    parser.add_argument('--n-estimators', type=int, default=2000)

    args = parser.parse_args()
    config = build_config(args)

    np.random.seed(config.RANDOM_SEED)

    print("=" * 80)
    print("NEORANK: TRAINING PIPELINE")
    print("=" * 80)

    # Ensure folder exists
    if not os.path.isdir(config.FOLDER_PATH):
        raise SystemExit(f"Folder not found: {config.FOLDER_PATH}")

    print("\n[1/5] Loading datasets...")
    epitope_path = os.path.join(config.FOLDER_PATH, config.EPITOPE_FILE)
    tcell_path = os.path.join(config.FOLDER_PATH, config.TCELL_FILE)

    if not os.path.isfile(epitope_path) or not os.path.isfile(tcell_path):
        raise SystemExit(f"Required files not found in {config.FOLDER_PATH}."
                         " Place epitope.tsv and tcell.tsv there or pass --epitope/--tcell.")

    epitope = pd.read_csv(epitope_path, sep="\t", low_memory=False)
    tcell = pd.read_csv(tcell_path, sep="\t", low_memory=False)

    print(f"  - Loaded {len(epitope)} epitope records")
    print(f"  - Loaded {len(tcell)} T-cell assay records")

    # Prepare base dataset
    print("\n[2/5] Preparing base dataset...")

    t = tcell[['Epitope - IEDB IRI', 'Assay - Qualitative Measurement']].copy()
    t = t[t['Assay - Qualitative Measurement'].isin(['Positive', 'Negative'])]

    e = epitope[['Epitope ID - IEDB IRI', 'Epitope - Name']].copy()
    e = e.rename(columns={'Epitope ID - IEDB IRI': 'Epitope - IEDB IRI',
                          'Epitope - Name': 'peptide'})

    df = t.merge(e, on='Epitope - IEDB IRI', how='inner')
    df['immunogenic'] = df['Assay - Qualitative Measurement'].map({'Positive': 1, 'Negative': 0})

    df_agg = df.groupby(['Epitope - IEDB IRI', 'peptide']).agg({
        'immunogenic': lambda x: 1 if x.mean() >= 0.5 else 0
    }).reset_index()

    df = df_agg.copy()

    hla_df = tcell[['Epitope - IEDB IRI', 'MHC Restriction - Name']].dropna().drop_duplicates()
    hla_df = hla_df.rename(columns={'MHC Restriction - Name': 'HLA'})
    hla_df = hla_df.groupby('Epitope - IEDB IRI').first().reset_index()
    df = df.merge(hla_df, on='Epitope - IEDB IRI', how='left')
    df['HLA'] = df['HLA'].fillna('HLA-A*02:01')

    print(f"  - {len(df)} unique peptides after aggregation")
    print(f"  - Class distribution: {df['immunogenic'].value_counts().to_dict()}")

    # Mutation features
    print("\n[3/5] Extracting Mutation Features...")

    df['length'] = df['peptide'].str.len()

    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    for aa in amino_acids:
        df[f'aa_{aa}'] = df['peptide'].apply(lambda x: x.count(aa) / len(x) if len(x) > 0 else 0)

    hydro = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
             'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
             'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
             'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}

    def avg_hydro(seq):
        vals = [hydro.get(a, 0) for a in seq]
        return np.mean(vals) if vals else 0

    df['hydrophobicity'] = df['peptide'].apply(avg_hydro)

    charge = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.5}

    def net_charge(seq):
        return sum(charge.get(a, 0) for a in seq)

    df['net_charge'] = df['peptide'].apply(net_charge)

    def aromatic_content(seq):
        return sum(1 for a in seq if a in 'FWY') / len(seq) if len(seq) > 0 else 0

    df['aromatic_content'] = df['peptide'].apply(aromatic_content)

    polar = set('STNQCYWDEHKR')

    def polarity(seq):
        return sum(1 for a in seq if a in polar) / len(seq) if len(seq) > 0 else 0

    df['polarity'] = df['peptide'].apply(polarity)

    print(f"  ✓ Extracted mutation features")

    # Binding features (NetMHCpan)
    print("\n[4/5] Extracting Binding Features (NetMHCpan)...")

    def normalize_hla(hla_string):
        if pd.isna(hla_string):
            return None

        hla = hla_string.strip()

        invalid_patterns = ['class', 'DRB', 'DPB', 'DQB', 'DRA', 'DPA', 'DQA',
                           'DR1', 'DR3', 'DR4', 'DR7', 'DR9', 'DR11', 'DR15', 'DR53',
                           'DQ6', 'DP', 'Cw', 'DPw']
        if any(pattern in hla for pattern in invalid_patterns):
            return None

        if len(hla) <= 3:
            return None

        if hla.startswith('HLA-'):
            hla = hla[4:]

        if '*' in hla and ':' in hla:
            hla = hla.replace('*', '')
            return f'HLA-{hla}'

        if ':' in hla and len(hla) >= 6:
            return f'HLA-{hla}'

        return None

    print("  - Running NetMHCpan predictions...")

    unique_hlas = df['HLA'].unique()
    valid_hlas = []
    for hla in unique_hlas:
        normalized = normalize_hla(hla)
        if normalized:
            valid_hlas.append((hla, normalized))

    print(f"    Found {len(valid_hlas)} valid HLA alleles")

    netmhc_data = []
    original_dir = os.getcwd()

    # Switch to the NetMHCpan directory if it exists, otherwise run in place
    if os.path.isdir(config.NETMHCPAN_DIR):
        os.chdir(config.NETMHCPAN_DIR)

    for original_hla, hla_formatted in valid_hlas:
        hla_mask = df['HLA'] == original_hla
        hla_peptides = df[hla_mask][['peptide']].copy()
        hla_peptides['idx'] = df[hla_mask].index

        for length, group in hla_peptides.groupby(hla_peptides['peptide'].str.len()):
            if length < 8 or length > 14:
                continue

            if len(group) == 0:
                continue

            # write a temp file for this group
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpf:
                pep_file = tmpf.name
                for idx, row in group.iterrows():
                    tmpf.write(f"{row['peptide']}\n")

            try:
                env = os.environ.copy()
                env['NETMHCpan'] = str(config.NETMHCPAN_DIR)

                cmd = [
                    str(config.NETMHCPAN_PATH),
                    "-p", pep_file,
                    "-a", hla_formatted,
                    "-BA"
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

                if result.returncode != 0:
                    continue

                lines = result.stdout.split('\n')

                for line in lines:
                    if (not line.strip() or line.startswith('#') or
                        line.startswith('-') or line.startswith('Protein') or
                        ('Pos' in line and 'MHC' in line)):
                        continue

                    if line.strip() and not line.startswith('#'):
                        parts = line.split()

                        if len(parts) >= 15:
                            try:
                                peptide = parts[2]
                                rank_ba = parts[13]
                                aff_nm = parts[14]

                                if rank_ba == 'NA' or aff_nm == 'NA':
                                    continue

                                affinity_rank = float(rank_ba)

                                if '<=' in aff_nm:
                                    affinity_nM = 0.01
                                else:
                                    affinity_nM = float(aff_nm)

                                matching = group[group['peptide'] == peptide]
                                if not matching.empty:
                                    idx = matching.iloc[0]['idx']
                                    netmhc_data.append({
                                        'idx': idx,
                                        'HLA': original_hla,
                                        'affinity_nM': affinity_nM,
                                        'affinity_rank': affinity_rank
                                    })
                            except (ValueError, IndexError):
                                continue

            except (subprocess.TimeoutExpired, Exception):
                continue
            finally:
                if os.path.exists(pep_file):
                    os.remove(pep_file)

    os.chdir(original_dir)

    if netmhc_data:
        netmhc_df = pd.DataFrame(netmhc_data)
        netmhc_df = netmhc_df.groupby('idx').agg({
            'affinity_nM': 'min',
            'affinity_rank': 'min'
        }).reset_index()

        df = df.reset_index(drop=True)
        df['idx'] = df.index
        df = df.merge(netmhc_df, on='idx', how='left')
        df = df.drop('idx', axis=1)
    else:
        df['affinity_nM'] = 500
        df['affinity_rank'] = 50

    df['affinity_nM'] = df['affinity_nM'].fillna(500)
    df['affinity_rank'] = df['affinity_rank'].fillna(50)

    df['strong_binder'] = (df['affinity_rank'] < 0.5).astype(int)
    df['weak_binder'] = (df['affinity_rank'] < 2.0).astype(int)
    df['log_affinity'] = np.log10(df['affinity_nM'] + 1)

    # Anchor residue features
    def get_anchor_aa(seq, pos):
        return seq[pos] if len(seq) > pos else 'X'

    df['anchor_P2'] = df['peptide'].apply(lambda x: get_anchor_aa(x, 1))
    df['anchor_P9'] = df['peptide'].apply(lambda x: get_anchor_aa(x, 8) if len(x) >= 9 else get_anchor_aa(x, -1))

    anchor_P2_common = ['L', 'M', 'V', 'I', 'F', 'Y']
    anchor_P9_common = ['L', 'V', 'F', 'Y', 'W']

    for aa in anchor_P2_common:
        df[f'P2_is_{aa}'] = (df['anchor_P2'] == aa).astype(int)

    for aa in anchor_P9_common:
        df[f'P9_is_{aa}'] = (df['anchor_P9'] == aa).astype(int)

    print(f"  ✓ Extracted anchor residue features")

    # Prepare features
    mutation_features = ['length', 'hydrophobicity', 'net_charge', 'aromatic_content',
                         'polarity'] + [f'aa_{aa}' for aa in amino_acids]

    binding_features = ['affinity_nM', 'affinity_rank', 'log_affinity',
                        'strong_binder', 'weak_binder'] + \
                   [col for col in df.columns if col.startswith('P2_is_')] + \
                   [col for col in df.columns if col.startswith('P9_is_')]

    all_features = mutation_features + binding_features

    X = df[all_features].fillna(0)
    y = df['immunogenic']

    print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Class balance: Positive={y.sum()}, Negative={len(y)-y.sum()}")

    # Training: CV and final model
    print("\n[5/5] Running {}-Fold Stratified Cross-Validation...".format(config.N_CV_FOLDS))

    rf = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        class_weight='balanced'
    )

    skf = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv_aurocs = []
    cv_auprcs = []
    cv_f1s = []
    cv_briers = []

    print(f"  Running {config.N_CV_FOLDS}-fold cross-validation...\n")

    fold_num = 1
    for train_idx, test_idx in skf.split(X_scaled, y):
        rf.fit(X_scaled[train_idx], y.iloc[train_idx])

        y_pred_fold = rf.predict(X_scaled[test_idx])
        y_proba_fold = rf.predict_proba(X_scaled[test_idx])[:, 1]

        auroc_fold = roc_auc_score(y.iloc[test_idx], y_proba_fold)
        auprc_fold = average_precision_score(y.iloc[test_idx], y_proba_fold)
        f1_fold = f1_score(y.iloc[test_idx], y_pred_fold)
        brier_fold = brier_score_loss(y.iloc[test_idx], y_proba_fold)

        cv_aurocs.append(auroc_fold)
        cv_auprcs.append(auprc_fold)
        cv_f1s.append(f1_fold)
        cv_briers.append(brier_fold)

        print(f"   Fold {fold_num:2d}: AUROC={auroc_fold:.4f}, AUPRC={auprc_fold:.4f}, F1={f1_fold:.4f}")
        fold_num += 1

    mean_auroc = np.mean(cv_aurocs)
    std_auroc = np.std(cv_aurocs)
    mean_auprc = np.mean(cv_auprcs)
    std_auprc = np.std(cv_auprcs)
    mean_f1 = np.mean(cv_f1s)
    std_f1 = np.std(cv_f1s)
    mean_brier = np.mean(cv_briers)
    std_brier = np.std(cv_briers)

    print(f"\n{'='*80}")
    print(f"  {config.N_CV_FOLDS}-FOLD CV RESULTS")
    print(f"{'='*80}")
    print(f"  AUROC:  {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"  AUPRC:  {mean_auprc:.4f} ± {std_auprc:.4f}")
    print(f"  F1:     {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Brier:  {mean_brier:.4f} ± {std_brier:.4f}")
    print(f"{'='*80}\n")

    # Train final model on full dataset
    print("  Training final model on full dataset...")
    rf.fit(X_scaled, y)

    # Save model
    print("\nSaving model...")

    with open('neorank_model.pkl', 'wb') as f:
        pickle.dump({
            'model': rf,
            'scaler': scaler,
            'features': all_features,
            'cv_results': {
                'auroc': mean_auroc,
                'auprc': mean_auprc,
                'f1': mean_f1,
                'brier': mean_brier
            }
        }, f)
    print("  ✓ Saved: neorank_model.pkl")

    results_df = pd.DataFrame({
        'Fold': list(range(1, config.N_CV_FOLDS + 1)) + ['Mean', 'Std'],
        'AUROC': cv_aurocs + [mean_auroc, std_auroc],
        'AUPRC': cv_auprcs + [mean_auprc, std_auprc],
        'F1': cv_f1s + [mean_f1, std_f1],
        'Brier': cv_briers + [mean_brier, std_brier]
    })
    results_df.to_csv('neorank_cv_results.csv', index=False)
    print("  ✓ Saved: neorank_cv_results.csv")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
