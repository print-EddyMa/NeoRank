"""
=====================================
NeoRank: Prediction Script
=====================================
Author: Eddy Ma
=====================================
"""

import argparse
import pandas as pd
import numpy as np
import os
import subprocess
import pickle
import warnings
from pathlib import Path
import tempfile

warnings.filterwarnings('ignore')


def build_config(args):
    repo_root = Path(__file__).resolve().parents[1]
    default_model = 'neorank_model.pkl'
    default_netmhc_dir = repo_root / 'tools'
    default_netmhc_path = default_netmhc_dir / 'netMHCpan-4.2' / 'Darwin_x86_64' / 'bin' / 'netMHCpan-4.2'

    class Config:
        MODEL_PATH = args.model if args.model else str(default_model)
        NETMHCPAN_PATH = args.netmhcpan_path if args.netmhcpan_path else str(default_netmhc_path)
        NETMHCPAN_DIR = args.netmhcpan_dir if args.netmhcpan_dir else str(default_netmhc_dir)

    return Config()


def main():
    parser = argparse.ArgumentParser(description='Make NeoRank predictions')
    parser.add_argument('input_csv', help='Input CSV file containing peptide,HLA')
    parser.add_argument('--model', help='Path to model pickle (default: neorank_model.pkl)')
    parser.add_argument('--netmhcpan-path', help='Path to netMHCpan binary')
    parser.add_argument('--netmhcpan-dir', help='Directory for netMHCpan')

    args = parser.parse_args()
    config = build_config(args)

    print("=" * 80)
    print("NEORANK: PREDICTION PIPELINE")
    print("=" * 80)

    print("\n[1/4] Loading trained model...")
    if not os.path.isfile(config.MODEL_PATH):
        raise SystemExit(f"Model not found: {config.MODEL_PATH}. Run training first.")

    with open(config.MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['features']

    print(f"  ✓ Model loaded successfully")
    print(f"  - CV AUROC: {model_data['cv_results']['auroc']:.4f}")
    print(f"  - CV AUPRC: {model_data['cv_results']['auprc']:.4f}")

    print("\n[2/4] Loading input data...")
    input_file = args.input_csv
    df = pd.read_csv(input_file)

    if 'peptide' not in df.columns or 'HLA' not in df.columns:
        raise SystemExit('Input CSV must have columns: peptide,HLA')

    print(f"  ✓ Loaded {len(df)} peptides for prediction")

    # Mutation features
    print("\n[3/4] Extracting features...")

    df['length'] = df['peptide'].str.len()

    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    for aa in amino_acids:
        df[f'aa_{aa}'] = df['peptide'].apply(lambda x: x.count(aa)/len(x) if len(x) > 0 else 0)

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

    print(f"  ✓ Mutation features extracted")

    # Binding features
    print("  - Running NetMHCpan predictions...")

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

    unique_hlas = df['HLA'].unique()
    valid_hlas = []
    for hla in unique_hlas:
        normalized = normalize_hla(hla)
        if normalized:
            valid_hlas.append((hla, normalized))

    netmhc_data = []
    original_dir = os.getcwd()

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

    # Anchor residues
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

    print(f"  ✓ Binding features extracted")

    # Make predictions
    print("\n[4/4] Making predictions...")

    X = df[feature_names].fillna(0)
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    df['prediction'] = predictions
    df['immunogenicity_score'] = probabilities
    df['prediction_label'] = df['prediction'].map({0: 'Non-Immunogenic', 1: 'Immunogenic'})

    output_file = input_file.replace('.csv', '_predictions.csv')
    output_df = df[['peptide', 'HLA', 'immunogenicity_score', 'prediction_label',
                    'affinity_nM', 'affinity_rank', 'strong_binder'
                    ]]
    output_df.to_csv(output_file, index=False)
    print(f"\n  ✓ Predictions saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  - Total peptides: {len(df)}")
    print(f"  - Predicted immunogenic: {predictions.sum()}")
    print(f"  - Predicted non-immunogenic: {len(predictions) - predictions.sum()}")
    print(f"  - Mean immunogenicity score: {probabilities.mean():.4f}")
    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
