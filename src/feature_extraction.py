"""
Feature Extraction Module
Author: Eddy Ma
"""

import pandas as pd
import numpy as np
import subprocess
import os
from typing import Dict, List
from .config import Config


class FeatureExtractor:
    """Extract mutation and binding features from peptides."""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        
    def extract_mutation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract peptide-based mutation features."""
        df = df.copy()
        
        # Length
        df['length'] = df['peptide'].str.len()
        
        # Amino acid composition
        for aa in self.config.AMINO_ACIDS:
            df[f'aa_{aa}'] = df['peptide'].apply(
                lambda x: x.count(aa) / len(x) if len(x) > 0 else 0
            )
        
        # Hydrophobicity
        df['hydrophobicity'] = df['peptide'].apply(self._avg_hydrophobicity)
        
        # Net charge
        df['net_charge'] = df['peptide'].apply(self._net_charge)
        
        # Aromatic content
        df['aromatic_content'] = df['peptide'].apply(self._aromatic_content)
        
        # Polarity
        df['polarity'] = df['peptide'].apply(self._polarity)
        
        return df
    
    def _avg_hydrophobicity(self, seq: str) -> float:
        """Calculate average hydrophobicity."""
        vals = [self.config.HYDROPHOBICITY.get(aa, 0) for aa in seq]
        return np.mean(vals) if vals else 0
    
    def _net_charge(self, seq: str) -> float:
        """Calculate net charge."""
        return sum(self.config.CHARGE.get(aa, 0) for aa in seq)
    
    def _aromatic_content(self, seq: str) -> float:
        """Calculate aromatic amino acid content."""
        count = sum(1 for aa in seq if aa in 'FWY')
        return count / len(seq) if len(seq) > 0 else 0
    
    def _polarity(self, seq: str) -> float:
        """Calculate polarity."""
        count = sum(1 for aa in seq if aa in self.config.POLAR_AA)
        return count / len(seq) if len(seq) > 0 else 0
    
    def extract_binding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract HLA binding features using NetMHCpan."""
        # Run NetMHCpan predictions
        netmhc_results = self._run_netmhcpan(df)
        
        # Merge results
        df = df.merge(netmhc_results, left_index=True, right_index=True, how='left')
        
        # Fill missing values
        df['affinity_nM'] = df['affinity_nM'].fillna(500)
        df['affinity_rank'] = df['affinity_rank'].fillna(50)
        
        # Derived features
        df['strong_binder'] = (df['affinity_rank'] < self.config.STRONG_BINDER_RANK).astype(int)
        df['weak_binder'] = (df['affinity_rank'] < self.config.WEAK_BINDER_RANK).astype(int)
        df['log_affinity'] = np.log10(df['affinity_nM'] + 1)
        
        # Anchor residues
        df = self._extract_anchor_features(df)
        
        return df
    
    def _run_netmhcpan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run NetMHCpan predictions and return DataFrame indexed like input with
        `affinity_nM` and `affinity_rank` columns.
        If NetMHCpan is not available or configured, returns an empty DataFrame
        so calling code can fill defaults.
        """
        import tempfile

        netmhc_path = self.config.NETMHCPAN_PATH
        netmhc_dir = self.config.NETMHCPAN_DIR

        # If not configured or binary missing, skip running
        if not netmhc_path or not os.path.isfile(netmhc_path):
            return pd.DataFrame(index=df.index, columns=['affinity_nM', 'affinity_rank'])

        def normalize_hla(hla_string):
            if pd.isna(hla_string):
                return None
            hla = hla_string.strip()
            invalid_patterns = ['class', 'DRB', 'DPB', 'DQB', 'DRA', 'DPA', 'DQA',
                               'DR1', 'DR3', 'DR4', 'DR7', 'DR9', 'DR11', 'DR15', 'DR53',
                               'DQ6', 'DP', 'Cw', 'DPw']
            if any(p in hla for p in invalid_patterns):
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
        orig_dir = os.getcwd()
        try:
            if netmhc_dir and os.path.isdir(netmhc_dir):
                os.chdir(netmhc_dir)

            for original_hla, hla_formatted in valid_hlas:
                mask = df['HLA'] == original_hla
                peptides = df[mask][['peptide']].copy()
                peptides['idx'] = df[mask].index

                for length, group in peptides.groupby(peptides['peptide'].str.len()):
                    if length < 8 or length > 14:
                        continue
                    if group.empty:
                        continue

                    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpf:
                        pep_file = tmpf.name
                        for _, row in group.iterrows():
                            tmpf.write(f"{row['peptide']}\n")

                    try:
                        env = os.environ.copy()
                        if netmhc_dir:
                            env['NETMHCpan'] = str(netmhc_dir)

                        cmd = [str(netmhc_path), '-p', pep_file, '-a', hla_formatted, '-BA']
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
                        if result.returncode != 0:
                            continue

                        lines = result.stdout.split('\n')
                        for line in lines:
                            if (not line.strip() or line.startswith('#') or
                                line.startswith('-') or line.startswith('Protein') or
                                ('Pos' in line and 'MHC' in line)):
                                continue
                            parts = line.split()
                            if len(parts) >= 15:
                                try:
                                    peptide = parts[2]
                                    rank_ba = parts[13]
                                    aff_nm = parts[14]
                                    if rank_ba == 'NA' or aff_nm == 'NA':
                                        continue
                                    affinity_rank = float(rank_ba)
                                    affinity_nM = 0.01 if '<=' in aff_nm else float(aff_nm)
                                    match = group[group['peptide'] == peptide]
                                    if not match.empty:
                                        idx = match.iloc[0]['idx']
                                        netmhc_data.append({
                                            'idx': idx,
                                            'affinity_nM': affinity_nM,
                                            'affinity_rank': affinity_rank
                                        })
                                except (ValueError, IndexError):
                                    continue
                    except Exception:
                        continue
                    finally:
                        if os.path.exists(pep_file):
                            os.remove(pep_file)
        finally:
            os.chdir(orig_dir)

        if netmhc_data:
            netmhc_df = pd.DataFrame(netmhc_data)
            netmhc_df = netmhc_df.groupby('idx').agg({'affinity_nM': 'min', 'affinity_rank': 'min'}).reset_index()
            out = pd.DataFrame(index=df.index, columns=['affinity_nM', 'affinity_rank'])
            out['idx'] = out.index
            out = out.merge(netmhc_df, on='idx', how='left').set_index('idx')
            return out[['affinity_nM', 'affinity_rank']]
        else:
            return pd.DataFrame(index=df.index, columns=['affinity_nM', 'affinity_rank'])
    
    def _extract_anchor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract anchor residue features."""
        df = df.copy()
        
        # P2 anchor
        df['anchor_P2'] = df['peptide'].apply(lambda x: x[1] if len(x) > 1 else 'X')
        for aa in self.config.ANCHOR_P2_COMMON:
            df[f'P2_is_{aa}'] = (df['anchor_P2'] == aa).astype(int)
        
        # P9 anchor
        df['anchor_P9'] = df['peptide'].apply(
            lambda x: x[8] if len(x) >= 9 else x[-1] if len(x) > 0 else 'X'
        )
        for aa in self.config.ANCHOR_P9_COMMON:
            df[f'P9_is_{aa}'] = (df['anchor_P9'] == aa).astype(int)
        
        return df
