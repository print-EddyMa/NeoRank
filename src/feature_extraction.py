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
        """Run NetMHCpan predictions."""
        # Implementation similar to original code
        # Returns DataFrame with affinity_nM and affinity_rank columns
        # Placeholder implementation: return empty frame with expected columns
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
