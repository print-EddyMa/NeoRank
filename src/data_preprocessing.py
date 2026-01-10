"""
Data Preprocessing Module
Author: Eddy Ma
"""

import pandas as pd
import numpy as np
from typing import Tuple
from .config import Config


class DataPreprocessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, epitope_file: str, tcell_file: str):
        self.epitope_file = epitope_file
        self.tcell_file = tcell_file
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load epitope and T-cell assay data."""
        epitope = pd.read_csv(self.epitope_file, sep="\t", low_memory=False)
        tcell = pd.read_csv(self.tcell_file, sep="\t", low_memory=False)
        return epitope, tcell
    
    def prepare_dataset(self, epitope: pd.DataFrame, 
                       tcell: pd.DataFrame) -> pd.DataFrame:
        """Prepare and merge datasets."""
        # Filter T-cell assays
        t = tcell[['Epitope - IEDB IRI', 'Assay - Qualitative Measurement']].copy()
        t = t[t['Assay - Qualitative Measurement'].isin(['Positive', 'Negative'])]
        
        # Prepare epitope data
        e = epitope[['Epitope ID - IEDB IRI', 'Epitope - Name']].copy()
        e = e.rename(columns={
            'Epitope ID - IEDB IRI': 'Epitope - IEDB IRI',
            'Epitope - Name': 'peptide'
        })
        
        # Merge
        df = t.merge(e, on='Epitope - IEDB IRI', how='inner')
        
        # Create immunogenicity label
        df['immunogenic'] = df['Assay - Qualitative Measurement'].map({
            'Positive': 1, 'Negative': 0
        })
        
        # Aggregate by peptide (majority vote)
        df_agg = df.groupby(['Epitope - IEDB IRI', 'peptide']).agg({
            'immunogenic': lambda x: 1 if x.mean() >= 0.5 else 0
        }).reset_index()
        
        # Add HLA information
        hla_df = tcell[['Epitope - IEDB IRI', 'MHC Restriction - Name']].dropna()
        hla_df = hla_df.drop_duplicates()
        hla_df = hla_df.rename(columns={'MHC Restriction - Name': 'HLA'})
        hla_df = hla_df.groupby('Epitope - IEDB IRI').first().reset_index()
        
        df_final = df_agg.merge(hla_df, on='Epitope - IEDB IRI', how='left')
        df_final['HLA'] = df_final['HLA'].fillna('HLA-A*02:01')
        
        return df_final
