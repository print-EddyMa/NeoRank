"""
NeoRank Configuration
Author: Eddy Ma
"""

import os
from pathlib import Path


class Config:
    """Configuration parameters for NeoRank pipeline."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # Data files
    EPITOPE_FILE = "epitope.tsv"
    TCELL_FILE = "tcell.tsv"
    
    # NetMHCpan configuration
    NETMHCPAN_PATH = None  # Set by user or auto-detected
    NETMHCPAN_DIR = None
    
    # Model parameters
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    N_CV_FOLDS = 10
    N_ESTIMATORS = 2000
    MAX_DEPTH = 15
    MIN_SAMPLES_SPLIT = 5
    MIN_SAMPLES_LEAF = 2
    MAX_FEATURES = 'sqrt'
    
    # Feature parameters
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Hydrophobicity scale (Kyte-Doolittle)
    HYDROPHOBICITY = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    
    # Charge
    CHARGE = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.5}
    
    # Polar amino acids
    POLAR_AA = set('STNQCYWDEHKR')
    
    # Anchor residues
    ANCHOR_P2_COMMON = ['L', 'M', 'V', 'I', 'F', 'Y']
    ANCHOR_P9_COMMON = ['L', 'V', 'F', 'Y', 'W']
    
    # Binding thresholds
    STRONG_BINDER_RANK = 0.5
    WEAK_BINDER_RANK = 2.0
    
    # Visualization
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'pdf'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        for dir_path in [cls.DATA_DIR, cls.RAW_DATA_DIR, 
                         cls.PROCESSED_DATA_DIR, cls.MODELS_DIR, 
                         cls.RESULTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def set_netmhcpan_path(cls, path: str):
        """Set NetMHCpan executable path."""
        cls.NETMHCPAN_PATH = path
        cls.NETMHCPAN_DIR = Path(path).parent.parent.parent
