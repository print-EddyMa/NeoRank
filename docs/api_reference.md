# API Reference

## neorank Module

### Config Class

```python
class Config:
    """Configuration parameters for NeoRank pipeline."""
```

#### Class Attributes

**Paths**
- `PROJECT_ROOT`: Path - Root project directory
- `DATA_DIR`: Path - Data directory
- `RAW_DATA_DIR`: Path - Raw data directory
- `PROCESSED_DATA_DIR`: Path - Processed data directory
- `MODELS_DIR`: Path - Models directory
- `RESULTS_DIR`: Path - Results directory

**Model Parameters**
- `RANDOM_SEED`: int = 42
- `TEST_SIZE`: float = 0.2
- `N_CV_FOLDS`: int = 10
- `N_ESTIMATORS`: int = 2000
- `MAX_DEPTH`: int = 15
- `MIN_SAMPLES_SPLIT`: int = 5
- `MIN_SAMPLES_LEAF`: int = 2
- `MAX_FEATURES`: str = 'sqrt'

**Feature Parameters**
- `AMINO_ACIDS`: str = 'ACDEFGHIKLMNPQRSTVWY'
- `HYDROPHOBICITY`: dict - Kyte-Doolittle scale
- `CHARGE`: dict - Amino acid charges
- `POLAR_AA`: set - Polar amino acids
- `ANCHOR_P2_COMMON`: list - Common P2 anchors
- `ANCHOR_P9_COMMON`: list - Common P9 anchors
- `STRONG_BINDER_RANK`: float = 0.5
- `WEAK_BINDER_RANK`: float = 2.0

#### Class Methods

```python
@classmethod
def create_directories(cls) -> None:
    """Create necessary directories if they don't exist."""

@classmethod
def set_netmhcpan_path(cls, path: str) -> None:
    """Set NetMHCpan executable path."""
```

---

### DataPreprocessor Class

```python
class DataPreprocessor:
    """Handles data loading and preprocessing."""
```

#### Methods

```python
def __init__(self, epitope_file: str, tcell_file: str):
    """Initialize preprocessor with data file paths."""

def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load epitope and T-cell assay data.
    
    Returns:
        (epitope_df, tcell_df): Tuple of DataFrames
    """

def prepare_dataset(self, epitope: pd.DataFrame, 
                   tcell: pd.DataFrame) -> pd.DataFrame:
    """Prepare and merge datasets.
    
    Args:
        epitope: Epitope DataFrame from IEDB
        tcell: T-cell assay DataFrame from IEDB
        
    Returns:
        Merged and processed DataFrame with columns:
        - peptide: Peptide sequence
        - HLA: HLA allele
        - immunogenic: Binary label (0/1)
    """
```

---

### FeatureExtractor Class

```python
class FeatureExtractor:
    """Extract mutation and binding features from peptides."""
```

#### Methods

```python
def __init__(self, config: Config = Config()):
    """Initialize feature extractor with configuration."""

def extract_mutation_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Extract peptide-based mutation features.
    
    Adds columns:
    - length: Peptide length
    - aa_*: Amino acid composition (20 features)
    - hydrophobicity: Average hydrophobicity
    - net_charge: Net charge
    - aromatic_content: Aromatic amino acid fraction
    - polarity: Polar amino acid fraction
    
    Args:
        df: DataFrame with 'peptide' column
        
    Returns:
        DataFrame with added mutation features
    """

def extract_binding_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Extract HLA binding features using NetMHCpan.
    
    Adds columns:
    - affinity_nM: NetMHCpan binding affinity
    - affinity_rank: NetMHCpan rank percentile
    - strong_binder: Binary strong binder flag
    - weak_binder: Binary weak binder flag
    - log_affinity: Log10-transformed affinity
    - anchor_P2: P2 anchor amino acid
    - P2_is_*: P2 anchor flags (6 features)
    - anchor_P9: P9 anchor amino acid
    - P9_is_*: P9 anchor flags (5 features)
    
    Args:
        df: DataFrame with 'peptide' and 'HLA' columns
        
    Returns:
        DataFrame with added binding features
    """
```

---

### NeoRankTrainer Class (Placeholder)

```python
class NeoRankTrainer:
    """Train Random Forest model for neoantigen prediction."""
    
    def __init__(self, epitope_file: str, tcell_file: str):
        """Initialize trainer with data paths."""
    
    def train(self) -> dict:
        """Train Random Forest model.
        
        Returns:
            Dictionary with training results and metrics
        """
    
    def cross_validate(self, n_folds: int = 10) -> dict:
        """Perform cross-validation.
        
        Returns:
            Dictionary with CV metrics (AUROC, AUPRC, F1, etc.)
        """
    
    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
```

---

### NeoRankPredictor Class (Placeholder)

```python
class NeoRankPredictor:
    """Make predictions with trained NeoRank model."""
    
    def __init__(self, model_path: str):
        """Initialize predictor with trained model."""
    
    def predict(self, peptides: List[str], 
               hla_types: List[str],
               top_n: int = 20) -> pd.DataFrame:
        """Predict immunogenicity for peptides.
        
        Args:
            peptides: List of peptide sequences
            hla_types: List of HLA alleles
            top_n: Return top N ranked predictions
            
        Returns:
            DataFrame with predictions and features
        """
    
    def predict_batch(self, peptides: List[str],
                     hla_types: List[str],
                     batch_size: int = 100) -> pd.DataFrame:
        """Batch predict with efficiency optimizations."""
```

---

## Data Structures

### Input Format

**Epitope Data (TSV)**
```
Epitope ID - IEDB IRI    Epitope - Name    ...
IEDB:0001               SIINFEKL          ...
```

**T-cell Assay Data (TSV)**
```
Epitope - IEDB IRI    Assay - Qualitative Measurement    MHC Restriction - Name    ...
IEDB:0001             Positive                           HLA-A*02:01               ...
```

### Output Format

**Prediction DataFrame**
```
   peptide         HLA        immunogenicity_score  binding_affinity_nM  rank
0  SIINFEKL  HLA-A*02:01              0.87            45.2               1
1  KVAELVHFL HLA-A*02:01              0.72            120.5              2
```

## Constants

### Amino Acid Sets

```python
Config.AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'  # All 20 standard AAs
Config.POLAR_AA = {'S', 'T', 'N', 'Q', 'C', 'Y', 'W', 'D', 'E', 'H', 'K', 'R'}
Config.ANCHOR_P2_COMMON = ['L', 'M', 'V', 'I', 'F', 'Y']
Config.ANCHOR_P9_COMMON = ['L', 'V', 'F', 'Y', 'W']
```

### Thresholds

```python
Config.STRONG_BINDER_RANK = 0.5     # Top 0.5% binding rank
Config.WEAK_BINDER_RANK = 2.0       # Top 2% binding rank
```

---

## Exceptions

All methods follow standard Python exception handling. Common exceptions:

- `FileNotFoundError`: Data files not found
- `ValueError`: Invalid input format
- `RuntimeError`: NetMHCpan execution failed
- `KeyError`: Missing required DataFrame columns

---

## Version

- **Current**: 1.0.0
- **Python**: 3.8+
- **Dependencies**: pandas>=1.5.0, numpy>=1.23.0, scikit-learn>=1.2.0
