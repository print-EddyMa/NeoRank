# Feature Documentation

## Overview

This document describes the features extracted by NeoRank for neoantigen immunogenicity prediction.

## Feature Categories

### 1. Mutation Features (Peptide-based)

#### Length
- **Description**: Number of amino acids in peptide
- **Range**: 8-14 (recommended)
- **Importance**: Moderate

#### Amino Acid Composition
- **Description**: Frequency of each amino acid (ACDEFGHIKLMNPQRSTVWY)
- **Range**: 0-1 (normalized)
- **Importance**: High

#### Hydrophobicity
- **Description**: Average Kyte-Doolittle hydrophobicity
- **Range**: -4.5 to 4.5
- **Importance**: High

#### Net Charge
- **Description**: Sum of positive (K, R) and negative (D, E) residues
- **Range**: -5 to 5
- **Importance**: Moderate

#### Aromatic Content
- **Description**: Proportion of aromatic amino acids (F, W, Y)
- **Range**: 0-1
- **Importance**: Moderate

#### Polarity
- **Description**: Proportion of polar amino acids (S, T, N, Q, C, Y, W, D, E, H, K, R)
- **Range**: 0-1
- **Importance**: Low-Moderate

### 2. Binding Features (HLA-based)

#### Binding Affinity (nM)
- **Description**: NetMHCpan-predicted HLA-peptide binding affinity
- **Range**: 0-500+ nM
- **Lower = Stronger binding**
- **Importance**: High

#### Binding Rank
- **Description**: NetMHCpan rank percentile (0-100)
- **Range**: 0-100%
- **Lower = Stronger binding**
- **Importance**: High

#### Strong Binder Flag
- **Description**: Binary indicator (1 if rank < 0.5%, else 0)
- **Range**: 0-1
- **Importance**: High

#### Weak Binder Flag
- **Description**: Binary indicator (1 if rank < 2%, else 0)
- **Range**: 0-1
- **Importance**: Moderate

#### Log Affinity
- **Description**: Log10-transformed binding affinity
- **Range**: 0-2.7+
- **Importance**: High

### 3. Anchor Features (Position-specific)

#### P2 Anchor
- **Description**: Amino acid at position 2
- **Common anchors**: L, M, V, I, F, Y
- **Importance**: Moderate

#### P9 Anchor
- **Description**: Amino acid at position 9 (or C-terminal for shorter peptides)
- **Common anchors**: L, V, F, Y, W
- **Importance**: Moderate

## Feature Statistics

### Training Data (IEDB)
- Total epitopes: 6,757
- Immunogenic: 60%
- Non-immunogenic: 40%

### Feature Correlation

Strong correlations with immunogenicity:
- Binding affinity (r > 0.45)
- Strong binder status (r > 0.40)
- Hydrophobicity (r > 0.30)

## Feature Importance from Model

Based on 2000-tree Random Forest (10-fold CV):

| Feature | Importance | Cumulative |
|---------|-----------|-----------|
| affinity_rank | 0.35 | 0.35 |
| strong_binder | 0.15 | 0.50 |
| log_affinity | 0.12 | 0.62 |
| hydrophobicity | 0.08 | 0.70 |
| length | 0.06 | 0.76 |
| net_charge | 0.04 | 0.80 |
| ... | ... | 1.00 |

## Missing Value Handling

- **Binding features**: Filled with conservative defaults (500 nM, 50% rank)
- **Mutation features**: Computed for all valid peptides
- **HLA types**: Default to HLA-A*02:01 if missing

## Preprocessing Steps

1. Remove duplicate peptides
2. Filter by length (8-14 residues)
3. Standardize HLA nomenclature
4. Compute all features
5. Handle missing values
6. Normalize/scale as needed for model

## Usage in Code

```python
from neorank.feature_extraction import FeatureExtractor

extractor = FeatureExtractor()

# Extract mutation features only
df = extractor.extract_mutation_features(df_input)

# Or extract binding features
df = extractor.extract_binding_features(df_input)

# Or get all features by chaining
df = extractor.extract_mutation_features(df_input)
df = extractor.extract_binding_features(df)
```

## References

- **Hydrophobicity**: Kyte & Doolittle, 1982
- **NetMHCpan**: Reynisson et al., 2020
- **IEDB**: Vita et al., 2019
