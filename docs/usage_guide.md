# Usage Guide

## Quick Start

### Training a Model

```python
from neorank import NeoRankTrainer

# Initialize trainer
trainer = NeoRankTrainer(
    epitope_file='data/raw/epitope.tsv',
    tcell_file='data/raw/tcell.tsv'
)

# Train model
results = trainer.train()

# Save model
trainer.save_model('models/my_neorank_model.pkl')
```

### Making Predictions

```python
from neorank import NeoRankPredictor

# Load trained model
predictor = NeoRankPredictor('models/neorank_model.pkl')

# Prepare input
peptides = ['SIINFEKL', 'KVAELVHFL', 'GILGFVFTL']
hla_types = ['HLA-A*02:01', 'HLA-A*02:01', 'HLA-A*02:01']

# Predict
results = predictor.predict(peptides, hla_types, top_n=20)

# Results DataFrame contains:
# - peptide: input sequence
# - HLA: input HLA type
# - immunogenicity_score: 0-1 probability
# - binding_affinity_nM: NetMHCpan prediction
# - rank: final ranking
print(results)
```

## Advanced Usage

### Custom Feature Extraction

```python
from neorank.feature_extraction import FeatureExtractor

extractor = FeatureExtractor()

# Extract only mutation features
mutation_features = extractor.extract_mutation_features(df)

# Extract only binding features
binding_features = extractor.extract_binding_features(df)
```

### Cross-Validation

```python
from neorank import NeoRankTrainer

trainer = NeoRankTrainer(...)
cv_results = trainer.cross_validate(n_folds=10)

print(f"Mean AUROC: {cv_results['mean_auroc']:.4f}")
print(f"Std AUROC: {cv_results['std_auroc']:.4f}")
```

### Batch Processing

```python
import pandas as pd

# Load peptides from file
peptide_df = pd.read_csv('my_peptides.csv')

# Batch predict
results = predictor.predict_batch(
    peptides=peptide_df['sequence'],
    hla_types=peptide_df['hla'],
    batch_size=100
)
```

## Input Format

### Peptides

- Amino acid sequences (8-14 residues recommended)
- Standard single-letter codes (ACDEFGHIKLMNPQRSTVWY)
- Example: `SIINFEKL`, `KVAELVHFL`

### HLA Types

- Format: `HLA-A*02:01`, `HLA-B*07:02`, etc.
- Both 4-digit and 2-digit alleles supported
- Common alleles perform best (model trained on top HLA types)

## Output Format

Results are returned as pandas DataFrame:

| Column | Description |
|--------|-------------|
| peptide | Input peptide sequence |
| HLA | Input HLA type |
| immunogenicity_score | Predicted probability (0-1) |
| binding_affinity_nM | NetMHCpan affinity (lower = stronger) |
| binding_rank | NetMHCpan rank percentile |
| strong_binder | Binary flag (<0.5% rank) |
| final_score | Combined immunogenicity + binding |
| rank | Final ranking (1 = best) |

## Performance Tips

1. **Batch Processing**: Process multiple peptides at once for efficiency
2. **HLA Filtering**: Pre-filter HLA types to supported alleles
3. **Length Filtering**: Focus on 8-11 residue peptides for best results
4. **Caching**: NetMHCpan predictions are cached automatically

## Examples

See `notebooks/` for complete examples:
- `01_data_exploration.ipynb`: Data analysis
- `02_feature_analysis.ipynb`: Feature importance
- `03_model_evaluation.ipynb`: Performance evaluation
