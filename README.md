# NeoRank: Accessible Neoantigen Immunogenicity Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/print-EddyMa/NeoRank/actions/workflows/ci.yml/badge.svg)](https://github.com/print-EddyMa/NeoRank/actions/workflows/ci.yml)


> **Making cancer immunotherapy research accessible worldwide with a $300 solution**

NeoRank is a machine learning pipeline that predicts neoantigen immunogenicity using only peptide sequences and HLA typing—achieving clinical-grade performance (AUROC: 0.824 ± 0.018) at ~10% of the cost of current multi-omics tools.

In 2024, cancer killed 9.7 million people worldwide, with 70% of deaths occurring at late stages where tumors resist standard treatment options. Immunotherapies rely on tumor-specific neoantigens to train the immune system to recognize and attack tumors. This approach has shown promising therapeutic potential in late-stage cancers, a 2023 pancreatic cancer trial prevented recurrence in 50% of the patients. However, neoantigen selection tools require extensive sequencing methods costing $1,000-$5,000 per patient and require equipment worth $500k-$1.5M; this limits access to ~5% of global laboratories. NeoRank uses Random Forest classifiers designed for only two inputs: peptide sequences and HLA-typing, methods that can be obtained through basic laboratory techniques in days, typically costing ~$300. The model was trained on 6,757 cancer epitopes and 15,929 T-cell assays from the Immune Epitope Database and extracts 41+ features from peptide and binding features to predict neoantigen immunogenicity. NeoRank was evaluated through 10-fold cross-validation and achieved AUROC 0.824 ± 0.018 (ability to distinguish between positive/negative classes), comparable to clinical-grade tools like NeoDisc, NeoTImmuML, and imNEO. When evaluated on tuberculosis, HIV, and COVID-19 datasets, the model achieved AUROC 0.0796-0.894, demonstrating strong universal immunogenicity prediction. NeoRank challenges the belief that expensive multi-omics data is necessary for accurate neoantigen prediction and enables resource-limited institutions worldwide to contribute to cancer vaccine development, one of the most rapidly developing treatments that may lead to the very first cures for cancer.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/NeoRank.git
cd NeoRank

# Create conda environment
conda env create -f environment.yml
conda activate neorank

# Install NetMHCpan 4.2 (see tools/netmhcpan_setup.md)
```

### Basic Usage

Using the package API (DataFrame-based):

```python
import pandas as pd
from neorank import NeoRankPredictor

# Load a saved model
predictor = NeoRankPredictor(model_path='models/neorank_model.pkl')

# Prepare input DataFrame
df = pd.DataFrame({
    'peptide': ['SIINFEKL', 'KVAELVHFL'],
    'HLA': ['HLA-A*02:01', 'HLA-A*02:01']
})

# Run predictions
out = predictor.predict_df(df)
print(out[['peptide', 'HLA', 'immunogenicity_score', 'prediction_label']])
```

Using the provided CLI script:

```bash
# Train on example data (expects epitope.tsv and tcell.tsv in the folder)
python scripts/train_neorank.py --folder data/example

# Predict on a CSV file (columns: peptide,HLA)
python scripts/predict_neorank.py data/example/example_peptides.csv
```

Note: `neorank_model.pkl` is saved to `models/` and CV results to `results/` by default.

## Documentation

- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage_guide.md)
- [Feature Documentation](docs/feature_documentation.md)
- [API Reference](docs/api_reference.md)
- [FAQ](docs/FAQ.md)

## Scientific Background

### The Problem

- 70% of cancer deaths occur at late stages where tumors resist treatment
- Current neoantigen selection tools require expensive multi-omics data
- Only ~5% of global labs have access to required infrastructure ($500K-$1.5M sequencers)

### The Solution

NeoRank demonstrates that **peptide features and HLA typing alone are sufficient** for accurate neoantigen prediction.

## Citation

If you use NeoRank in your research, please cite:

```bibtex
@software{ma2025neorank,
  author = {Ma, Eddy},
  title = {NeoRank: Accessible Neoantigen Immunogenicity Prediction},
  year = {2025},
  institution = {Raleigh Charter High School},
  url = {https://github.com/yourusername/NeoRank}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Data Source**: [Immune Epitope Database (IEDB)](https://www.iedb.org)
- **Binding Predictions**: [NetMHCpan 4.2](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/)
- **Inspiration**: Research teams advancing personalized cancer immunotherapy

## Author

**Eddy Ma**  
North Carolina, USA  

---

**❤️ For accessible cancer research**
