# Frequently Asked Questions (FAQ)

## Installation & Setup

### Q1: How do I install NeoRank?

**A:** See [Installation Guide](installation.md) for detailed steps. Quick version:

```bash
git clone https://github.com/yourusername/NeoRank.git
cd NeoRank
conda env create -f environment.yml
conda activate neorank
```

### Q2: Do I need NetMHCpan?

**A:** Yes, for HLA binding affinity predictions. However, basic mutation features work without it. Download from [DTU Health Tech](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/).

### Q3: Can I use NeoRank on Windows?

**A:** NetMHCpan doesn't have native Windows support. Use WSL (Windows Subsystem for Linux), Docker, or a virtual machine.

### Q4: What Python version do I need?

**A:** Python 3.8 or higher.

---

## Usage & Features

### Q5: What input data does NeoRank need?

**A:** Only:
1. **Peptide sequences** (8-14 amino acids)
2. **HLA typing** (e.g., HLA-A*02:01)

No WES, RNA-seq, or other expensive omics data required.

### Q6: What are the performance metrics?

**A:** On IEDB training data (10-fold CV):
- **AUROC**: 0.824 ± 0.018
- Competitive with tools requiring full multi-omics
- 10% of the cost (~$300 vs ~$3,000)

### Q7: Can I train my own model?

**A:** Yes. Download IEDB data (see `data/raw/download_instructions.md`) and run:

```bash
python scripts/train_model.py
```

### Q8: What HLA alleles are supported?

**A:** Any HLA-A, HLA-B, or HLA-C allele supported by NetMHCpan 4.2 (200+ alleles). Common 4-digit and 2-digit formats work.

### Q9: What peptide lengths work best?

**A:** 8-11 residues for optimal performance. 8-14 residues are generally acceptable.

### Q10: How long does prediction take?

**A:** Depends on:
- Peptide count
- NetMHCpan installation
- Hardware

Typical: 1-5 seconds per 100 peptides.

---

## Data & Files

### Q11: Where do I get training data?

**A:** From [IEDB](https://www.iedb.org). See `data/raw/download_instructions.md` for step-by-step guide.

### Q12: What does "Positive" vs "Negative" mean?

**A:** 
- **Positive**: Epitope elicited immune response in T-cell assays
- **Negative**: No immune response detected

### Q13: Why are my predictions different from other tools?

**A:** NeoRank uses only peptide features and HLA binding, while other tools incorporate:
- Gene expression (RNA-seq)
- VAF (variant allele frequency)
- Copy number variations
- Other omics data

Different inputs → different predictions. NeoRank focuses on what's accessible to resource-limited labs.

### Q14: Can I use pre-trained models?

**A:** Currently, we recommend training on your own IEDB subset. Pre-trained models will be provided in future releases.

---

## Technical Questions

### Q15: How does NeoRank compare to NetMHCpan?

**A:**

| Aspect | NetMHCpan | NeoRank |
|--------|-----------|---------|
| Input | Peptide + HLA | Peptide + HLA |
| Method | Binding affinity only | Random Forest (100+ features) |
| Output | Affinity (nM) | Immunogenicity probability |
| AUROC | ~0.65 | ~0.824 |

### Q16: What's the Random Forest architecture?

**A:**
- **Trees**: 2,000
- **Max depth**: 15
- **Min samples split**: 5
- **Min samples leaf**: 2
- **Max features**: sqrt

Tuned via cross-validation on IEDB.

### Q17: How do I extract features without training?

**A:**

```python
from neorank.feature_extraction import FeatureExtractor

extractor = FeatureExtractor()
df_with_features = extractor.extract_mutation_features(df)
df_with_binding = extractor.extract_binding_features(df)
```

### Q18: What's the reproducibility guarantee?

**A:** Set `RANDOM_SEED = 42` in config. Results are reproducible across runs on the same hardware with the same data.

### Q19: How are missing values handled?

**A:**
- **Binding affinity**: Default to 500 nM (non-binder)
- **Binding rank**: Default to 50% (non-binder)
- **HLA missing**: Assume HLA-A*02:01

---

## Performance & Scaling

### Q20: Can I batch process large datasets?

**A:** Yes, use `predictor.predict_batch()`:

```python
results = predictor.predict_batch(
    peptides=peptide_list,
    hla_types=hla_list,
    batch_size=100  # Adjust based on memory
)
```

### Q21: How much memory does NeoRank need?

**A:** ~500 MB for model + data. Can process 100K peptides with standard laptop.

### Q22: Can I parallelize predictions?

**A:** Yes. Model is saved as pickle, can load in multiple processes. NetMHCpan is inherently parallelizable.

---

## Scientific Questions

### Q23: Why does NeoRank work without RNAseq?

**A:** Because mutation (peptide) + binding features explain 55-45% of variance. Key insight: immunogenicity is primarily determined by:

1. **MHC binding** (affinity, anchors) - 45%
2. **Peptide properties** (composition, hydrophobicity) - 55%

Gene expression is secondary in many contexts.

### Q24: Is NeoRank validated on other datasets?

**A:** Trained on IEDB cancer epitopes. Tested on:
- Internal validation: 10-fold CV
- Other cancers: Good generalization
- Viruses (TB, HIV, COVID): Reasonable performance

### Q25: What about off-target effects?

**A:** NeoRank predicts MHC-immunogenicity only. Doesn't assess:
- T-cell cross-reactivity
- Tolerance/regulatory T cells
- Tumor microenvironment factors

Recommended: Use with caution in clinical contexts; validate experimentally.

---

### Q29: Citing NeoRank

**A:** Please cite as:

```bibtex
@software{ma2025neorank,
  author = {Ma, Eddy},
  title = {NeoRank: Accessible Neoantigen Immunogenicity Prediction},
  year = {2025},
  institution = {Raleigh Charter High School},
  url = {https://github.com/yourusername/NeoRank}
}
```


---

## Additional Resources

- **IEDB**: https://www.iedb.org
- **NetMHCpan**: https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/
