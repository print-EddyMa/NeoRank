# IEDB Data Download Instructions

## Required Files

1. `epitope.tsv` - Epitope sequences
2. `tcell.tsv` - T-cell assay results

## Download Steps

### 1. Visit IEDB

Go to [https://www.iedb.org](https://www.iedb.org)

### 2. Navigate to Search

Click on "Search" → "T Cell Search"

### 3. Apply Filters

**Epitope Filters:**
- Epitope Type: `Linear Peptide`
- Epitope Source: `Homo sapiens`

**Assay Filters:**
- Assay Type: `T Cell`
- Qualitative Measure: Select `Positive` AND `Negative`

**MHC Filters:**
- MHC Restriction: `Any`

**Host Filters:**
- Host: `Homo sapiens`

**Disease Filters:**
- Disease: `Cancer`

### 4. Download Data

After applying filters:

1. Click "Download" button
2. Select format: `Tab-delimited (TSV)`
3. Download both:
   - Epitope data → save as `epitope.tsv`
   - T Cell Assay data → save as `tcell.tsv`

### 5. Place Files

Move downloaded files to `data/raw/`:

```bash
mv ~/Downloads/epitope.tsv data/raw/
mv ~/Downloads/tcell_assay.tsv data/raw/tcell.tsv
```

## Expected Data Size

- `epitope.tsv`: ~6,757 records, ~5 MB
- `tcell.tsv`: ~15,929 records, ~10 MB

## Verification

Check that files contain required columns:

```python
import pandas as pd

epitope = pd.read_csv('data/raw/epitope.tsv', sep='\t')
tcell = pd.read_csv('data/raw/tcell.tsv', sep='\t')

print("Epitope columns:", epitope.columns.tolist())
print("T-cell columns:", tcell.columns.tolist())
```

Required epitope columns:
- `Epitope ID - IEDB IRI`
- `Epitope - Name`

Required T-cell columns:
- `Epitope - IEDB IRI`
- `Assay - Qualitative Measurement`
- `MHC Restriction - Name`

## Troubleshooting

**Issue: Different number of records**
- IEDB is continuously updated
- Minor variations are expected
- If significantly different (>10%), verify filters

**Issue: Missing columns**
- Ensure TSV format (not CSV)
- Re-download with correct filters

**Issue: Download fails**
- IEDB may have rate limits
- Try smaller date ranges
- Contact IEDB support if persistent

## Citation

If you use IEDB data, please cite:

> Vita R, Mahajan S, Overton JA, et al. The Immune Epitope Database (IEDB): 2018 update. Nucleic Acids Research. 2019;47(D1):D339-D343.
