# Installation Guide

## Prerequisites

- Python 3.8 or higher
- Conda (recommended) or pip
- NetMHCpan 4.2 (for binding predictions)
- ~5GB disk space for data and models

## Step 1: Clone Repository

```bash
git clone [https://github.com/yourusername/NeoRank.git] i.e. (https://github.com/print-EddyMa/NeoRank.git)
cd NeoRank
```

## Step 2: Create Environment

### Option A: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate neorank
```

### Option B: Using pip

```bash
python -m venv neorank_env
source neorank_env/bin/activate  # On Windows: neorank_env\Scripts\activate
pip install -r requirements.txt
```

## Step 3: Install NetMHCpan 4.2

NetMHCpan is required for HLA binding predictions. See detailed instructions in `tools/netmhcpan_setup.md`.

**Quick steps:**

1. Download from [DTU Health Tech](https://services.healthtech.dtu.dk/software.php)
2. Extract to `tools/netMHCpan-4.2/`
3. Configure path in code or environment variable

## Step 4: Download Training Data

See `data/raw/download_instructions.md` for detailed steps.

**Quick steps:**

1. Visit [IEDB](https://www.iedb.org)
2. Apply filters (see download_instructions.md)
3. Download epitope.tsv and tcell.tsv
4. Place in `data/raw/`

## Step 5: Verify Installation

```bash
python -c "import neorank; print('NeoRank installed successfully!')"
pytest tests/  # Run tests
```

## Troubleshooting

### NetMHCpan not found

Set the path explicitly:

```python
from neorank.config import Config
Config.set_netmhcpan_path('/path/to/netMHCpan-4.2')
```

### Missing dependencies

```bash
pip install -r requirements.txt --upgrade
```

### IEDB data format issues

Ensure TSV files are tab-delimited and contain required columns. See example files in `data/example/`.
