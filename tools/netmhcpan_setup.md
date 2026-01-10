# NetMHCpan 4.2 Setup Guide

NetMHCpan is required for HLA binding affinity predictions.

## Download

1. Visit [DTU Health Tech](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/)
2. Request academic license (free for non-commercial use)
3. Download version 4.2 for your operating system

## Installation

### macOS / Linux

```bash
# Extract archive
tar -xzf netMHCpan-4.2.Linux.tar.gz  # or .Darwin for macOS

# Move to tools directory
mv netMHCpan-4.2 NeoRank/tools/

# Set execute permissions
chmod +x NeoRank/tools/netMHCpan-4.2/*/bin/netMHCpan-4.2

# Set environment variable (add to ~/.bashrc or ~/.zshrc)
export NETMHCpan=/path/to/NeoRank/tools/netMHCpan-4.2
```

### Windows

NetMHCpan does not have native Windows support. Use:
- Windows Subsystem for Linux (WSL)
- Docker container
- Virtual machine with Linux

## Configuration

### Option 1: Environment Variable

```bash
export NETMHCpan=/path/to/netMHCpan-4.2
```

### Option 2: In Code

```python
from neorank.config import Config

Config.set_netmhcpan_path('/path/to/netMHCpan-4.2/Darwin_x86_64/bin/netMHCpan-4.2')
```

## Verification

Test NetMHCpan installation:

```bash
cd tools/netMHCpan-4.2
./Darwin_x86_64/bin/netMHCpan-4.2 -h
```

Expected output: NetMHCpan help message

## Supported HLA Alleles

NetMHCpan 4.2 supports:
- HLA-A, HLA-B, HLA-C (Class I)
- 200+ HLA alleles
- Both 4-digit and 2-digit resolution

Full list: See `data/` directory in NetMHCpan installation

## Common Issues

**"Command not found"**
- Check path to executable
- Verify execute permissions: `chmod +x netMHCpan-4.2`

**"Cannot execute binary file"**
- Wrong OS version (e.g., using Linux binary on macOS)
- Re-download correct version

**"Permission denied"**
- Run: `chmod +x netMHCpan-4.2`
- Check directory permissions

**Slow predictions**
- Normal for large datasets
- Consider batch processing
- Enable parallelization if available

## Alternative: Docker

Use NetMHCpan via Docker:

```bash
docker pull dholab/netmhcpan:4.1
```

Modify NeoRank to call Docker container instead of local binary.

## License

NetMHCpan is free for academic use. Commercial users must contact DTU for licensing.

## Citation

> Reynisson B, Alvarez B, Paul S, Peters B, Nielsen M. NetMHCpan-4.1 and NetMHCIIpan-4.0: improved predictions of MHC antigen presentation by concurrent motif deconvolution and integration of MS MHC eluted ligand data. Nucleic Acids Research. 2020;48(W1):W449-W454.
