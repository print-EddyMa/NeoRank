# Tools Directory

This directory contains setup guides and external tool configurations for NeoRank.

## Contents

- `netmhcpan_setup.md` - Setup instructions for NetMHCpan 4.2

## External Dependencies

### NetMHCpan 4.2

**Purpose**: HLA binding affinity prediction

**Download**: https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/

**License**: Free for academic use (commercial license available)

**Installation**: See `netmhcpan_setup.md`

## Using External Tools

All external tool integrations are abstracted through configuration:

```python
from neorank.config import Config

# Set path to external tool
Config.set_netmhcpan_path('/path/to/netMHCpan-4.2')
```

Tools can be managed via:
- Environment variables
- Configuration files
- Direct code specification
