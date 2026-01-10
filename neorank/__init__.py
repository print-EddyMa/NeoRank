"""Top-level neorank package wrapper for the library implementation located in `src/`.
This file makes `import neorank` work for users without installing the package.
It loads the `src/` modules into a `neorank._impl` namespace so relative imports inside
those modules resolve correctly.
"""
import importlib.util
import sys
import types
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
if not SRC_DIR.exists():
    raise ImportError("Required package sources not found in 'src/'. Ensure repository layout is intact.")

_IMPL_PACKAGE = 'neorank._impl'
# create a package module to host the implementation
_impl_pkg_module = types.ModuleType(_IMPL_PACKAGE)
_impl_pkg_module.__path__ = [str(SRC_DIR)]
sys.modules[_IMPL_PACKAGE] = _impl_pkg_module

def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# load constituent modules under neorank._impl
_load_module(f"{_IMPL_PACKAGE}.config", SRC_DIR / 'config.py')
_load_module(f"{_IMPL_PACKAGE}.data_preprocessing", SRC_DIR / 'data_preprocessing.py')
_load_module(f"{_IMPL_PACKAGE}.feature_extraction", SRC_DIR / 'feature_extraction.py')
# finally load the package initializer which defines Trainer/Predictor
_impl = _load_module(_IMPL_PACKAGE, SRC_DIR / '__init__.py')

# Re-export commonly used names
Config = _impl.Config
DataPreprocessor = _impl.DataPreprocessor
FeatureExtractor = _impl.FeatureExtractor
NeoRankTrainer = _impl.NeoRankTrainer
NeoRankPredictor = _impl.NeoRankPredictor

__all__ = [
    'Config', 'DataPreprocessor', 'FeatureExtractor', 'NeoRankTrainer', 'NeoRankPredictor'
]