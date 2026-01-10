"""
Test suite for NeoRank
"""

import pytest


def test_imports():
    """Test that core modules can be imported."""
    from neorank import Config, DataPreprocessor, FeatureExtractor
    assert Config is not None
    assert DataPreprocessor is not None
    assert FeatureExtractor is not None


def test_config_paths():
    """Test configuration paths are set correctly."""
    from neorank import Config
    assert Config.PROJECT_ROOT.exists()
    assert Config.AMINO_ACIDS == 'ACDEFGHIKLMNPQRSTVWY'
    assert len(Config.HYDROPHOBICITY) == 20
