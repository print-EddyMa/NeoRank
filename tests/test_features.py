"""
Tests for feature extraction module
"""

import pytest
import pandas as pd
import numpy as np
from neorank.feature_extraction import FeatureExtractor


def test_feature_extractor_init():
    """Test FeatureExtractor initialization."""
    extractor = FeatureExtractor()
    assert extractor.config is not None


def test_mutation_features():
    """Test mutation feature extraction."""
    extractor = FeatureExtractor()
    df = pd.DataFrame({
        'peptide': ['SIINFEKL', 'KVAELVHFL'],
        'label': [1, 0]
    })
    
    result = extractor.extract_mutation_features(df)
    
    # Check that new columns were added
    assert 'length' in result.columns
    assert 'hydrophobicity' in result.columns
    assert 'net_charge' in result.columns
    assert 'aromatic_content' in result.columns
    assert 'polarity' in result.columns
    
    # Check length feature
    assert result['length'].iloc[0] == 8
    assert result['length'].iloc[1] == 10


def test_avg_hydrophobicity():
    """Test hydrophobicity calculation."""
    extractor = FeatureExtractor()
    
    # Test known hydrophobic sequence
    hydro = extractor._avg_hydrophobicity('LLLL')
    assert hydro == 3.8  # L has 3.8 hydrophobicity
    
    # Test mixed sequence
    hydro = extractor._avg_hydrophobicity('AE')
    expected = (1.8 + (-3.5)) / 2
    assert abs(hydro - expected) < 0.001


def test_net_charge():
    """Test net charge calculation."""
    extractor = FeatureExtractor()
    
    # Test positive charges
    charge = extractor._net_charge('KKK')
    assert charge == 3
    
    # Test negative charges
    charge = extractor._net_charge('DDE')
    assert charge == -3
    
    # Test mixed
    charge = extractor._net_charge('KDE')
    assert charge == 1 - 1 - 1  # K - D - E


def test_aromatic_content():
    """Test aromatic content calculation."""
    extractor = FeatureExtractor()
    
    # Test pure aromatic
    arom = extractor._aromatic_content('FWY')
    assert arom == 1.0
    
    # Test no aromatic
    arom = extractor._aromatic_content('AAA')
    assert arom == 0.0
    
    # Test mixed
    arom = extractor._aromatic_content('AAAF')
    assert arom == 0.25
