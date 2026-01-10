"""
Tests for data preprocessing module
"""

import pytest
import pandas as pd
from neorank.data_preprocessing import DataPreprocessor


def test_datapreprocessor_init():
    """Test DataPreprocessor initialization."""
    preprocessor = DataPreprocessor('epitope.tsv', 'tcell.tsv')
    assert preprocessor.epitope_file == 'epitope.tsv'
    assert preprocessor.tcell_file == 'tcell.tsv'


@pytest.mark.skip(reason="Requires IEDB data files")
def test_load_data():
    """Test data loading (requires actual IEDB files)."""
    preprocessor = DataPreprocessor('data/raw/epitope.tsv', 'data/raw/tcell.tsv')
    epitope, tcell = preprocessor.load_data()
    assert isinstance(epitope, pd.DataFrame)
    assert isinstance(tcell, pd.DataFrame)
