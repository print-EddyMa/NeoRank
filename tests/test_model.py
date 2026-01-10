"""
Tests for model training and evaluation
"""

import pytest
from neorank import NeoRankTrainer, NeoRankPredictor


def test_trainer_placeholder():
    """Test that trainer raises NotImplementedError as expected."""
    with pytest.raises(NotImplementedError):
        trainer = NeoRankTrainer(
            epitope_file='data/raw/epitope.tsv',
            tcell_file='data/raw/tcell.tsv'
        )


def test_predictor_placeholder():
    """Test that predictor raises NotImplementedError as expected."""
    with pytest.raises(NotImplementedError):
        predictor = NeoRankPredictor(model_path='models/neorank_model.pkl')
