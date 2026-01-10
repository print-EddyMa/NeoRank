"""
NeoRank package
"""

from .config import Config
from .data_preprocessing import DataPreprocessor
from .feature_extraction import FeatureExtractor


# Placeholder classes for trainer and predictor to be implemented
class NeoRankTrainer:
    """Placeholder for NeoRankTrainer - to be implemented."""
    def __init__(self, epitope_file=None, tcell_file=None):
        raise NotImplementedError("NeoRankTrainer is a placeholder. Implement training logic.")


class NeoRankPredictor:
    """Placeholder for NeoRankPredictor - to be implemented."""
    def __init__(self, model_path=None):
        raise NotImplementedError("NeoRankPredictor is a placeholder. Implement prediction logic.")


__all__ = [
    'Config',
    'DataPreprocessor',
    'FeatureExtractor',
    'NeoRankTrainer',
    'NeoRankPredictor',
]
