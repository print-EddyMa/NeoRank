import pandas as pd
from neorank import FeatureExtractor


def test_mutation_features_basic():
    df = pd.DataFrame({'peptide': ['ACD', 'WYW']})
    fe = FeatureExtractor()
    out = fe.extract_mutation_features(df)

    # Basic checks
    assert 'length' in out.columns
    assert 'hydrophobicity' in out.columns
    assert 'net_charge' in out.columns
    assert 'aa_A' in out.columns

    assert out.loc[0, 'length'] == 3
    assert out.loc[1, 'length'] == 3
    assert abs(out.loc[0, 'aa_A'] - (1/3)) < 1e-8
    assert abs(out.loc[1, 'aa_W'] - (2/3)) < 1e-8
