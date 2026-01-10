import tempfile
import pandas as pd
from neorank import NeoRankTrainer, NeoRankPredictor, Config


def test_smoke_train_and_predict():
    cfg = Config()
    cfg.N_ESTIMATORS = 10
    cfg.N_CV_FOLDS = 2

    ep = pd.DataFrame({
        'Epitope ID - IEDB IRI': ['IEDB:1', 'IEDB:2', 'IEDB:3', 'IEDB:4'],
        'Epitope - Name': ['SIINFEKL', 'KVAELVHFL', 'GILGFVFTL', 'LLGATCMFV']
    })

    tc = pd.DataFrame({
        'Epitope - IEDB IRI': ['IEDB:1', 'IEDB:2', 'IEDB:3', 'IEDB:4'],
        'Assay - Qualitative Measurement': ['Positive', 'Negative', 'Positive', 'Negative'],
        'MHC Restriction - Name': ['HLA-A*02:01', 'HLA-A*02:01', 'HLA-A*02:01', 'HLA-A*02:01']
    })

    with tempfile.TemporaryDirectory() as tmp:
        ep.to_csv(tmp + '/epitope.tsv', sep='\t', index=False)
        tc.to_csv(tmp + '/tcell.tsv', sep='\t', index=False)

        trainer = NeoRankTrainer(epitope_file=tmp + '/epitope.tsv', tcell_file=tmp + '/tcell.tsv', config=cfg)
        model_data = trainer.run(save_model_path=tmp + '/model.pkl', save_results_path=tmp + '/cv.csv')

        assert 'model' in model_data

        predictor = NeoRankPredictor(model_path=tmp + '/model.pkl', config=cfg)
        df_in = pd.DataFrame({'peptide': ['SIINFEKL'], 'HLA': ['HLA-A*02:01']})
        out = predictor.predict_df(df_in)
        assert 'immunogenicity_score' in out.columns
        assert out.shape[0] == 1
