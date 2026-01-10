"""
NeoRank package
"""

from .config import Config
from .data_preprocessing import DataPreprocessor
from .feature_extraction import FeatureExtractor


import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, brier_score_loss)
from pathlib import Path


# Trainer and Predictor implementations
class NeoRankTrainer:
    """Train NeoRank models using the library components."""

    def __init__(self, epitope_file: str = None, tcell_file: str = None, config: 'Config' = None):
        self.config = config or Config()
        self.epitope_file = epitope_file or str(self.config.DATA_DIR / self.config.EPITOPE_FILE)
        self.tcell_file = tcell_file or str(self.config.DATA_DIR / self.config.TCELL_FILE)

        # ensure directories exist
        self.config.create_directories()

    def run(self, save_model_path: str = None, save_results_path: str = None):
        """Run the full training pipeline and save model + CV results."""
        print("Starting NeoRank training pipeline...")
        dp = DataPreprocessor(self.epitope_file, self.tcell_file)
        ep, tc = dp.load_data()
        df = dp.prepare_dataset(ep, tc)

        fe = FeatureExtractor(self.config)
        df = fe.extract_mutation_features(df)
        df = fe.extract_binding_features(df)

        # Prepare features and labels
        mutation_features = ['length', 'hydrophobicity', 'net_charge', 'aromatic_content', 'polarity'] + [f'aa_{aa}' for aa in self.config.AMINO_ACIDS]
        binding_features = ['affinity_nM', 'affinity_rank', 'log_affinity', 'strong_binder', 'weak_binder'] + \
                           [c for c in df.columns if c.startswith('P2_is_')] + \
                           [c for c in df.columns if c.startswith('P9_is_')]

        all_features = mutation_features + binding_features

        X = df[all_features].fillna(0)
        y = df['immunogenic']

        # Cross-validation
        skf = StratifiedKFold(n_splits=self.config.N_CV_FOLDS, shuffle=True, random_state=self.config.RANDOM_SEED)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestClassifier(
            n_estimators=self.config.N_ESTIMATORS,
            max_depth=self.config.MAX_DEPTH,
            min_samples_split=self.config.MIN_SAMPLES_SPLIT,
            min_samples_leaf=self.config.MIN_SAMPLES_LEAF,
            max_features=self.config.MAX_FEATURES,
            random_state=self.config.RANDOM_SEED,
            n_jobs=-1,
            class_weight='balanced'
        )

        cv_aurocs, cv_auprcs, cv_f1s, cv_briers = [], [], [], []

        for train_idx, test_idx in skf.split(X_scaled, y):
            rf.fit(X_scaled[train_idx], y.iloc[train_idx])
            y_proba = rf.predict_proba(X_scaled[test_idx])[:, 1]
            y_pred = rf.predict(X_scaled[test_idx])

            cv_aurocs.append(roc_auc_score(y.iloc[test_idx], y_proba))
            cv_auprcs.append(average_precision_score(y.iloc[test_idx], y_proba))
            cv_f1s.append(f1_score(y.iloc[test_idx], y_pred))
            cv_briers.append(brier_score_loss(y.iloc[test_idx], y_proba))

        results = {
            'auroc': float(np.mean(cv_aurocs)),
            'auprc': float(np.mean(cv_auprcs)),
            'f1': float(np.mean(cv_f1s)),
            'brier': float(np.mean(cv_briers)),
            'auroc_per_fold': cv_aurocs,
            'auprc_per_fold': cv_auprcs,
            'f1_per_fold': cv_f1s,
            'brier_per_fold': cv_briers,
        }

        # Train final model on full data
        rf.fit(X_scaled, y)

        save_model_path = save_model_path or str(Path(self.config.MODELS_DIR) / 'neorank_model.pkl')
        save_results_path = save_results_path or str(Path(self.config.RESULTS_DIR) / 'neorank_cv_results.csv')

        # Persist model
        model_data = {
            'model': rf,
            'scaler': scaler,
            'features': all_features,
            'cv_results': results
        }
        with open(save_model_path, 'wb') as fh:
            pickle.dump(model_data, fh)

        # Save CV results CSV
        import pandas as pd
        folds = list(range(1, self.config.N_CV_FOLDS + 1))
        results_df = pd.DataFrame({
            'Fold': folds + ['Mean'],
            'AUROC': cv_aurocs + [results['auroc']],
            'AUPRC': cv_auprcs + [results['auprc']],
            'F1': cv_f1s + [results['f1']],
            'Brier': cv_briers + [results['brier']]
        })
        results_df.to_csv(save_results_path, index=False)

        print(f"Model saved to: {save_model_path}")
        print(f"CV results saved to: {save_results_path}")

        return model_data


class NeoRankPredictor:
    """Load a trained NeoRank model and run predictions."""

    def __init__(self, model_path: str = None, config: 'Config' = None):
        self.config = config or Config()
        model_path = model_path or str(Path(self.config.MODELS_DIR) / 'neorank_model.pkl')

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, 'rb') as fh:
            data = pickle.load(fh)

        self.model = data['model']
        self.scaler = data['scaler']
        self.features = data['features']

    def predict_df(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Accept a dataframe with columns ['peptide','HLA'] and return predictions."""
        fe = FeatureExtractor(self.config)
        df2 = fe.extract_mutation_features(df)
        df2 = fe.extract_binding_features(df2)

        X = df2[self.features].fillna(0)
        X_scaled = self.scaler.transform(X)

        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)[:, 1]

        out = df.copy()
        out['prediction'] = preds
        out['immunogenicity_score'] = probs
        out['prediction_label'] = out['prediction'].map({0: 'Non-Immunogenic', 1: 'Immunogenic'})
        return out

    def predict_csv(self, input_csv: str, output_csv: str = None) -> str:
        import pandas as pd
        df = pd.read_csv(input_csv)
        out = self.predict_df(df)
        output_csv = output_csv or input_csv.replace('.csv', '_predictions.csv')
        out.to_csv(output_csv, index=False)
        return output_csv


__all__ = [
    'Config',
    'DataPreprocessor',
    'FeatureExtractor',
    'NeoRankTrainer',
    'NeoRankPredictor',
]
