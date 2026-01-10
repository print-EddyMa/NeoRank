# Models Directory

Store trained NeoRank models here.

## Model Storage

Save trained Random Forest models as pickle files:

```python
import joblib
joblib.dump(model, 'models/neorank_model.pkl')
```

## Loading Models

```python
import joblib
model = joblib.load('models/neorank_model.pkl')
```

## Model Naming Convention

- `neorank_model.pkl` - Default trained model
- `neorank_cv_fold_*.pkl` - Cross-validation fold models
- `neorank_retrained_YYYYMMDD.pkl` - Date-stamped retraining

## Storage Notes

- Models are excluded from git (.gitignore)
- Large files: ~10-50 MB per model
- Keep backup copies of production models
