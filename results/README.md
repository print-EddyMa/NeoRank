# Results Directory

Output directory for NeoRank predictions, figures, and benchmarks.

## Subdirectories

- **figures/** - Visualization outputs (plots, charts)
- **tables/** - Tabular results (CSV, Excel)
- **benchmarks/** - Benchmark comparison results

## Result Types

### Figures
- Feature importance plots
- ROC/PR curves
- Confusion matrices
- UMAP/t-SNE visualizations

### Tables
- Predictions with scores
- Cross-validation results
- Benchmark comparisons
- Statistical summaries

### Benchmarks
- Performance metrics across tools
- Sensitivity/specificity analysis
- Dataset-specific comparisons

## File Organization

Results are automatically organized by:
- Analysis type (training, prediction, evaluation)
- Dataset name
- Date/timestamp

Example:
```
results/
├── figures/
│   ├── feature_importance_20250110.pdf
│   └── roc_curve_test_set.pdf
├── tables/
│   ├── cv_results_10fold.csv
│   └── predictions_sample.csv
└── benchmarks/
    └── comparison_vs_netmhcpan.csv
```

## Size Considerations

Results files are typically:
- Figures: 1-5 MB each
- Tables: 0.1-10 MB each
- Total: Can grow with analysis

Files are excluded from git (.gitignore) to prevent bloat.
