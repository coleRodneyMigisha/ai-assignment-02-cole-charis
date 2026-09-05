EDA and preprocessing for Known_Patients_01.csv

Run the analysis script which produces plots and prints recommendations:

```bash
python eda_known_patients.py
```

Outputs:
- `plots/` directory with PNG figures (distributions, boxplots, heatmap, pairplot, learning curves)

Notes:
- Script will infer the target column (looks for common names, otherwise picks last column).
- Categorical encoding: One-hot for low-cardinality, frequency encoding for high-cardinality.
