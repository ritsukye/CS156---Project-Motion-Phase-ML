# CS156---Project-Motion-Phase-ML
This project automatically identifies an athlete’s motion phase via wearable biosensor data. The expected outcome is a supervised machine learning model capable of accurately predicting the athlete’s motion phase based solely on sensor readings, demonstrating how wearable sensor devices can support real-time athletic monitoring.

The dataset contains college athlete tracking. Specific age, gender, and ethnicit y data not available in the used dataset version.

## Preprocessing Handoff

The preprocessing pipeline is implemented in `scripts/preprocessing.py`.

Run from project root: `python3 scripts/preprocessing.py`

### Artifacts produced per run

- Log artifact: `out/preprocessing_artifact.log`
- Segmented window dataset: `out/segmented_windows.csv`

### For feature engineering / modeling teammate

- Suggested target: `Event_Label` (window label in segmented output)
- Suggested input features: sensor window features (for example `*_mean`, `*_std`, plus optional `n_samples`, `label_purity`)
- Leakage prevention: split by `Athlete_ID` (or strict time blocks), not random row splits.
