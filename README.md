# Steel Plant Energy Consumption Forecasting — Track 2

Predict the next 15-minute electricity usage (`next_interval_Usage_kWh`) from 11 current telemetry columns and causal history. **Primary metric: RMSE; secondary: MAE.**

## Results (chronological split, no shuffling)

70% train / 15% validation / 15% final holdout. A one-observation purge prevents the training label from referencing the first observation of the following split. The selected model is LightGBM with 350 estimators.

- Validation RMSE: **8.5772 kWh**; MAE: **4.0902 kWh**.
- Untouched holdout RMSE: **8.7751 kWh**; MAE: **4.0793 kWh**.
- Holdout persistence baseline RMSE: **12.5233 kWh**; MAE: **5.5566 kWh**.

These are local scores, not organizer leaderboard results. See `reports/validation_comparison.csv` and `reports/holdout_results.csv`.

## Installation

Python 3.10+ recommended. Run `pip install -r requirements.txt`.

## Prediction without retraining

```python
import pandas as pd
from src.inference import predict

test_df = pd.read_csv('test.csv') # exactly the 11 input columns, target absent
predictions = predict(test_df)
print(predictions)
```

`models/model.joblib` is a saved bundle containing the trained LightGBM model, the feature schema, and the last 672 training observations for boundary history. The function uses saved training-tail history only when the first test observation follows the last training observation by 15 minutes. It supports both the dataset's unusual midnight date labels and conventional midnight timestamps; otherwise it computes history from the test data alone. For the first rows without historical context, lag features may be missing, which LightGBM can handle. The function never reads the target column. The function assumes test rows are already in chronological order as specified by the organizers.

## Verify the saved model

From the project root, run:

```bash
python tests/test_inference.py
```

These tests use the saved model and do **not** require `train.csv`. They check 11-column inference, numeric predictions, missing-value handling, and midnight continuity.

## Reproducing training

Place the organizers' `train.csv` in this repository's root (alongside `train.py`), then run `python train.py`. **The CSV is intentionally excluded from the GitHub package**, since judges already have the competition data. This overwrites the saved model and validation reports. The supplied model is already trained and does not need retraining for evaluation.

## Files

- `train.py`: model comparison, chronological validation, untouched holdout, and final training
- `src/inference.py`: feature engineering and `predict(test_df)`
- `models/model.joblib`: trained model artifact (required by organizers)
- `reports/methodology.md` and `reports/methodology.pdf`: 1–2 page methodology summary
- `notebooks/energy_forecasting_model.ipynb`: notebook guide for reproducible training
- `tests/test_inference.py`: basic inference tests

## Model and report provenance

The report PDF is the team-approved two-page methodology summary and is included **unchanged**. `reports/methodology.md` contains a matching editable transcription. The saved model and CSV scores are from the earlier Phase 5 training run; the inference code was checked for both conventional and organizer-style midnight transitions. No new leaderboard score is claimed.

## Caveats

This is a single chronological holdout, not evidence of guaranteed hidden-test ranking. The dataset has unusual midnight timestamp labels; the feature code corrects 00:00 labels by adding one day. The model uses observed current `Usage_kWh` and past observed usages, never future usages. Test rows must include actual `Usage_kWh` readings.
