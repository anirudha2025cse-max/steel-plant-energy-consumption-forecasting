"""Smoke tests that do not require the organizer's private training CSV."""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import predict, features, INPUT_COLUMNS, TARGET


def test_predictions():
    bundle = joblib.load(ROOT / 'models/model.joblib')
    history = bundle['history'].copy()
    assert list(history.columns) == INPUT_COLUMNS
    assert TARGET not in history

    # Independent test rows: missing lag history must not crash prediction.
    sample = history.iloc[50:65].copy().reset_index(drop=True)
    output = predict(sample)
    assert output.shape == (15,)
    assert np.isfinite(output).all() and (output >= 0).all()

    # The inference interface is independent of the target column.
    sample.loc[2, 'CO2(tCO2)'] = np.nan
    assert np.isfinite(predict(sample)).all()

    # Simulated consecutive midnight transition using the final training row.
    midnight = history.tail(1).copy().reset_index(drop=True)
    midnight.loc[0, 'date'] = '19/10/2018'
    midnight.loc[0, 'NSM'] = 0
    old_style = features(midnight, history)
    assert np.isfinite(old_style.loc[0, 'usage_lag_1'])
    midnight.loc[0, 'date'] = '20/10/2018'
    standard_style = features(midnight, history)
    assert np.isfinite(standard_style.loc[0, 'usage_lag_1'])
    assert old_style.loc[0, 'dayofweek'] == standard_style.loc[0, 'dayofweek']
    assert len(predict(midnight)) == 1

    try:
        predict(sample.drop(columns=['Usage_kWh']))
    except ValueError:
        pass
    else:
        raise AssertionError('Missing required column was not rejected')


if __name__ == '__main__':
    test_predictions()
    print('All inference tests passed.')
