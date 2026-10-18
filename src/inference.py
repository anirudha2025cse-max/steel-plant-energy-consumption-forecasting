"""Track 2 inference. Input: chronological 11-column feature DataFrame."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

TARGET = 'next_interval_Usage_kWh'
INPUT_COLUMNS = ['date','Usage_kWh','Lagging_Current_Reactive.Power_kVarh','Leading_Current_Reactive_Power_kVarh','CO2(tCO2)','Lagging_Current_Power_Factor','Leading_Current_Power_Factor','NSM','WeekStatus','Day_of_week','Load_Type']
LAGS = [1,2,3,4,8,12,24,48,96,192,672]
WINDOWS = [4,8,24,96]

def features(data, history=None):
    if not isinstance(data, pd.DataFrame):
        raise TypeError('Expected a pandas DataFrame')
    missing = set(INPUT_COLUMNS)-set(data.columns)
    if missing:
        raise ValueError(f'Missing required feature columns: {sorted(missing)}')
    d = data[INPUT_COLUMNS].reset_index(drop=True).copy()
    h = d.iloc[:0].copy() if history is None else history[INPUT_COLUMNS].reset_index(drop=True).copy()
    # Compare the raw date plus NSM rather than assuming midnight is dated
    # consistently. The organizer's original data labels 00:00 with the
    # previous day's date, while a future test may use conventional dates.
    def boundary_time(frame, index, previous=None):
        day = pd.to_datetime(frame['date'].iloc[index], dayfirst=True, errors='coerce')
        seconds = pd.to_numeric(frame['NSM'].iloc[index], errors='coerce')
        if pd.isna(day) or pd.isna(seconds):
            return pd.NaT
        stamp = day.normalize() + pd.to_timedelta(seconds, unit='s')
        if previous is not None and pd.notna(previous) and stamp == previous - pd.Timedelta(hours=23, minutes=45):
            stamp += pd.Timedelta(days=1)
        return stamp

    if len(h) and len(d):
        a = boundary_time(h, len(h)-1)
        b = boundary_time(d, 0, previous=a)
        if pd.isna(a) or pd.isna(b) or b-a != pd.Timedelta(minutes=15):
            h = h.iloc[:0]
    combined = pd.concat([h,d],ignore_index=True)
    # The supplied midnight records have an old date label. Correct only
    # actual backwards midnight transitions; ordinary midnight dates stay put.
    stamps = []
    prev = None
    for i in range(len(combined)):
        stamp = boundary_time(combined, i, previous=prev)
        stamps.append(stamp)
        prev = stamp
    timestamps = pd.Series(pd.to_datetime(stamps))
    x = combined.drop(columns=['date','WeekStatus','Day_of_week','Load_Type']).copy()
    nsm = pd.to_numeric(combined['NSM'],errors='coerce')
    x['hour'] = nsm // 3600
    x['quarter'] = nsm // 900
    x['dayofweek'] = timestamps.dt.dayofweek
    x['is_weekend'] = (timestamps.dt.dayofweek>=5).astype(float)
    x['hour_sin'] = np.sin(2*np.pi*nsm/86400)
    x['hour_cos'] = np.cos(2*np.pi*nsm/86400)
    x['day_sin'] = np.sin(2*np.pi*timestamps.dt.dayofweek/7)
    x['day_cos'] = np.cos(2*np.pi*timestamps.dt.dayofweek/7)
    usage = pd.to_numeric(combined['Usage_kWh'],errors='coerce')
    for lag in LAGS:
        x[f'usage_lag_{lag}'] = usage.shift(lag)
    for window in WINDOWS:
        x[f'usage_mean_{window}'] = usage.rolling(window,min_periods=1).mean()
        x[f'usage_std_{window}'] = usage.rolling(window,min_periods=2).std()
    x['usage_diff_1'] = usage.diff()
    x = x.apply(pd.to_numeric, errors='coerce')
    return x.iloc[len(h):].reset_index(drop=True)

def predict(test_df, model_path=None):
    """Return one numeric next-interval prediction per test row, in input order."""
    path = Path(model_path) if model_path else Path(__file__).resolve().parents[1] / 'models' / 'model.joblib'
    bundle = joblib.load(path)
    x = features(test_df, bundle.get('history'))
    x = x.reindex(columns=bundle['feature_columns'])
    return np.maximum(0, bundle['model'].predict(x))
