"""Train, validate and export the Track 2 model. Run: python train.py"""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
sys.path.insert(0,str(Path(__file__).parent / 'src'))
from inference import features, INPUT_COLUMNS, TARGET

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'train.csv'
if not DATA.exists():
    raise FileNotFoundError('Place the organizer-provided train.csv beside train.py')
df=pd.read_csv(DATA)
assert TARGET in df and all(c in df for c in INPUT_COLUMNS)
assert df['Usage_kWh'].shift(-1).iloc[:-1].round(6).equals(df[TARGET].iloc[:-1].round(6)), 'Check target alignment'
X=features(df[INPUT_COLUMNS]); y=df[TARGET]
n=len(df); a=int(n*.70); b=int(n*.85)
models={
 'HistGB_200':HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=25,learning_rate=.07,l2_regularization=1,random_state=42),
 'HistGB_350':HistGradientBoostingRegressor(max_iter=350,max_leaf_nodes=20,learning_rate=.045,l2_regularization=3,random_state=42),
 'XGBoost_350':XGBRegressor(n_estimators=350,max_depth=5,learning_rate=.04,subsample=.85,colsample_bytree=.85,reg_lambda=5,tree_method='hist',n_jobs=4,random_state=42),
 'LightGBM_350':LGBMRegressor(n_estimators=350,num_leaves=20,learning_rate=.035,max_depth=-1,reg_lambda=5,verbosity=-1,n_jobs=4,random_state=42),
 'CatBoost_350':CatBoostRegressor(iterations=350,depth=5,learning_rate=.045,l2_leaf_reg=5,verbose=False,thread_count=4,random_seed=42)
}
rows=[]
for name,model in models.items():
    model.fit(X.iloc[:a-1],y.iloc[:a-1]); p=model.predict(X.iloc[a:b]);
    rmse=float(np.sqrt(mean_squared_error(y.iloc[a:b],p))); mae=float(mean_absolute_error(y.iloc[a:b],p))
    rows.append({'model':name,'validation_RMSE':rmse,'validation_MAE':mae})
    print(name,round(rmse,4),round(mae,4),flush=True)
results=pd.DataFrame(rows).sort_values(['validation_RMSE','validation_MAE'])
(ROOT/'reports').mkdir(exist_ok=True); results.to_csv(ROOT/'reports'/'validation_comparison.csv',index=False)
name=results.iloc[0]['model']; print('Selected:',name,flush=True)
model=models[name]; model.fit(X.iloc[:b-1],y.iloc[:b-1]); p=model.predict(X.iloc[b:]);
hold={'selected_model':name,'holdout_RMSE':float(np.sqrt(mean_squared_error(y.iloc[b:],p))),'holdout_MAE':float(mean_absolute_error(y.iloc[b:],p)),'persistence_RMSE':float(np.sqrt(mean_squared_error(y.iloc[b:],df['Usage_kWh'].iloc[b:]))),'persistence_MAE':float(mean_absolute_error(y.iloc[b:],df['Usage_kWh'].iloc[b:]))}
print('HOLDOUT',hold,flush=True)
pd.DataFrame([hold]).to_csv(ROOT/'reports'/'holdout_results.csv',index=False)
# Train on full labeled dataset only after holdout evaluation.
model=models[name];model.fit(X,y)
bundle={'model':model,'feature_columns':list(X.columns),'history':df[INPUT_COLUMNS].tail(672).copy(),'selected_model':name}
(ROOT/'models').mkdir(exist_ok=True); joblib.dump(bundle,ROOT/'models'/'model.joblib')
from inference import predict
sample=df[INPUT_COLUMNS].iloc[-5:].copy()
assert len(predict(sample))==5
assert np.isfinite(predict(sample)).all()
print('Saved model; inference smoke test passed.',flush=True)
