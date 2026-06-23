import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc')
plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
plt.rcParams['axes.unicode_minus'] = False
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

Path('outputs').mkdir(exist_ok=True)

# ── 데이터 로드 & 병합
sales = pd.read_csv('data/raw/sales.csv', parse_dates=['date'])
features = pd.read_csv('data/raw/features.csv', parse_dates=['date'])
stores = pd.read_csv('data/raw/stores.csv')

df = sales.merge(features, on=['store_id', 'date', 'is_holiday'], how='left')
df = df.merge(stores, on='store_id', how='left')
df = df.sort_values(['store_id', 'department', 'date']).reset_index(drop=True)

# ── 피처 엔지니어링
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week.astype(int)
df['quarter'] = df['date'].dt.quarter

season_map = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
df['season_num'] = df['season'].map(season_map)

store_type_map = {'A': 0, 'B': 1, 'C': 2}
df['store_type_num'] = df['store_type'].map(store_type_map)

region_map = {r: i for i, r in enumerate(df['region'].unique())}
df['region_num'] = df['region'].map(region_map)

# lag features (같은 store+dept 그룹 내)
df = df.sort_values(['store_id', 'department', 'date'])
df['lag_1'] = df.groupby(['store_id', 'department'])['weekly_sales'].transform(lambda x: x.shift(1))
df['lag_4'] = df.groupby(['store_id', 'department'])['weekly_sales'].transform(lambda x: x.shift(4))
df['lag_13'] = df.groupby(['store_id', 'department'])['weekly_sales'].transform(lambda x: x.shift(13))
df['rolling_mean_4'] = df.groupby(['store_id', 'department'])['weekly_sales'].transform(lambda x: x.shift(1).rolling(4).mean())
df['rolling_mean_12'] = df.groupby(['store_id', 'department'])['weekly_sales'].transform(lambda x: x.shift(1).rolling(12).mean())
df['rolling_std_4'] = df.groupby(['store_id', 'department'])['weekly_sales'].transform(lambda x: x.shift(1).rolling(4).std())

KEEP_COLS = [
    'store_id', 'department', 'date', 'store_type_num', 'store_size', 'region_num',
    'year', 'month', 'week', 'quarter', 'season_num', 'is_holiday',
    'temperature', 'fuel_price',
    'markdown_1', 'markdown_2', 'markdown_3', 'markdown_4', 'markdown_5',
    'cpi', 'unemployment',
    'lag_1', 'lag_4', 'lag_13', 'rolling_mean_4', 'rolling_mean_12', 'rolling_std_4',
    'weekly_sales'
]
df = df[KEEP_COLS].dropna().reset_index(drop=True)

# ── Train / Test split (마지막 12주 = 테스트)
cutoff = df['date'].max() - pd.Timedelta(weeks=12)
train = df[df['date'] <= cutoff]
test = df[df['date'] > cutoff]

FEATURES = [
    'store_id', 'department', 'store_type_num', 'store_size', 'region_num',
    'year', 'month', 'week', 'quarter', 'season_num', 'is_holiday',
    'temperature', 'fuel_price',
    'markdown_1', 'markdown_2', 'markdown_3', 'markdown_4', 'markdown_5',
    'cpi', 'unemployment',
    'lag_1', 'lag_4', 'lag_13', 'rolling_mean_4', 'rolling_mean_12', 'rolling_std_4'
]
TARGET = 'weekly_sales'

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── LightGBM 학습
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'verbosity': -1,
    'n_estimators': 500,
}
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
model = lgb.train(params, dtrain, valid_sets=[dval], callbacks=callbacks)

# ── 평가
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100

print(f"\n=== 모델 성능 ===")
print(f"RMSE:  {rmse:,.0f}")
print(f"MAE:   {mae:,.0f}")
print(f"R²:    {r2:.4f}")
print(f"MAPE:  {mape:.2f}%")

metrics = {'rmse': round(rmse, 2), 'mae': round(mae, 2), 'r2': round(r2, 4), 'mape': round(mape, 2)}
with open('outputs/metrics.json', 'w') as f:
    json.dump(metrics, f)

# ── 피처 중요도
fi = pd.Series(model.feature_importance(importance_type='gain'), index=FEATURES).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
fi.head(15).plot(kind='barh', ax=ax, color='#2563EB')
ax.invert_yaxis()
ax.set_title('Feature Importance (Top 15)', fontsize=13)
ax.set_xlabel('Gain')
plt.tight_layout()
plt.savefig('outputs/model_01_feature_importance.png', dpi=120)
plt.close()

# ── 예측 vs 실제 (샘플: store 1, dept 1)
sample = test[(test['store_id'] == 1) & (test['department'] == 1)].copy()
if len(sample) > 0:
    sample['pred'] = model.predict(sample[FEATURES])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sample['date'], sample['weekly_sales'], label='Actual', color='#2563EB', linewidth=2)
    ax.plot(sample['date'], sample['pred'], label='Predicted', color='#DC2626', linewidth=2, linestyle='--')
    ax.set_title('Actual vs Predicted - Store 1, Dept 1', fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Sales')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/model_02_actual_vs_pred.png', dpi=120)
    plt.close()

# ── 잔차 분포
residuals = y_test - preds
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(residuals, bins=50, color='#2563EB', edgecolor='white', alpha=0.8)
axes[0].set_title('Residual Distribution', fontsize=12)
axes[0].set_xlabel('Residual')
axes[0].axvline(0, color='red', linestyle='--')

axes[1].scatter(preds, residuals, alpha=0.3, color='#7C3AED', s=5)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title('Residuals vs Fitted', fontsize=12)
axes[1].set_xlabel('Fitted Values')
axes[1].set_ylabel('Residuals')
plt.tight_layout()
plt.savefig('outputs/model_03_residuals.png', dpi=120)
plt.close()

print("\n모델링 완료. outputs/ 에 결과 저장됨.")
