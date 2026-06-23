import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

fm.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc')
plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
plt.rcParams['axes.unicode_minus'] = False
from pathlib import Path

Path('outputs').mkdir(exist_ok=True)

sales = pd.read_csv('data/raw/sales.csv', parse_dates=['date'])
features = pd.read_csv('data/raw/features.csv', parse_dates=['date'])
stores = pd.read_csv('data/raw/stores.csv')

df = sales.merge(features, on=['store_id', 'date', 'is_holiday'], how='left')
df = df.merge(stores, on='store_id', how='left')
df = df.sort_values(['store_id', 'department', 'date']).reset_index(drop=True)

print(f"병합 shape: {df.shape}")
print(f"날짜 범위: {df['date'].min()} ~ {df['date'].max()}")
print(f"매장 수: {df['store_id'].nunique()}, 부서 수: {df['department'].nunique()}")
print(f"\n기술통계:\n{df['weekly_sales'].describe()}")

# 1. 주간 매출 추이 (전체)
weekly = df.groupby('date')['weekly_sales'].sum().reset_index()
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(weekly['date'], weekly['weekly_sales'] / 1e6, color='#2563EB')
ax.set_title('전체 주간 매출 추이 (단위: 백만)', fontsize=13)
ax.set_xlabel('날짜')
ax.set_ylabel('매출 (백만)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/eda_01_weekly_trend.png', dpi=120)
plt.close()

# 2. 매장 타입별 평균 매출
fig, ax = plt.subplots(figsize=(7, 4))
store_type_sales = df.groupby('store_type')['weekly_sales'].mean().sort_values(ascending=False)
store_type_sales.plot(kind='bar', ax=ax, color=['#2563EB', '#7C3AED', '#DC2626'])
ax.set_title('매장 타입별 평균 주간 매출', fontsize=13)
ax.set_xlabel('매장 타입')
ax.set_ylabel('평균 매출')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda_02_store_type.png', dpi=120)
plt.close()

# 3. 시즌별 매출
fig, ax = plt.subplots(figsize=(7, 4))
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
season_sales = df.groupby('season')['weekly_sales'].mean().reindex(season_order)
season_sales.plot(kind='bar', ax=ax, color=['#16A34A', '#F59E0B', '#EA580C', '#0EA5E9'])
ax.set_title('시즌별 평균 주간 매출', fontsize=13)
ax.set_xlabel('시즌')
ax.set_ylabel('평균 매출')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda_03_season.png', dpi=120)
plt.close()

# 4. 휴일 vs 비휴일 매출
fig, ax = plt.subplots(figsize=(6, 4))
holiday_sales = df.groupby('is_holiday')['weekly_sales'].mean()
holiday_sales.index = ['비휴일', '휴일']
holiday_sales.plot(kind='bar', ax=ax, color=['#6B7280', '#2563EB'])
ax.set_title('휴일 여부에 따른 평균 매출', fontsize=13)
ax.set_xlabel('')
ax.set_ylabel('평균 매출')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda_04_holiday.png', dpi=120)
plt.close()

# 5. 상관관계 히트맵
numeric_cols = ['weekly_sales', 'temperature', 'fuel_price', 'markdown_1',
                'markdown_2', 'markdown_3', 'markdown_4', 'markdown_5',
                'cpi', 'unemployment', 'store_size']
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=ax, annot_kws={'size': 8})
ax.set_title('피처 상관관계 히트맵', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/eda_05_correlation.png', dpi=120)
plt.close()

# 6. 상위 부서별 매출
fig, ax = plt.subplots(figsize=(12, 4))
top_depts = df.groupby('department')['weekly_sales'].mean().nlargest(20)
top_depts.plot(kind='bar', ax=ax, color='#2563EB')
ax.set_title('부서별 평균 주간 매출 (상위 20)', fontsize=13)
ax.set_xlabel('부서')
ax.set_ylabel('평균 매출')
plt.tight_layout()
plt.savefig('outputs/eda_06_top_departments.png', dpi=120)
plt.close()

print("\nEDA 완료. outputs/ 에 그래프 저장됨.")
print(f"가설:\n1. 휴일 매출 > 비휴일 매출: {holiday_sales['휴일'] > holiday_sales['비휴일']}")
print(f"2. Winter 매출 > 다른 시즌: {season_sales.idxmax()} 시즌이 최고")
print(f"3. 매장 타입 A > B > C 순서: {store_type_sales.index.tolist()}")
