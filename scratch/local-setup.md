---
skill: local-setup
dataset_ref: noopurbhatt/retail-store-sales-forecasting-dataset
data_path: data/raw/
status: ready
---

## 데이터 구조
- sales.csv: shape=(156000, 5) — 타깃: weekly_sales
- features.csv: shape=(7800, 14) — 외부 피처 (온도, 유가, 마크다운, CPI, 실업률)
- stores.csv: shape=(50, 4) — 매장 메타 (타입, 크기, 지역)

## 컬럼
- target: weekly_sales
- date range: 2022-01-01 ~
- stores: 50개, departments: 복수
