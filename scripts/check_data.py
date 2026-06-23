import pandas as pd
from pathlib import Path

csv_files = list(Path('data/raw').rglob('*.csv'))
print(f"CSV 파일: {len(csv_files)}개")
for f in csv_files:
    df = pd.read_csv(f)
    print(f"\n{'='*50}")
    print(f"{f.name}: shape={df.shape}")
    print(df.dtypes.to_string())
    print(f"\n결측치:\n{df.isnull().sum().to_string()}")
    print(f"\n샘플:\n{df.head(3).to_string()}")
