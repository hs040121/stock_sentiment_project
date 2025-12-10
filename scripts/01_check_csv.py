import pandas as pd

CSV_PATH = "../data/raw/naver_board_kospi100_cleaned.csv"

df = pd.read_csv(CSV_PATH, encoding="utf-8")

print("\n=== CSV Columns ===")
print(df.columns.tolist())

print("\n=== 샘플 10개 ===")
print(df.head(10))

print("\n=== 결측값 ===")
print(df.isnull().sum())
