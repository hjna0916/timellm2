import pandas as pd

df = pd.read_csv("md_data_gnn_lite_with_date_reordered.csv")
print("전체 컬럼 수:", len(df.columns))
print(df.columns)