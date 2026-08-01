import pandas as pd
import numpy as np
df = pd.read_csv("tests.csv")
print(df)
print(df.shape)
print(df.columns)
count = 0
for value in df["Data_value"]:
    if(pd.isnull(value)):
        count+=1
print(count)

print(df.isnull().sum().to_dict())
print(df.duplicated().sum())
print(df.describe().to_dict())

corr = df.corr(numeric_only=True)

corr_no_diagonal = corr.where(~np.eye(len(corr),dtype=bool))
overall_max = corr_no_diagonal.max().max()
print(corr_no_diagonal.max().max())
max_col = corr_no_diagonal.max().idxmax()
print(max_col)

max_row = corr_no_diagonal[max_col].idxmax()
print(max_row)

summary = f"The strongest relationship found was between {max_col} and {max_row}, with a correlation of {overall_max:.2f}."
print(summary)