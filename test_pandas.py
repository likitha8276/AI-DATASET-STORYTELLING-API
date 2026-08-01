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
print(corr_no_diagonal.max().max())