import pandas as pd
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