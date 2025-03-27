import pandas as pd;
df=pd.read_csv(r'C:\Users\RAMAN\Desktop\healthcare_data.csv');
print(df.info())
print(df.head())
print(df.tail())
print(df.describe())
print(df.columns)
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
