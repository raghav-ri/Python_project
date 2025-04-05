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
print(df.isnull().sum().sum())

#  0   ID                                3000 non-null   int64  
#  1   Name                              3000 non-null   object 
#  2   Age                               2850 non-null   float64
#  3   Gender                            3000 non-null   object 
#  4   State                             3000 non-null   object 
#  5   City Type                         3000 non-null   object 
#  6   Disease                           2098 non-null   object
#  7   Cause                             3000 non-null   object
#  8   Hospital Visits Per Year          2700 non-null   float64
#  9   Out-of-Pocket Expenses (Rs)       2550 non-null   float64
#  10  Expense Covered by Insurance (%)  2700 non-null   float64
#  11  Government Hospital Use           3000 non-null   object
#  12  BMI                               2700 non-null 
df.drop_duplicates(inplace=True)
df.loc[:,'Age']=df['Age'].fillna(df['Age'].mean())
df.loc[:,'Hospital Visits Per Year']=df['Hospital Visits Per Year'].fillna(df['Hospital Visits Per Year'].mean())

df.loc[:,'Disease']=df['Disease'].fillna("Unknown Disease")
df.loc[:,'Out-of-Pocket Expenses (Rs)']=df['Out-of-Pocket Expenses (Rs)'].fillna(df['Out-of-Pocket Expenses (Rs)'].mean())

# If the data has a normal distribution (symmetrical), use the mean.
# If the data has outliers or is skewed, use the median.
df.loc[:,'Expense Covered by Insurance (%)']=df['Expense Covered by Insurance (%)'].fillna(df['Expense Covered by Insurance (%)'].median())
df.loc[:,'BMI']=df['BMI'].fillna(df['BMI'].mean())

#converting data type of BMI and Hospital Visit Per Year
df.loc[:,'BMI']=df['BMI'].astype(int)
df.loc[:,'Hospital Visits Per Year']=df['Hospital Visits Per Year'].astype(int)

print(df.info())

#  #   Column                            Non-Null Count  Dtype
# ---  ------                            --------------  -----
#  0   ID                                3000 non-null   int64
#  1   Name                              3000 non-null   object
#  2   Age                               3000 non-null   float64
#  3   Gender                            3000 non-null   object
#  4   State                             3000 non-null   object
#  5   City Type                         3000 non-null   object
#  6   Disease                           3000 non-null   object
#  7   Cause                             3000 non-null   object
#  8   Hospital Visits Per Year          3000 non-null   float64
#  9   Out-of-Pocket Expenses (Rs)       3000 non-null   float64
#  10  Expense Covered by Insurance (%)  3000 non-null   float64
#  11  Government Hospital Use           3000 non-null   object
#  12  BMI                               3000 non-null   float64


#Patient count with particular disease
print("Disease Count: ")
print(df['Disease'].value_counts())

#Age-wise Disease 
print("Chances of Disease at Paticular Age: ")
print(df.groupby('Disease')['Age'].value_counts())

#Frequency for Doctor Consultant for Particular Disease
print(df.groupby('Disease')['Hospital Visits Per Year'].mean())

#Average BMI for any Particular Disease
print(df.groupby('Disease')['BMI'].mean())

#Distribution Of Disease (Rural Vs Urban)
print(df.groupby('City Type')['Disease'].value_counts())

df=df[(df['Expense Covered by Insurance (%)']>=0) & (df['Expense Covered by Insurance (%)']<=100)]


#Correlation Between BMI and Disease
print("Relation between Expenses incured from pocket or from insurance")
print(df['Expense Covered by Insurance (%)'].corr(df['Out-of-Pocket Expenses (Rs)']))


#Cause and disease
print("Disease and their main cause and patient affected from that ")
print(df.groupby('Disease')['Cause'].value_counts())


#Average Expense occured during treatment of Disease
print(df.groupby('Disease')['Out-of-Pocket Expenses (Rs)'].mean())







