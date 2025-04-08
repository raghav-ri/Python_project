import pandas as pd;
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# hospital preferred in Rural Area and Urban Area 
print(df.groupby('City Type')['Government Hospital Use'].value_counts())

print(df.info())

#  #   Column                            Non-Null Count  Dtype
# ---  ------                            --------------  -----
#  0   ID                                2999 non-null   int64
#  1   Name                              2999 non-null   object
#  2   Age                               2999 non-null   float64
#  3   Gender                            2999 non-null   object
#  4   State                             2999 non-null   object
#  5   City Type                         2999 non-null   object
#  6   Disease                           2999 non-null   object
#  7   Cause                             2999 non-null   object
#  8   Hospital Visits Per Year          2999 non-null   float64
#  9   Out-of-Pocket Expenses (Rs)       2999 non-null   float64
#  10  Expense Covered by Insurance (%)  2999 non-null   float64
#  11  Government Hospital Use           2999 non-null   object
#  12  BMI                               2999 non-null   float64

#Box plot for detecting outlier in Insurance Cover
sns.boxplot(y='Expense Covered by Insurance (%)', data=df, color='salmon')
plt.show()


#Box plot for detecting outlier in age
sns.boxplot(y='Age',data=df,color='red')
plt.show()

#Disease Count
df['Disease'].value_counts().head(10).plot(kind='bar',color='skyblue')
plt.show()


#Check how BMI is distributed among patients.
plt.hist(df['BMI'],bins=20,color='mediumseagreen',edgecolor='black')
plt.title("Distribution Of BMI")
plt.xlabel("BMI")
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#Show how different diseases vary across city types.
pd.crosstab(df['Disease'],df['City Type']).plot(kind='bar',stacked=True )
plt.show()

#Show how different diseases vary across city types.
df['Government Hospital Use'].value_counts().plot.pie(autopct="%1.1f",color=['red','gold'])
plt.show()

#Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()


# Gender-wise Disease Analysis
sns.countplot(x='Disease', hue='Gender', data=df)
plt.xticks(rotation=90)
plt.title("Disease Count by Gender")
plt.show()


#Government vs. Private Hospital Use by Disease
pd.crosstab(df['Disease'], df['Government Hospital Use']).plot(kind='bar', stacked=True, figsize=(12,6))
plt.title("Hospital Type Preference per Disease")
plt.ylabel("Patient Count")
plt.xticks(rotation=90)
plt.show()


#Age Distribution for Specific Diseases (e.g., Diabetes or Cancer)
sns.histplot(data=df[df['Disease'] == 'Diabetes'], x='Age', bins=15, kde=True, color='orange')
plt.title("Age Distribution - Diabetes Patients")
plt.show()



