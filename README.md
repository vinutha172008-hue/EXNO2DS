# EXNO2DS
# AIM:

To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("titanic_dataset.csv");
print(df.head(10));
```
<img width="628" height="633" alt="Screenshot 2026-11-22 190556" src="https://github.com/user-attachments/assets/c5fbaaef-694c-472f-9c2d-60f57ede1e7b" />

```
print(df.info());

```
<img width="495" height="362" alt="Screenshot 2026-11-22 190733" src="https://github.com/user-attachments/assets/eec7cd2e-8c2c-4de4-a5ac-8364425da282" />


```
print(df.shape);
```
<img width="125" height="50" alt="Screenshot 2026-11-22 190958" src="https://github.com/user-attachments/assets/4454cd5a-e11a-491b-81ec-8007889cf0b5" />

```
df.set_index("PassengerId",inplace=True)
df.describe()
```
<img width="640" height="245" alt="Screenshot 2026-11-22 191905" src="https://github.com/user-attachments/assets/50fe4ba0-e218-48f3-a7b5-3dbc8bf4a47d" />

```
print("Data types:")
display(df.dtypes.to_frame("Data Type"))
```
<img width="280" height="391" alt="Screenshot 2026-11-22 191949" src="https://github.com/user-attachments/assets/72a9df83-9076-49f5-b90a-58b040428432" />

```

print("Missing values:")
missing=pd.DataFrame({
    "Missing count":df.isnull().sum(),
    "Missing percentage":(df.isna().mean()*100).round(2)
})
display(missing.sort_values("Missing percentage",ascending=False))
```
<img width="373" height="373" alt="Screenshot 2026-11-22 192024" src="https://github.com/user-attachments/assets/680e939e-c5a6-49b5-bbe0-ea6a884c9c56" />

```
print("Exact duplicate rows:",df.duplicated().sum())
print("Numeric summary")
display(df.describe())

```

<img width="620" height="322" alt="Screenshot 2026-11-22 192442" src="https://github.com/user-attachments/assets/a9a7a2a7-6b3e-41df-abe3-1a846ab38d94" />
```
df.nunique()
```

<img width="205" height="226" alt="Screenshot 2026-11-22 192522" src="https://github.com/user-attachments/assets/e6087090-4e7a-456b-a69d-bcd3d563c5de" />
```
df["Survived"].value_counts()
```

<img width="327" height="75" alt="Screenshot 2026-11-22 192601" src="https://github.com/user-attachments/assets/5a8701f4-e7a3-4f36-ab81-08c4230a50d9" />

```
per=(df["Survived"].value_counts()/df.shape[0]*100).round(2)
per
```

<img width="285" height="87" alt="Screenshot 2026-11-22 192641" src="https://github.com/user-attachments/assets/dcc6c2d4-a716-4c02-99c0-796492a21294" />

```
sns.countplot(data=df, x="Survived")
```

<img width="578" height="405" alt="image" src="https://github.com/user-attachments/assets/5ff3fabc-197a-4c78-9f87-d30365ca6d28" />
```
df.rename(columns={'Sex':'Gender'},inplace=True) 
df
```

<img width="672" height="390" alt="Screenshot 2026-11-22 192931" src="https://github.com/user-attachments/assets/39025b16-cb44-4ca5-a4ad-feb33cfe71ed" />

```
sns.catplot(x='Gender',col='Survived',kind='count',data=df,height=5,aspect=.7)
```

<img width="642" height="452" alt="Screenshot 2026-11-22 193042" src="https://github.com/user-attachments/assets/4c6c4fad-c770-4d61-9697-210f8eaa04d6" />

```
sns.catplot(x='Survived', hue="Gender", data=df, kind = "count")
```

<img width="570" height="447" alt="Screenshot 2026-11-22 193152" src="https://github.com/user-attachments/assets/251fbf81-f49e-4f4f-9fa7-c720311faac6" />

```
df.boxplot(column='Age',by='Survived')
```

<img width="553" height="432" alt="Screenshot 2026-11-22 193250" src="https://github.com/user-attachments/assets/028a4c8d-098c-4dfd-a073-6122f2522d2d" />

```
sns.scatterplot(x=df['Age'],y=df['Fare'])

```


<img width="541" height="412" alt="Screenshot 2026-11-22 193355" src="https://github.com/user-attachments/assets/e921a6f1-8e3b-4622-9b62-8245c1d67edb" />

```
sns.jointplot(x="Age",y="Fare", data=df)
```

<img width="563" height="537" alt="Screenshot 2026-11-22 193458" src="https://github.com/user-attachments/assets/7c7c1a09-a928-4469-a902-be0a190c6309" />

```
fig, ax1= plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x="Pclass",y='Age', hue='Gender', data=df)
```

<img width="652" height="387" alt="Screenshot 2026-11-22 193536" src="https://github.com/user-attachments/assets/0400ee9f-fd46-4a30-b837-2ad5d97501e0" />

```
sns.catplot(data=df, col ="Survived", x = "Gender", hue="Pclass", kind = "count")
```

<img width="672" height="352" alt="Screenshot 2026-11-22 193632" src="https://github.com/user-attachments/assets/6ed0aa2c-8af1-47f9-b7e9-729c1155846a" />

```
corr=df.corr(numeric_only=True)

sns.heatmap(corr,annot=True,cmap='coolwarm')
```

<img width="532" height="390" alt="Screenshot 2026-11-22 193717" src="https://github.com/user-attachments/assets/5fb2d620-a2a9-4707-8380-d760fec6b94a" />

```
sns.pairplot(df)
```

<img width="687" height="702" alt="Screenshot 2026-11-22 193817" src="https://github.com/user-attachments/assets/42863562-07e4-428c-9412-83239010b46c" />

## INTERPRETATION
1. The Titanic dataset contains 891 rows and 12 columns.
2. The first 10 records are displayed.
3. The dataset contains integer, float and object data types.
4. Missing values are found in Age, Cabin and Embarked columns.
5. No duplicate records are found in the dataset.
6. Out of 891 passengers, 342 survived and 549 did not survive.
7. The survival percentage is 38.38% and the death percentage is 61.62%.
8. Female passengers had a higher survival rate than male passengers.
9. The graphs show the distribution of age, fare and passenger class.
10. The heatmap and pairplot show the relationships between numerical columns.
# RESULT
Thus, the Exploratory Data Analysis on the given data set was performed successfully.   
