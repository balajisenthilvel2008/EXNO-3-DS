## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from category_encoders import BinaryEncoder
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
data=pd.read_csv(r"C:\Users\acer\Downloads\data.csv")
df=pd.DataFrame(data)
df

```

![alt text](image.png)

```
df.isnull()

```

![alt text](image-1.png)

```
le=LabelEncoder()
df3=pd.DataFrame()
df3["City"]=df["City"]
df3["City_encoded"]=le.fit_transform(df['City'])
print(df3)
```

![alt text](image-2.png)

```
df2=pd.DataFrame()
df2['Ord_2']=df["Ord_2"]
education=['High School','Diploma','Bachelors','Masters','PhD']
enc=OrdinalEncoder(categories=[education])
encoded=enc.fit_transform(df2[['Ord_2']])
df2['Encoded']=encoded
print(df2)
```

![alt text](image-3.png)

```
df4=pd.DataFrame()
df4['Ord_1']=df["Ord_1"]
ohe=OneHotEncoder(sparse_output=False)
enc=pd.DataFrame(ohe.fit_transform(df[["Ord_1"]]))
df4=pd.concat([df4,enc],axis=1)
df4
```

![alt text](<Screenshot 2025-10-16 081005.png>)

```
df5=pd.DataFrame()
df5['City']=df['City']
be=BinaryEncoder(cols=['City'])
encoded=be.fit_transform(df['City'])
df5=pd.concat([df5,encoded],axis=1)
print(df5)
```

![alt text](image-4.png)

```
df=pd.DataFrame(pd.read_csv(r"C:\Users\acer\Downloads\Data_to_Transform.csv"))
df

```

![alt text](image-5.png)

```
df['Highly PositiveSkew']=1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

```

![alt text](image-6.png)

```
df['Moderate PositiveSkew']=np.sqrt(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

```
![alt text](image-7.png)

```
df['Highly Positive Skew']=np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
```

![alt text](image-8.png)

```
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df["Moderate Negative Skew"]=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```

![alt text](image-9.png)

# RESULT:
Thus Feature encoding and feature transformation has been done successfully with a given data set

       
