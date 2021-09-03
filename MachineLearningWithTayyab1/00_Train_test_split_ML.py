# (00)**********************************What is train test Split *******************************************

# -----Difinition----train_test_split() is a scikit-learn class help to Split data in to 4 datasets (like: X_train, X_test, y_train,y_test)

# ----Practical---

# ---importing librarires
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# ---loading the dataset
df = sns.load_dataset("titanic")
df2=df[['survived','pclass','age', 'parch']]
df3 = df2.fillna(df2.mean())
# print(df3)

# ----converting features and labels
X=df3.drop('survived',axis=1)
y=df3['survived']
print("Shape of X : ", X.shape)
print("Shape of y : ", y.shape)

# ----spliting the data into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=51)
print("Shape of X_train = ",X_train.shape)
print("Shape of X_test = ",X_test.shape)
print("Shape of y_train = ",y_train.shape)
print("Shape of y_test = ",y_test.shape)