# --------Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# -----importing dataset
data=pd.read_csv(r'datasets\\bangalore house price prediction OHE-data.csv')
# ---checking dataset
# print(data.head())

# Creating features and labels
X = data.drop('price', axis=1)
y=data['price']
# print('Shape of X = ', X.shape)
# print("Shape of y = ",y.shape)


# ----Now splitting the data into trainb and test data set
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=51)
# print("Shape of X_train = ", X_train.shape)
# print("Shape of X_train = ", y_train.shape)
# print("Shape of X_train = ", X_test.shape)
# print("Shape of X_train = ", y_test.shape)


# ******************Feature Scaling
sc=StandardScaler()
sc.fit(X_train,y_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


# *****************Linear Regression - ML Model Training
lr = LinearRegression()

lr.fit(X_train,y_train)
# print(lr.coef_)
# print(lr.intercept_)


# **************Now predicted the values and test it
# print(X_test[0,:])
pred=lr.predict([X_test[0,:]])
# print(pred)
# -----check it that the values who is predict is correct or not
# print(lr.predict(X_test))
# print(y_test)


# --------Now printed the Score of Accuracy of model 
print(lr.score(X_test,y_test))