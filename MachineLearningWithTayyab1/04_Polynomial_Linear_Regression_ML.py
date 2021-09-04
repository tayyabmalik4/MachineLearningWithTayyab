# (04)********************Polynomial Linear Regression in Machine Learning Using Python********************************

# --------Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


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


# *****************Apply Polynomial Model 
poly_reg = PolynomialFeatures(degree=2)
poly_reg.fit(X_train)
X_train_poly = poly_reg.transform(X_train)
X_test_poly = poly_reg.transform(X_test)
# print(X_train_poly.shape)
# print(X_test_poly.shape)


# ****************Now finally apply Linear Regression model
lr = LinearRegression()
lr.fit(X_train_poly,y_train)
# print(lr.score(X_test_poly, y_test))


# **************Now Prediccted the values in the test data
y_pred = lr.predict(X_test_poly)
# print(y_pred)
# print(y_test)


# *************Now check the mean and root mean square error
mse = mean_squared_error(y_test,y_pred)
print(mse)
rmse=np.sqrt(mse)
print(rmse)
