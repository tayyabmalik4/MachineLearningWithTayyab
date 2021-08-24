# (10)**************************Linear Regression Practical in Machine Learning*************************

# -----importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# ------importing datasets of diabetes
diabetes=datasets.load_diabetes()
# print(diabetes)

# ----for checking the columns of the diabetes
chk_clm=diabetes.keys()
# print(chk_clm)
# ----these are the columns of diabetes
# ----['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename']

# ----for checking the data of DESCR column
# print(diabetes.DESCR)

# ----for plotting the graph we check the columns by names
# diabetes_X=diabetes.data[:,np.newaxis,2]
diabetes_X=diabetes.data
# print(diabetes_X)
# ----Now we just 30 attributes are values in the project in the train data
diabetes_X_train=diabetes_X[:-30]
# print(diabetes_X_train)
# ----And for testing the data we atrive first 20 values in the project
diabetes_X_test=diabetes_X[-30:]
# print(diabetes_X_test)
# ----Now we atrive the target from dataset diabetes for training the data
diabetes_Y_train=diabetes.target[:-30]
# print(diabetes_Y_train)
# ----Now we atrive the target from dataset diabetes for training the data
diabetes_Y_test=diabetes.target[-30:]
# print(diabetes_Y_test)

# ----creating model using sklearn library
model=linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_Y_train)
diabetes_y_predicted=model.predict(diabetes_X_test)

# ----for checking the mean squared error in the model
print("Mean Squared error is: " ,mean_squared_error(diabetes_Y_test,diabetes_y_predicted))

# ----for checking the weight and intercepts of the dataset
print("Weight: ", model.coef_)
print("Intercept: ", model.intercept_)

# -----plotting the graph from matplotlib library
# plt.scatter(diabetes_X_test,diabetes_Y_test)
# plt.plot(diabetes_X_test,diabetes_y_predicted)
plt.show()

# ----these are the values to find out for some reasons
# Mean Squared error is:  3035.0601152912695
# Weight:  [941.43097333]
# Intercept:  153.39713623331698