# (11)********************How Does Linear Regression Model Work in Machine Learning*****************

# ----first of all we know that the linear line formula is -----y=mx+b
# ----but the SSE(Some of Squared Error) formula is----SSE=(mx+b-y')2
# ----X=[[1],[2],[3]]  And Y=[[3],[2],[4]]
# ----Now we calculate the values as the x puts x values and y puts y values
# ----then we calculated the SSE by differient formula
# ----and then first calculate differient by m and 2nd calculate differient by b
# ----then calculate m and b values as using which find out differient values 
# ----And the final values of m and b is: -----m=1/2, b=2


# ----Now importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# ----Creating dataset
x_train=np.array([[1],[2],[3]])
x_test=np.array([[1],[2],[3]])
y_train=np.array([[3],[2],[4]])
y_test=np.array([[3],[2],[4]])

model=linear_model.LinearRegression()
model.fit(x_train,y_train)
y_test_predicted=model.predict(x_test)
# ----for printing the mean square error we use this method
print("this is the mean square error: " , mean_squared_error(y_test,y_test_predicted))
# ----for check out the weight of the values we use this method
print("Weight: ", model.coef_)
# ----for check out the intercept of the values we use this method
print("Intercept: ",model.intercept_)

# ----Now we plot the graph of the values we use this matplotlib library
plt.scatter(x_test,y_test)
plt.plot(x_test,y_test_predicted )
plt.show()