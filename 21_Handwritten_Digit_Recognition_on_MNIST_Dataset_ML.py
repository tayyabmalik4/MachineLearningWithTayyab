# (21)*****************************Handwritten Digit Recognition on MNIST Dataset in Machine Learning***********************

# ******What is MNIST?
# ---Set of 70000 small images of digits handwritten by high school students and employees of the US Census Bureau
# ---All images are labelled with the respective digit they represent
# ---MNIST is the hello world of machine learning
# ---There are 70000 images and each image has 784 features
# ---Each image is 28X28 pixels, and each feature simply represents one pic

# **************Agenda
# ----In the last tutorials we saw how to solve an ML problem end to end
# ----In this tutorial we will first fetch the data and then split it into train and test sets
# ----After that we will apply few ML algorithms to detect a given digit

# ----importing library
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ----importing dataset of mnist
mnist= fetch_openml('mnist_784')
print(mnist)
# ----So we also known that the features and labels so we convert the features and labels in to x and y shape 
x,y=mnist['data'],mnist['target']
# print(x)
# print(y)
x.shape
# y.shape

# ----the dataset is dataframe shape we convert the array shape----if we use any sklearn function then it is automatically converted array shape but now the data is all set thats whu we convert array shape
x0=np.array(x)
print(x0.dtype)

# ---check a data
x1=x0[5101]
print(x1.shape)
x3=x1.reshape(28,28)
# ----plotting the graph
plt.imshow(x3,cmap=matplotlib.cm.binary,interpolation='nearest')
plt.axis('off')
plt.show()
# ----checking the value by using target
y[3601]

# ---Now splitting the data into train and test -----we also splitting the data using sklearn library
x_train , x_test = x0[:6000],x0[6000:7000]
y_train, y_test = y[:6000], y[6000:7000]


# -----shuffiling the training dataset because we want to equally distributed data in training and testing
shuffle_index= np.random.permutation(6000)
y_train= y_train[shuffle_index]
x_train= x_train[shuffle_index]


# *********************Creating a 2 Detector
# ----we creat a binary classification
# ----We simply check that the number is achully 2 or not
y_train=y_train.astype(np.int8)
y_test= y_test.astype(np.int8)
y_train_2= (y_train==2)
y_test_2 = (y_test==2)

# y_train_2
y_test_2
# ************************Creating a classifier model
model=LogisticRegression(tol=0.1)
# -----Now fit the model
model.fit(x_train,y_train_2)
# ----Now predicted the value
model.predict([x0[3689]])


# ********************Cross Validation
a=cross_val_score(model,x_train,y_train_2 ,cv=3,scoring='accuracy')
a.mean()



