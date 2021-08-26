# (15)*********************************** K-nearest Neighbor Classification in Machine Learning********************

# ----K-nearest neighbors (k-NN) is a pattern recognition algorithm that uses training datasets to find the k closest relatives in future examples. 

# ----When k-NN is used in classification, you calculate to place data within the category of its nearest neighbor. If k = 1, then it would be placed in the class nearest 1. K is classified by a plurality poll of its neighbors.



# ----importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tlf
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# ----loading the data using sklearn
iris=datasets.load_iris()

# ----printing description and features and also labels
# print(iris.DESCR)
# ----there are 4 features-----sepal lenght,sepal width, petal lenght, petal width
features=iris.data
# ----there are 3 labels ------0 stands for setosa, 1 stands for versicolor, 2 stands for Virginica
labels=iris.target 
# print(features[0],labels[0])

# ----Now starting the training the Classifier
# ----Now we define kneighborsClassifier in sklearn
model=KNeighborsClassifier()
# ----Now fit these values in the sklearn using fit function
model.fit(features,labels)

# ----Now we just pridected the values as chk that the model train is well define or not 
# ----we just input the features and the model is return the very close label as we define it
predict=model.predict([[31,1,1,1]])
# print(predict)

# ----chk that the value which model is return is true or false
predict1=model.predict([[5.1,3.5,1.4,0.2]])
print(predict1)


