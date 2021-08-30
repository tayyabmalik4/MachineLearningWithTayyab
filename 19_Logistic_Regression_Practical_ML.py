# (19)****************************Logistic Regression practical in machine learning*****************
# ----The formula which we use the Logistic Regression----y=1/1+e**-x-----y=1/1+e**-thita**T*X
#  ----The formula which we use the Regression Line----y=Thita**T * X
# -----this is the formula to find out the probability---- p^=sigmoid(Thita**T . X)


# -----What is Logistic Regression
# *A Regression Algorithm which does classification
# *Calculates probability of belonging to a particular class
# *if p>50% -->> 1
# *if p<50% -->> 0

# -----How does Logistic Regression Work?
# *It takes your feartures and labels [Training Data]
# *Fits a linear model (Weights and baises)
# *And instead of giving you the result, it gives you the logistic of the result.
# *Why logistic?

# ******Training a Logistic Regression Model?
# *We need values of parameters in theta
# *We need high values of probabilities near 1 for positive instances
# *We also want low values of probabilities near 0 for negative instances


# ----Cost for single training instance
# cost(thita)={-log(p^) if y=1      and       -log(i-p^) if y=0}
# ----C(thita)=-[y log(p^)+ (1-y)log(i-p^)]
# ----We use cost function to predict the correct values and to avoid the errors
# ----when -log t is very high then t->0 and this is the correct  

# ----Total training instance
# ---J(thita)=-1/m [i=1**Sigma**m y**i log(p^**(i))+ (1-y**i)log(1-p^**i)]


# ----bad news----we didnot calculate minimize in this equation
# ----Good News---We use Gradient Decent function and approch the minima of the Training values


# ----Starting practical
# ----import libraries
from numpy.core.defchararray import mod
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


# ---importing dataset
iris=datasets.load_iris()
# ----Now checking the keys of irus dataset
# print(list(iris.keys()))
# ---Now abstracting the data of irus dataset
# print(iris['data'])
# ---Now abstracting the target of the irus dataset
# print(iris['target'])
# ---Now abstracting the Description of iris dataset
# print(iris['DESCR'])
# ----checking the shape of the iris datasets
# print(iris['data'].shape)

# -------Question----train a logistic regression classifier to predict whether a flower is iris virginica or not

# ---Now training the iris dataset
# ---we train the data using just one colum
X=iris['data'][:,3:]
# print(X)
y=(iris['target']==2).astype(np.int) 
# print(y )

# ----Now we train a Logistic Regression classsifier
model=LogisticRegression()
# ---Now fit the values of Logistic Regression
model.fit(X,y)
# ---Now predicted the values using pridict function
example=model.predict(([[1.6]]))
example=model.predict(([[2.6]]))
# print(example)

# ----Using matplotlib to plot the visualization
X_new=np.linspace(0,3,1000).reshape(-1,1)
# print(X_new)
y_prob=model.predict_proba(X_new)
# ---plotting the graph
plt.plot(X_new,y_prob[:,1],'g--',label='virginica')
plt.show( )
