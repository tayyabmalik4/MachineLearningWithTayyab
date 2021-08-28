# (18)*****************************Logistic Regression in Machine Learning***************************************

# ----Logistic Regression Definition----Logistic Regression is a Machine Learning algorithm which is used for the classification problems, it is a predictive analysis algorithm and based on the concept of probability. ... The hypothesis of logistic regression tends it to limit the cost function between 0 and 1 .
# -----Therefore linear functions fail to represent it as it can have a value greater than 1 or less than 0 which is not possible as per the hypothesis of logistic regression.

# ----What is Sigmoid Function in Logistic Regression
# ----In order to map predicted values to probabilities, we use the Sigmoid function. The function maps any real value into another value between 0 and 1. In machine learning, we use sigmoid to map predictions to probabilities.
# ----This is the formula of sigmoid function
# ----S(x)= \frac {1}{1+e^{-x}}
# ----S(x)	=	sigmoid function
# ----e	=	Euler's number

# ----Dicision Boundry---We expect our classifier to give us a set of outputs or classes based on probability when we pass the inputs through a prediction function and returns a probability score between 0 and 1.
# ----Examples----For Example, We have 2 classes, let’s take them like cats and dogs(1 — dog , 0 — cats). We basically decide with a threshold value above which we classify values into Class 1 and of the value goes below the threshold then we classify it in Class 2.

# ----in the images which we use as a example we have chosen the threshold as 0.5, if the prediction function returned a value of 0.7 then we would classify this observation as Class 1(DOG). If our prediction returned a value of 0.2 then we would classify the observation as Class 2(CAT).


# -----Cost function----We learnt about the cost function J(θ) in the Linear regression, the cost function represents optimization objective i.e. we create a cost function and minimize it so that we can develop an accurate model with minimum error.
# ----Formula of Cost Function---
# ----If we try to use the cost function of the linear regression in ‘Logistic Regression’ then it would be of no use as it would end up being a non-convex function with many local minimums, in which it would be very difficult to minimize the cost value and find the global minimum.

# ----Solve to mathematical form
# ---- −log(hθ(x)) if y = 1
# ----- −log(1−hθ(x)) if y = 0
# ----We also write in the form is ----   -log(hθ(x))-log(1-hθ(x))

# ----by the concept of harry we use Loss function and also use this formula----
# ----Loss=-ylogy'  if y=1
# ----Loss=-(1-y)log(1-y')  if y=0
# ----but we use the formula in one line---Loss=-ylogy'-(1-y)log(1-y') ---when we put the values the ans is automatically callout
