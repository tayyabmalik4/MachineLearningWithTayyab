# (09)***********************Multivariable Linear Regression in Machine Learning**************************

# -----Definition of Linear Regression---Linear Regression is a supervised machine learning algorithm where the predicted output is continuous and has a constant slope. It's used to predict values within a continuous range, (e.g. sales, price) rather than trying to classify them into categories (e.g. cat, dog). There are two main types:
# ----(1)-Simple Linear Regression
# ----(2)-Multivariable regression

# ----Simple Linear Regression Definition----Simple linear regression uses traditional slope-intercept form, where m and b are the variables our algorithm will try to “learn” to produce the most accurate predictions. x represents our input data and y represents our prediction.
# ----Formula----y=mx+b----f(x1)=w1x1+w0

# ----Multivariable Regression Definition-----A mpre complex, multi-variable linear equation might look like this, where W represents the coeficients, or weight, our model will try to learn.
# ----Fromula---f(x,y,z)=w1x+w2y+w3z
# ----Fromula by harry---f(x,y,z)=w0+w1x+w2y+w3z
# ----Fromula by harry---f(x1,x2,x3,.....,xn)=w0+w1x1+w2x2+w3x3+....+wnxn.
# The variables x,y,z represent the attributes, or distinct pieces of information, we have about each observation. For sales predictions, these attributes might include a company's advertising spend on radio,TV, and newspapers
# Sales=w1Radio +w2TV+w3News