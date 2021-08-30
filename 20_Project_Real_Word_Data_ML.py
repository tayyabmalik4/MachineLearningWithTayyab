# (20)*********************Project 01 Real Word Data in Machine Learning*****************************


# ----We Creat a Machine Learning model which is returning the price of the house
# ----We work on Real Data
# ---We Work the Dragon Real Estates
# ---And the Company's Owner Name is Mr.Joseph
# ---We Train the Data With some features and Labels

# ----Problem Teardown
# *We are given dataset of house price with some features like no.bathrooms, no.bedrooms, etc.
# *Our task is to creat a model which will predict the price for any new house by looking at the features.
# *While learning about Machine Learning it is best to achully work with real-world data, not just artificial datasets
# *There are hundreds of open datasets to choose from. We have already talked about the same in this ML course!


# ----Getting Started
# *The First Question Tayyab should ask Mr.Joseph is what is the business objective and end goal?How wil Dragon real estate benifit from the model?
# *Mr.joseph tells Tayyab that Dragon real estates use this model to predict house prices in a givenarea and will invest in the area if its undervalued
# *Next question Tayyab should ask Mr.Joseph is how does the current solution look like? The answer is -Manual experts who is analyze the features
# *The predictions made by so called 'experts' are not very good (error rate is 25%) Which is why Dragon real estates Pvt Ltd. is countiung on Tayyab
 

# ----Finding The Type of Model to Build
# *Supervised,unsupervised or Reinforcement Learning? 
# *Classification task or Regression Task?
# *Batch learning or online learning techniques?

# ----Batch learning techniques----the data is already present 
# ----Online learning techniques---the data is importing online and this is change in every secend

# --------Selecting a performance measure
# *A typical performance measure for regression problem is the Root Mean Squre Error(RMSE)
# *RMSE is generally the preferred performance measure for regression tasks, so we choose it for this particular problem we are solving for Dragon real estates Pvt.Ltd
# *Other performance measures include Mean Absolute Error, Manhattan norm, etc but we will use RMSE for this problem
# -----RMSE formula----RMSE(X,h)=sq root(1/m i=1**sigma**m (h(x**i)-y**i)**2)



# ---------Checking the assumptions
# *It is very important for Tayyab to check for any assumptions he might have made and correct them before launching the ml system
# *For example, he should make sure that the team needs the price and not the categories like expensive, cheap, etc.
# *If latter is the case, formulating the problem as a regression task will be counted as a big mistake.
# *Tayyab talked to the Dragon real estate team members and ensured that he is aware of all the assumptions.

# ------All Set For Coding Now....
# *The lights are green and tayyab is all set for coding now

# ----installing all Pakeges like ----jupyter nootbook, numpy, pandas, matplotlib, scipy, seaborn,sklearn,tensorflow

# -----Starting Now

