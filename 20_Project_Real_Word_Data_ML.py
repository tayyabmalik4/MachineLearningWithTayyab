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

# -----importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

# ---importing the dataset
housing=pd.read_csv('data.csv')
print(housing.head())
# ----for giving the information we use info function
# print(housing.info())

# ----for droping the columns we use drop function
# housing=housing.dropna(axis=1)
# print(housing.info())


# ----Now checking the CHAS values in housing dataframe
# print(housing['CHAS'].value_counts())

# -----Now we chk all the description like min, max, per, mean, count
# print(housing.describe())


# ----for plotting the graph
# %matplotlib inline
# housing.hist(bins=25, figsize=(25,25))
# plt.show() 


# **********************Train Test Splitting
# ----we creat a function which is splitting the train data(80%) and test data(20%) 
# ----this is just a learning purpose----we split the dataset using sklearn by simple one line
def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)* test_ratio)
    test_indices=shuffled[:test_set_size] 
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

#  -----Now splitting the datasets into train and test
train_set, test_set = split_train_test(housing,0.2)
# print(f"Rows in train dataset{len(train_set)}\nRows in test dataset {len(test_set)}")


#  ----Now we work splitting train and test data in sklearn
train_set, test_set =train_test_split(housing,test_size=0.2,random_state=42)
# print(f"Train dataset size is {len(train_set)} \n Test dataset size is {len(test_set)}")


# -----if our one or more fetchers are categorical and it is shows in 0 or 1 form and 1 values are very short and then when we split test and train data may be possible that the 1 value to not gone in the test data or may be not gone train data so we splitting these fechers are equally destributed in train data or test data
# --So we use stratefy sampling function.
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set= housing.loc[test_index]
# print(strat_test_set)
# print(strat_test_set.describe())
# print(strat_test_set.info())
# print(strat_test_set['CHAS'].value_counts())
# print(strat_train_set)
# print(strat_train_set.describe())
# print(strat_train_set.info())
# print(strat_train_set['CHAS'].value_counts())


# -----After splitting the dataset into train and test we creat a copy of train dataset in housing
housing=strat_train_set.copy()

# -----Now we find out the correlation in dataset
# ----Correlation tells us that the price is increase by who many fetchers increase and price are decrese who many fetchers use it
corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

# -----Now Plotting the correlation graphs using pandas_plotting library
# attributes=['MEDV','RM', 'ZN', 'LSTAT']
# scatter_matrix(housing[attributes],figsize=(12,8))


# -----Now Plotting the correlation graphs using pandas_plotting library
# attributes=['MEDV','RM', 'ZN', 'LSTAT']
# scatter_matrix(housing[attributes],figsize=(12,8))


# *********************Attribute Combinations
corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
# ----Now plotting the scatter in TAXRM and MEDV fetchers
# housing.plot(kind='scatter',x='TAXRM',y='MEDV',alpha=0.8

# ----Now we splits the features and lables
housing=strat_train_set.drop("MEDV",axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# *********************Missing Attributes Handles
# ----To take care of missing attributes, you have three options:
# ----1. Get rid of the missing data points
# ----2. Get rid of the whole attribute
# ----3. Set the value to some value(0, mean or median)
# -----option 1
# -----Note that the original housing dataframe will remain unchanged
a=housing.dropna(subset=["RM"])
# print(a.shape)
# -----option 2
# -----Note that the real housing dataframe will remain unchanged until we use inplace=True function
drp=housing.drop("RM",axis=1)
# print(drp.shape)
# ----compute median for  option 3
median=housing["RM"].median()
# print(median)
# ----fill the median values in RM column
# housing["RM"].fillna(median)

# ----before Imputing the values we chk that the describe of the housing dataset
# print(housing.describe())

# -----Now filll the nan values as median using sklearn library
imputer=SimpleImputer(strategy='median')
imputer.fit(housing)

# ----Now check that the median values using statistics_ function
# print(imputer.statistics_)
# print(imputer.statistics_.shape)
# ----Now we transform the dataFrame in one variable
X=imputer.transform(housing)

# ----sklearn is returning the arrays form but we need a data in matrixs form
# ----So we use pandas library to convert the array to datarame
housing_tr=pd.DataFrame(X,columns=housing.columns)

# ----Now chk it the dataframe using describe function
# print(housing_tr.describe())

# **************************Scikit-learn Design
# ----primarily, three types of objects
# 1. Estimaters
# 2. Transformers
# 3. Predictors

# -1. Estimators ---It estimates some parameter based on a dataset. Eg. Imputer. It has a fit method and transform method.
# ----Fit method - Fits the dataset and calculates internal parameters
# -2. Transformers ---- transform method takes input and returns output based on the learning from fit().
# ----It also has a convenience function called fit_transform(). which fit and then transforms.
# -3. Predictors ---LinearRegression model is an example of predictor. fit() and predict are two common functions. It also gives score() function which will evaluate the predictions


# ***************************Features Scalling
# ----------------primarily, two types of features scaling methods:
# --1. Min-max scaling (Normalization)
# --Formula---(value - min)/(max - min)
# --Sklearn provides a class called MinMaxScaler for this

# --2. Standardization
# --formula-- (value - mean)/std
# --Sklearn provides a class called standard Scaler for this


# ****************************Creating a Pipeline
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
#     -----add as many as you want in your pipeline
    ("std_Scaler",StandardScaler()),
])
# ----Now we fit and transform the data in other variable
housing_num_tr=my_pipeline.fit_transform(housing)
# print(housing_num_tr)


# ****************************Appling the Algorithms
# ----Now we impletes the Algorithm of LinearRegression
# ----we use decisionTreeRegressor because LinearRegression is not work accuratilly
# ---if we use DecisionTreeRegressor it converted the model into overfitting which is not a correct
# model=LinearRegression()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# ---Taking some values of houing dataset
some_data=housing.iloc[:5]
# ----taking some labels of housing dataset
some_labels=housing_labels.iloc[:5]
# ----Now we predicted the values using sklearn library and using function of predict()
prepared_data=my_pipeline.transform(some_data)
# ----Now we predict the values
# print(model.predict(prepared_data))
# ----Now checking the lables of the data
# print(list(some_labels))


# *******************************Evaluating the model
# ---Now we check the mean square error
housing_predictions=model.predict(housing_num_tr)
lin_mse=mean_squared_error(housing_labels, housing_predictions)
# ----rmse means that square_root_mean_square_error
lin_rmse=np.sqrt(lin_mse)
# print(lin_mse)
# print(lin_rmse)


# ********************************Using better evaluation techniques -Cross Validation
# ----dividing groups 1 2 3 4 5 6 7 8 9 10
# ----cross validation works that the model is divided into many groups and than then we check the errors
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)
# print(rmse_scores)
# ----Now we just make a function which is printed that the scores, score.mean and score.std
def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ",scores.std())
# print_scores(rmse_scores)


# **********************************Saving the Model
# ----Now we use joblib as to run the Dragon Real Estates 
dump(model, 'Dragon.joblib')


# **********************************Testing the model on test data
X_test = strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set['MEDV'].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions= model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test, final_predictions)
final_rmse=np.sqrt(final_mse)
# print(final_rmse)



# ----Now checking the values 
# print(final_predictions)
# print(list(Y_test))
# print(prepared_data[0])


# *******************************Using the Model
# ----importing libraries
from joblib import dump, load
import numpy as np
model = load('Dragon.joblib')
features=np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.23979304, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
# print(model.predict(features))



