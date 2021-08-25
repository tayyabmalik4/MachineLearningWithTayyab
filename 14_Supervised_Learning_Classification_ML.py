# (14)************************************Supervised Learning Classification in Machine Learning**************************

# -----this is the source which providing all the data----https://monkeylearn.com/blog/classification-algorithms/

# ----Classification----Classification is a natural language processing task that depends on machine learning algorithms.

# ----Classification Algorithm Definition----Classification algorithms in machine learning use input training data to predict the likelihood that subsequent data will fall into one of the predetermined categories. One of the most common uses of classification is filtering emails into spam or non spam.
# ----In short, classification is a form of pattern recognition with classification algorithms applied to the training data to find the same pattern(similar words or sentiments, number sequences,etc) in future sets of data.
# ----Using classification algorithms, which we'll go into more detail about below, text analysis software can perform tasks like aspect-based sentiment analysis to categorize unstructured text by topic and polarity of opinion(positive,negative,neutral, and beyond).
# ----Try out this pre trained sentiment classifier to understand how classification algorithms work in practice, then read on to learn more about different types of classification algorithms.

# ----there are 5 types of the classification algorithms
# --(1)-Logistic Regression
# --(2)-Naive Bayes
# --(3)-K-Nearest Neighbours
# --(4)-Decision Tree
# --(5)-Support Vector Machines


# ****Logistic Regression
# -----Logistic regression is a calculation used to predict a binary outcome: either something happens, or does not. This can be exhibited as Yes/No, Pass/Fail, Alive/Dead, etc. 

# -----Independent variables are analyzed to determine the binary outcome with the results falling into one of two categories. The independent variables can be categorical or numeric, but the dependent variable is always categorical. Written like this:

# -----P(Y=1|X) or P(Y=0|X)

# -----It calculates the probability of dependent variable Y, given independent variable X. 

# -----This can be used to calculate the probability of a word having a positive or negative connotation (0, 1, or on a scale between). Or it can be used to determine the object contained in a photo (tree, flower, grass, etc.), with each object given a probability between 0 and 1. 


# *********Naive Bayes
# ------Naive Bayes calculates the possibility of whether a data point belongs within a certain category or does not. In text analysis, it can be used to categorize words or phrases as belonging to a preset “tag” (classification) or not. For example:

# ------To decide whether or not a phrase should be tagged as “sports,” you need to calculate:
# ----Formula----    P(A|B)=P(B|A) X P(A) / P(B)

# ------Or… the probability of A, if B is true, is equal to the probability of B, if A is true, times the probability of A being true, divided by the probability of B being true.


# ********K-nearest Neighbors
# ----K-nearest neighbors (k-NN) is a pattern recognition algorithm that uses training datasets to find the k closest relatives in future examples. 

# ----When k-NN is used in classification, you calculate to place data within the category of its nearest neighbor. If k = 1, then it would be placed in the class nearest 1. K is classified by a plurality poll of its neighbors.



# ********Decision Tree
# -----A decision tree is a supervised learning algorithm that is perfect for classification problems, as it’s able to order classes on a precise level. It works like a flow chart, separating data points into two similar categories at a time from the “tree trunk” to “branches,” to “leaves,” where the categories become more finitely similar. This creates categories within categories, allowing for organic classification with limited human supervision.
# -----To continue with the sports example, this is how the decision tree works:

# ----there is also divided into 2 types----------
# ---(1)-Random Forest
# ---(2)-Supprot Vector Machines