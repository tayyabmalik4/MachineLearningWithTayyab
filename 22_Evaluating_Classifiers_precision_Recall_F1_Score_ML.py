# (22)*************************Evaluating classifiers:    Precision Recall and F1 Scores in Machine Learning using Python*********************

# **************Why is Accuracy Not a Good Metric to Evaluate Classifiers?
# ---We saw accuracy as a metric to evaluate classifiers
# ---Accuracy is not always the best way to evaluate classifiers(especually in skewed datasets)
# ---Evaluating a classifier is trickier than evaluating a regressor


# *****************What is a Confusion Matrix?
# ---it is a mush better way to evaluate classifiers
# ---How does it work?
# ---Just count the number of times instances of class A are classified(or confused) as class B 

# ****************How to Creat Confusion Matrix?
# --Get the set of predictions, so they can be compared to the actual targets.
# --Each row in a confusion matrix represents an actual class, while each column represents a predicted class.
# --First row is actual negative class & secend row is actual positive class.
# --First column is predicted negative class & secend column is predicted posit ive class.
# --Best classifier is the one gaving only true positives and and true negatives, ie. confusion matrix would have nonzero values only on its main diagonal(top left to bottom right):


# ***************What is precision?
# ---Precision in simple terms means - What precent of positive predictions made were correct?
# ---In Mathematical terms--------Precision= True Positives/ True positive + False psoitive---------------- = TP/TP+FP

# *************** What is Recall?
# ---Recall in simple terms means - What percent of Achual Positive values were correctly classified by your classifier?
# --- In Mathematical terms:------Recall= True positive/true positive+False Negative----------------- = TP/TP+FN 


# ***************F1-Score
# ---It is convenient to combine the performance of a classifier(precision and recall) into a single metric called the F1-Score.
# ---F1-Score is the harmonic mean of precision and recall/
# ---In Mathematical terms:------F1-Score= 2 * Precision * Recall/ Precision + Recall