# (13)*******************************Mini Batch and Stochastic Gradient Desent in Machine Learning***************************

# ----Root----features----Model(Prediction function)--->>>Compute Loss---->>>Update Parameters-->>>---Again--Model(Predicted function)---
# ------------Labels--------------------------------->>>>^^^^^^^^^^^^^

# -----Batch-----The lenght of the rows in dataset
# -----Gradient Desent-----full lenght of the Batch in datasets
# -----Mini Batch Gradient Decent----As we choose the Batch -----for example----we choose the 100 baches out of 1M batches in dataset
# -----Stochastic Gradient Desent----One Batch of the datasets----for example----one batch out of 1M.  

# ----this is the internet difinitions of batch (batch mode, mini batch and stochastic batch)
# ----Batch Definition-----Batch size is a term used in machine learning and refers to the number of training examples utilized in one iteration. The batch size can be one of three options:
# ----batch mode-----------batch mode: where the batch size is equal to the total dataset thus making the iteration and epoch values equivalent
# ----mini-batch mode------mini-batch mode: where the batch size is greater than one but less than the total dataset size. Usually, a number that can be divided into the total dataset size.
# ----stochastic mode------stochastic mode: where the batch size is equal to one. Therefore the gradient and the neural network parameters are updated after each sample.