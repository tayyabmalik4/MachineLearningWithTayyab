# (16)*****************************K-Nearest Neighbors pros and Cons and how to Working in Machine Learning****************

# ----KNN algo is predict the very close value and calculates the values and return the labels
#  -----The k-nearest neighbors(KNN) algorithm is a simple, easy--implement supervised machine learning algorithm that can be used to solve both classification and regression problems. 

# ----A supervised machine learning algorithm is one that relies on labeled input data to learn a function that produces an appropriate output when given new unlabeld data

# ----Imagin a computer is a child, we are its superviser(parent,guardien, or teacher) and we want the child(computer) to learn what a pig looks like. We Will show the child several different pictures, some of which are pigs and the rest could be pictures of anything(cat,dogs,etc).
# ----When we see a pig, we shout 'pig' When its not a pig, we shout 'no pig' After doing this several times with the child, we show them a picture and ask 'pig?' and they will correctly(most of the time) say 'pig' or 'no pig' depending on what the picture is. That is supervised machine learning.


# -----k-nearest-neighbor----The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.

# ----Advantages----
# --(1)-The Algorithm is simple and easy to implement
# --(2)-There's no need to build a model, tune several parameters, or make additional assumptions.
# --(3)-The Algorithm is versatile. It can be used for classification, regression, and search(as we will see in the next section).

# ---Disadvantages----The algorithm gets significantly slower as the number of examples and / or predictors/ indepemdemt variable increse----first KNN take a value then it predict the values step by step so it should be a lot of time when it is calculating the value---- so it is slow


# ----KNN in practice
# -----KNN’s main disadvantage of becoming significantly slower as the volume of data increases makes it an impractical choice in environments where predictions need to be made rapidly. Moreover, there are faster algorithms that can produce more accurate classification and regression results.
# -----However, provided you have sufficient computing resources to speedily handle the data you are using to make predictions, KNN can still be useful in solving problems that have solutions that depend on identifying similar objects. An example of this is using the KNN algorithm in recommender systems, an application of KNN-search.


# -----Recommender Systems
# -----At scale, this would look like recommending products on Amazon, articles on Medium, movies on Netflix, or videos on YouTube. Although, we can be certain they all use more efficient means of making recommendations due to the enormous volume of data they process.
# -----However, we could replicate one of these recommender systems on a smaller scale using what we have learned here in this article. Let us build the core of a movies recommender system.