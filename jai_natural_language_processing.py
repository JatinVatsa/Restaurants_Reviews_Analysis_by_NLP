# Natural Language Processing

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset (tsv means tab separate values  , csv means comma separate values)
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) # quoting is used to ignore double quates which might crate a problem

# Cleaning the texts (so that we can use relevant words which can be used in bag of word model, & converting upper case into lower case )
# steming means replacement of words into other but of same meaning eg. loved is change to love , so that we reduce size of words
import re
import nltk # used to remove unrelevant  words which are not used to prdict in review
#nltk.download('stopwords')  # stopwords is a list which contains words which  are unrelevant 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

corpus = []
for a in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][a]) 
    review = review.lower() 
    review = review.split()  # it split the review which is a str into different words (list)
    ps=PorterStemmer()
    review = [ps.stem(i) for i in review if not i in set(stopwords.words('english'))]
    
    # joining back into the previous section i.e back in to string
    review = ' '.join(review)  # joining with space
    corpus.append(review)
    
# Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features= 1500)  # if words are more in corpus,then we can use max_features parameters in CountVctorizer to avoid more unrelevant word
X = cv.fit_transform(corpus).toarray() 

Y = dataset.iloc[:,1].values

# classification template ( Navies Bayes Classification)
#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

# we can neg feature scaling because in this we have 0,1 

#fitting navie classifier to the training set
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB() # no prameters
classifier.fit(X_train,Y_train)


#predicting test  set result
y_pred=classifier.predict(X_test) 

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred) 
accuracy= (55+91)/200

