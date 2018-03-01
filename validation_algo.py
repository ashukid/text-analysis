import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

import re

train = pd.read_csv('train.csv')
# dropped the train relevant column
# as relevant and condition contains same infromation
train = train.drop(["Relevant?"],axis=1)
# dropped row with nan values
train = train.dropna(axis=0,how='any')
y_train = np.ravel(train["Condition"])


def data_preprocessing(dataset):
    new_dataset = []
    stop_words = set(stopwords.words('english'))
    for i in range(len(dataset)):
        sent = dataset.loc[i,["Statment"]].values[0].decode('utf-8')
        sent = re.sub('[^a-zA-Z]', ' ', sent)
        sent_token = sent.lower().split()
        word_token = [word for word in sent_token if not word in stop_words]
        stem = SnowballStemmer("english")
        # stem = PorterStemmer()
        word_token = [stem.stem(word) for word in word_token]
        cleaned_sent = " ".join(word_token)
        new_dataset.append(cleaned_sent)
    return new_dataset    


x_train = data_preprocessing(train)
# splitting the dataset into training and testing
# (x,y) -> training
# (xx,yy) -> testing
x,xx,y,yy = train_test_split(x_train,y_train,test_size=0.2,random_state=100)


vectorizer=TfidfVectorizer(stop_words='english')
train_tfidf=vectorizer.fit_transform(x)
train_tfidf_array=train_tfidf.toarray()
test_tfidf=vectorizer.transform(xx)
test_tfidf_array=test_tfidf.toarray()


# Decision tree classifier
clf_dt=DecisionTreeClassifier()
clf_dt.fit(train_tfidf_array,y)
prediction_dt = clf_dt.predict(test_tfidf_array)


accuracy = clf_dt.score(test_tfidf_array,yy)
loss = log_loss(yy, prediction_dt)


print("Accuracy : {}".format(accuracy*100))
print("Loss : {}".format(loss))

