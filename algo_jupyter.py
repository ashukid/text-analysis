
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import re


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.drop(["Relevant?"],axis=1)
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
x_test = data_preprocessing(test)


vectorizer=TfidfVectorizer(stop_words='english')
train_tfidf=vectorizer.fit_transform(x_train)
train_tfidf_array=train_tfidf.toarray()
test_tfidf=vectorizer.transform(x_test)
test_tfidf_array=test_tfidf.toarray()


# Random Forest Classifier
clf_rf=RandomForestClassifier(max_depth=20,random_state=80)
clf_rf.fit(train_tfidf_array,y_train)
prediction_rf = clf_rf.predict(test_tfidf_array)
print("Random Forest Accuracy : {}".format(clf_rf.score(train_tfidf_array,y_train)))



# naive bayes classifier
clf_nb=MultinomialNB()
clf_nb.fit(train_tfidf_array,y_train)
prediction_nb = clf_nb.predict(test_tfidf_array)
print("Multinomail Naive Bayes Accuracy : {}".format(clf_nb.score(train_tfidf_array,y_train)))


# Decision tree classifier
clf_dt=DecisionTreeClassifier()
clf_dt.fit(train_tfidf_array,y_train)
prediction_dt = clf_dt.predict(test_tfidf_array)
print("Decision Tree Accuracy : {}".format(clf_dt.score(train_tfidf_array,y_train)))


# Voting classifier
# this classifier just uses 2 or more classifier to predict
# and then takes the majority vote
clf = VotingClassifier(estimators=[('rf',clf_rf),('dt',clf_dt),('nb',clf_nb)],voting='soft')
clf.fit(train_tfidf_array,y_train)
prediction = clf.predict(test_tfidf_array)
print("Voting Classifier Accuray : {}".format(clf.score(train_tfidf_array,y_train)))


# print(test_tfidf_array)
output = pd.DataFrame()
output["Statment"] = test["Statment"]

# Decision tree as final classifer
# as it gave maximum accuracy
output["condition"] = prediction_dt
output['condition'].value_counts()

# final prediction is saved in output.csv
output.to_csv('output.csv')
