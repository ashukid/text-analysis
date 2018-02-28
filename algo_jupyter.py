
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
from sklearn.feature_extraction.text import TfidfVectorizer



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.drop(["Relevant?"],axis=1)
train = train.dropna(axis=0,how='any')
y_train = np.ravel(train["Condition"])



train.count()



def data_preprocessing(dataset):
    new_dataset = []
    stop_words = set(stopwords.words('english'))
    for i in range(len(dataset)):
        sent = dataset.loc[i,["Statment"]].values[0].decode('utf-8')
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
train_tfidf=train_tfidf.toarray()
clf=RandomForestClassifier(max_depth=20,random_state=80)
eq=clf.fit(train_tfidf,y_train)



test_tfidf=vectorizer.transform(x_test)
test_tfidf_array=test_tfidf.toarray()
# print(test_tfidf_array)
prediction = eq.predict(test_tfidf_array)
output = pd.DataFrame()
output["Statment"] = test["Statment"]
output["condition"] = prediction
# print(output['condition'].value_counts())
print(output.head())

