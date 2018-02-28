import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# print(train.head())

# dropping rows with any 'nan' column
train = train.dropna(axis=0,how='any')


stop_words = set(stopwords.words('english'))
for i in range(len(train)):
	current_sentence = str(train.loc[i,["Statment"]])
	current_sentence = current_sentence.lower()
	current_sentence_token = current_sentence.split()
	word_token = [word for word in current_sentence_token if not word in stop_words]
	stem = PorterStemmer()
	# word_token = [stem.stem(word) for word in word_token]
	cleaned_sentence = ' '.join(word_token)
	train.loc[i,["Statment"]] = cleaned_sentence

print(train.head(1))

