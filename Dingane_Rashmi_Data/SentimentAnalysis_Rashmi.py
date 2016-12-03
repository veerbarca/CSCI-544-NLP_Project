import json
import pickle
import os
import time
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import  RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
start_time = time.clock()
data = []
i = 0
reviews = []
labels = []
features = []
with open('/Users/Rashmi/Downloads/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as f:
    for line in f:
        if i %10000==0:
            print (i)
        if i==100000:
            break
        a = json.loads(line)
        reviews.append((a['text'],a['stars']))
        i += 1
corpus = []
labels = []
for i,j in reviews:
    corpus.append(i)
    labels.append(j)
vectorizer = TfidfVectorizer(min_df=1,smooth_idf=True,lowercase=True,max_features=100)
X = vectorizer.fit_transform(corpus).toarray()

train_data = X[:80000]
train_labels = labels[:80000]
test = X[80000:]
test_labels = labels[80000:]

# Extra Trees Classifier
clf1 = ExtraTreesClassifier()
clf1.fit(train_data,train_labels)
y_pred = clf1.predict(test)
accuracy = clf1.score(test,test_labels)
f1_score(test_labels, y_pred, average='weighted')
print("\n Extra Trees Classifier \n")
print(f1_score(test_labels, y_pred, average='weighted'))


# Gradient Boosting
clf1 = GradientBoostingClassifier()
clf1.fit(train_data,train_labels)
y_pred = clf1.predict(test)
accuracy = clf1.score(test,test_labels)
f1_score(test_labels, y_pred, average='weighted')
print("\n Gradient Boosting \n")
print(f1_score(test_labels, y_pred, average='weighted'))


# RandomForestClassifier
clf1 = RandomForestClassifier()
clf1.fit(train_data,train_labels)
y_pred = clf1.predict(test)
accuracy = clf1.score(test,test_labels)
f1_score(test_labels, y_pred, average='weighted')
print("\n Random Forest Classifier \n")
print(f1_score(test_labels, y_pred, average='weighted'))


# SVM
clf1 = SVC()
clf1.fit(train_data,train_labels)
y_pred = clf1.predict(test)
accuracy = clf1.score(test,test_labels)
f1_score(test_labels, y_pred, average='weighted')
print("\n SVM \n")
print(f1_score(test_labels, y_pred, average='weighted'))