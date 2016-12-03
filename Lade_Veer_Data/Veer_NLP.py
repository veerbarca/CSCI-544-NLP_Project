import json
import pickle 
import os
import time 
from sklearn.svm import SVC 
from nltk.corpus import stopwords 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.metrics import f1_score 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.tree import DecisionTreeClassifier 
from nltk.corpus import wordnet as wn
import textblob
from textblob import TextBlob
import nltk
from textblob import Word
start_time = time.clock()
data = [] 
i = 0 
reviews = [] 
labels = [] 
features = [] 
with open('yelp_academic_dataset_review.json') as f: 
    for line in f: 
        if i %10000==0: 
            print i 
        if i==100000: 
            break 
        a = json.loads(l)
        # Currently Using a single Business ID
        if a['business_id']=='KayYbHCt-RkbGcPdGOThNg':
            reviews.append((a['text'],a['stars'])) 
            i += 1
    print "done"
print "done"
       



corpus = []
print len(corpus)
labels = [] 
for i,j in reviews: 
    corpus.append(i) 
    labels.append(j)
print len(corpus)
print len(labels)
vectorizer = TfidfVectorizer(min_df=1,smooth_idf=True,use_idf=True,lowercase=True,max_features=2000,stop_words='english') 
X = vectorizer.fit_transform(corpus).toarray()
train_data = X[:80000] 
train_labels = labels[:80000] 
test = X[80000:] 
test_labels = labels[80000:]
classify(train_data,train_labels,test,test_labels)
food_count=0
service_count=0
ambiance_count=0
discount_count=0
food_sentiment=0
service_sentiment=0
ambiance_sentiment=0
discount_sentiment=0
answer_list=[]
sent_list=[]
for entry in corpus:
    sent_list=sent_list+nltk.tokenize.sent_tokenize(entry)
for i in range(0,len(sent_list)):
    if (i%1000)==0:
        print i
    answer_list.append(get_similarity(sent_list[i]))
    if(get_similarity(sent_list[i])=='food'):
        food_count+=1
        food_sentiment+=get_sentiment(sent_list[i])
    elif (get_similarity(sent_list[i])=='service'):
        service_count+=1
        service_sentiment+=get_sentiment(sent_list[i])
    elif (get_similarity(sent_list[i])=='ambiance'):
        ambiance_count+=1
        ambiance_sentiment+=get_sentiment(sent_list[i])
    elif (get_similarity(sent_list[i])=='discount'):
        print sent_list[i]
        discount_count+=1
        discount_sentiment+=get_sentiment(sent_list[i])
print "Number of Counts in Categories"
print food_count
print service_count
print ambiance_count
print discount_count
print "Sentiment per Category"
print float(food_sentiment)/(food_count)
if service_count!=0:
    print float(service_sentiment)/(service_count)
else:
    print 0
if ambiance_count!=0:
    print float(ambiance_sentiment)/(ambiance_count)
else:
    print 0
if discount_count!=0:
    print float(discount_sentiment)/(discount_count)
else:
    print 0

def classify(train_data,train_labels,test,test_labels)
    #Gaussian Naive Bayes
    clf1 = GaussianNB()
    clf1.fit(train_data,train_labels)
    y_pred = clf1.predict(test)
    accuracy = clf1.score(test,test_labels)
    f1_score(test_labels, y_pred, average='weighted')
    print("\n Gaussian Naive Bayes \n")
    print(f1_score(test_labels, y_pred, average='weighted'))
    # Used by other teammates
    # # Decision Tree
    # clf1 = DecisionTreeClassifier()
    # clf1.fit(train_data,train_labels)
    # y_pred = clf1.predict(test)
    # accuracy = clf1.score(test,test_labels)
    # f1_score(test_labels, y_pred, average='weighted')
    # print("\n Decision Tree \n")
    # print(f1_score(test_labels, y_pred, average='weighted'))
    # # # Extra Trees Classifier
    # clf1 = ExtraTreesClassifier()
    # clf1.fit(train_data,train_labels)
    # y_pred = clf1.predict(test)
    # accuracy = clf1.score(test,test_labels)
    # f1_score(test_labels, y_pred, average='weighted')
    # print("\n Extra Trees Classifier \n")
    # print(f1_score(test_labels, y_pred, average='weighted'))
    # # Gradient Boosting
    # clf1 = GradientBoostingClassifier()
    # clf1.fit(train_data,train_labels)
    # y_pred = clf1.predict(test)
    # accuracy = clf1.score(test,test_labels)
    # f1_score(test_labels, y_pred, average='weighted')
    # print("\n Gradient Boosting \n")
    # print(f1_score(test_labels, y_pred, average='weighted'))
    # # # Logistic regression
    # clf1 = LogisticRegression() 
    # clf1.fit(train_data,train_labels) 
    # y_pred = clf1.predict(test) 
    # accuracy = clf1.score(test,test_labels) 
    # f1_score(test_labels, y_pred, average='weighted') 
    # print("\n Logistic Regression \n") 
    # print(f1_score(test_labels, y_pred, average='weighted'))
    # # # RandomForestClassifier
    # clf1 = RandomForestClassifier()
    # clf1.fit(train_data,train_labels)
    # y_pred = clf1.predict(test)
    # accuracy = clf1.score(test,test_labels)
    # f1_score(test_labels, y_pred, average='weighted')
    # print("\n Random Forest Classifier \n")
    # print(f1_score(test_labels, y_pred, average='weighted'))
    # Used by other teammates
    # # LDA
    clf1 = LinearDiscriminantAnalysis() 
    clf1.fit(train_data,train_labels) 
    y_pred = clf1.predict(test) 
    accuracy = clf1.score(test,test_labels) 
    f1_score(test_labels, y_pred, average='weighted') 
    print("\n LDA \n") 
    print(f1_score(test_labels, y_pred, average='weighted'))
    # Used by other teammates
    # # SVM
    # clf1 = SVC()
    # clf1.fit(train_data,train_labels)
    # y_pred = clf1.predict(test)
    # accuracy = clf1.score(test,test_labels)
    # f1_score(test_labels, y_pred, average='weighted')
    # print("\n SVM \n")
    # print(f1_score(test_labels, y_pred, average='weighted'))
    # Used by other teammates

def get_similarity(sentence):
    sentence_words = sentence.split(" ")
    categories = ["food", "service","ambiance", "discount"]
    categories_score = [0, 0, 0, 0]

    for i in range (0, 4):
        category_synset = wn.synset(categories[i] + '.n.01')
        for word in sentence_words:
            try:
                syn = Word(word)
                word_synset = syn.synsets[0]
                #Using Path Similarity
                categories_score[i] += category_synset.path_similarity(word_synset)
                #Using Path Similarity
            except Exception, e:
                pass
    flag=False
    if categories_score[1] >= 0.9*(categories_score[0]):
        flag=True
    elif categories_score[2] >= 0.9*(categories_score[0]):
        flag=True 
    elif categories_score[3] >= 0.9*(categories_score[0]):
        flag=True
    if (flag):
        if (categories_score[3] > categories_score[2]):
            if (categories_score[3] > categories_score[1]):
                return 'discount'
        if (categories_score[2] > categories_score[1]):
            if (categories_score[2] > categories_score[3]):
                return 'ambiance'
        if (categories_score[1] > categories_score[2]):
            if (categories_score[1] > categories_score[3]):
                return 'service'
    return categories[categories_score.index(max(categories_score))]

def get_sentiment(text):
    normalization_constant = 1;
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    normalized_polarity = (polarity + 1.0)/(2.0) * normalization_constant
    return normalized_polarity

