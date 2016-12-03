import json
import pickle
import os
import nltk
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
from aylienapiclient.http import Request
from aylienapiclient.errors import HttpError
from aylienapiclient.errors import MissingParameterError
from aylienapiclient.errors import MissingCredentialsError
from textblob import TextBlob

from aylienapiclient import textapi

def get_sentiment(text):
    normalization_constant = 1;
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    normalized_polarity = (polarity + 1.0)/(2.0) * normalization_constant
#     print 'polarity', normalized_polarity
    return normalized_polarity   


def get_similarity(sentence,client,classes):
    
    strnull=""
    try :
        classifications = (client.UnsupervisedClassify({'text': sentence[0:100], 'class': classes}))
        return(classifications['classes'][0]['label'])
    except:
        pass
    return strnull
    
    
    

start_time = time.clock()
data = []
i = 0
reviews = []
labels = []
features = []
with open('/Users/Rashmi/Downloads/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as f:
    for line in f:
        if i %100==0:
            print (i)
        a = json.loads(line)
      
        if(a['business_id'] == 'KayYbHCt-RkbGcPdGOThNg'):
            reviews.append((a['text'],a['stars']))
            i = i+1
            
corpus = []
labels = []
for i,j in reviews:
    corpus.append(i)
    labels.append(j)
    
sent_list=[]

client = textapi.Client( "3be274d5", "0bb74e5b5d6f69a8596b6089ed5ba9da")
classes = ['food', 'service','ambience','deals']


for entry in corpus:
    sent_list=sent_list+nltk.tokenize.sent_tokenize(entry)

food_count=0
service_count=0
ambience_count=0
deals_count=0
food_sentiment=0
service_sentiment=0
ambience_sentiment=0
deals_sentiment=0
answer_list=[]


for i in range(0,len(sent_list)):
    if (i%100)==0:
        print (i)
        
    
    print(sent_list[i])
    data = sent_list[i] #.encode('utf-8',errors='replace')
    temp1 = get_similarity(data,client,classes)
    
    answer_list.append(temp1)
    
    print(temp1)
    sentiment = get_sentiment(sent_list[i])
    if(temp1 == 'food'):
        food_count+=1 
        food_sentiment += sentiment
    elif (temp1 == 'service'):
        service_count+=1
        service_sentiment += sentiment
        
    elif (temp1 == 'ambience'):
        ambience_count+=1
        ambience_sentiment += sentiment
        
    elif (temp1 == 'deals'):
        deals_count+=1
        deals_sentiment += sentiment
    
print (food_count)
print (service_count)
print (ambience_count)
print (deals_count)

print (food_sentiment/food_count)
print (service_sentiment/service_count)
print (ambience_sentiment/ambience_count)
print (deals_sentiment/deals_count)





