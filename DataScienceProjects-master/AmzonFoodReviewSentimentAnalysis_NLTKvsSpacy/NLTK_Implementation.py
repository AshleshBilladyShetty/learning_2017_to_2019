
import time
Start = time.ctime()
import nltk
#from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

#nltk.download()

import sqlite3
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

#imports data file
messages = pd.read_csv("Reviews.csv")

#messages.head()

#create a score binary variable
def partition(x):
    if x < 3:
        return 'negative'
    return 'positive'

Score = messages['Score']
Score = Score.map(partition)

#create a summary column
Summary = messages['Summary']

#Dividing to training and test 
X_train, X_test, y_train, y_test = train_test_split(Summary, Score, test_size=0.2, random_state=42)

X_train.head
X_test.head
y_train.head
y_test.head


# Creating a user defined function that can stm the word: crisping each word
'''
NLTK Functions being used
(1) PorterStemmer()  ->   stemmer.stem(item)
(2) nltk.word_tokenize(text)
'''
#stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords #stopwords actually removes all non important words from sentences

def stem_tokens(tokens, lemmatizer):
    stemmed = []
    for item in tokens:
        stemmed.append(lemmatizer.lemmatize(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    #tokens = [word for word in tokens if word not in stopwords.words('english')]  
    stems = stem_tokens(tokens, lemmatizer)
    return ' '.join(stems) #concatenation 

#creating a function using string library to convert every punctuation to the outtaab 
import string 
intab = string.punctuation
outtab = "                                "
trantab = string.maketrans(intab, outtab)

TimeStart = time.ctime()

#--- Training set
#this for loop takes each row of reviews, lower cases it, replaces all the punctuation and then tokenizes and stems the words
corpus = []
for text in X_train:
    text = str(text).lower()
    text = text.translate(trantab) #this is a function from string lirary that converts punctuation to tabs
    text= tokenize(text.decode('utf-8'))
    corpus.append(text)

# corpus_1 = corpus[:54]
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(corpus_1).toarray()   
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts).toarray()
# 
# pd.DataFrame(X_train_tfidf).to_csv('test.csv')

#sikit learn creates a training data matrix using word bag
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)

'''
tfidf_transformer 
The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document 
is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence 
empirically less informative than features that occur in a small fraction of the training corpus.
'''
tfidf_transformer = TfidfTransformer() #normalises the frequency
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#--- Test set

test_set = []
for text in X_test:
    text = str(text).lower()
    text = text.translate(trantab)
    text=tokenize(text.decode('utf-8'))
    test_set.append(text)

X_new_counts = count_vect.transform(test_set)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

TimeEnd = time.ctime()

#just to see a how the data has been transformed
# from pandas import *
# df = DataFrame({'Before': X_train, 'After': corpus})
# print(df.head(20))

#
prediction = dict()

#Modelling
# from sklearn.naive_bayes import MultinomialNB
# model = MultinomialNB().fit(X_train_tfidf, y_train)
# prediction['Multinomial'] = model.predict(X_test_tfidf)
# 
# from sklearn.naive_bayes import BernoulliNB
# model = BernoulliNB().fit(X_train_tfidf, y_train)
# prediction['Bernoulli'] = model.predict(X_test_tfidf)

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train_tfidf, y_train)
prediction['Logistic'] = logreg.predict(X_test_tfidf)


def formatt(x):
    if x == 'negative':
        return 0
    return 1
vfunc = np.vectorize(formatt)

cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.map(formatt), vfunc(predicted))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1

plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print(metrics.classification_report(y_test, prediction['Logistic'], target_names = ["positive", "negative"]))


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(set(Score)))
    plt.xticks(tick_marks, set(Score), rotation=45)
    plt.yticks(tick_marks, set(Score))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Compute confusion matrix
cm = confusion_matrix(y_test, prediction['Logistic'])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm)    

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()
End = time.ctime()

print Start 
print End
print TimeStart 
print TimeEnd