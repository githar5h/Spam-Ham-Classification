# Spam/ham classifier using NLP by Harsh Khati

# importing all the necessary modules
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# reading the spam/ham database
dataset=pd.read_csv("smsspamcollection\SMSSPamCollection.tsv", sep="\t",header=None)
dataset.columns=['label','body_text']

# Constructing functions for preprocessing of the data

# punctuation removal
def punctuation_removal(text):
    text_no_punctuation="".join([char for char in text if char not in string.punctuation])
    return text_no_punctuation

dataset['no_punctuation_text']=dataset['body_text'].apply(lambda x:punctuation_removal(x))

# tokenization
def tokenization(text):
    tokens=re.split('\W',text)
    return tokens

dataset['tokenized_text']=dataset['no_punctuation_text'].apply(lambda x:tokenization(x.lower()))

# stopwords removal
stopwords= nltk.corpus.stopwords.words('english')

def stopwords_removal(tokenized_list):
    text=[word for word in tokenized_list if word not in stopwords]
    return text

dataset['nostopwords_text']=dataset['tokenized_text'].apply(lambda x:stopwords_removal(x))

# Stemming
ps=nltk.PorterStemmer()

def stemming(tokenized_text):
    text=[ps.stem(word) for word in tokenized_text]
    return text

dataset['stemmed_text']=dataset['nostopwords_text'].apply(lambda x:stemming(x))

# applying all the preprocessing functions and getting clean text
def clean_text(text):
    text="".join([word.lower() for word in text if word not in string.punctuation])
    tokens=re.split('\W',text)
    text=[ps.stem(word) for word in tokens if word not in stopwords]
    return text

# tfidf Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vector= TfidfVectorizer(analyzer=clean_text)
X_tfidf= tfidf_vector.fit_transform(dataset['body_text'])

# lambda function for text message length
dataset["body_len"]=dataset["body_text"].apply(lambda x:len(x)-x.count(" "))

# plotting graph of text length
bins=np.linspace(0,200,40)
plt.hist(dataset['body_len'],bins, label = 'Body Length Distribution')

# constructing function for punctuation percentage in the text
def count_punct(text):
    count=sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text)-text.count(" ")),3)*100

dataset["punct%"]=dataset['body_text'].apply(lambda x:count_punct(x))

# plotting graph of punctuation percentage
bins=np.linspace(0,50,40)
plt.hist(dataset['punct%'],bins, label = 'Punctuation Percentage Distribution')

plt.legend()

X_features= pd.concat([dataset['body_len'],dataset['punct%'],pd.DataFrame(X_tfidf.toarray())],axis=1)
X_features.head()

# test-train split model
X_train, X_test, y_train, y_test= train_test_split(X_features, dataset['label'],test_size=0.3,random_state=0)

#random forest classification algorithm is used
rf= RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)
rf_model=rf.fit(X_train, y_train)

sorted(zip(rf_model.feature_importances_, X_train.columns),reverse=True)[0:10]

y_pred= rf_model.predict(X_test)

# printing the precision, recall and accuracy values of the prediction
precision, recall, fscore, support= score(y_test, y_pred, pos_label='spam', average='binary')

print('Precision {} /Recall {} / Accuracy {}'.format(round(precision,3),
                                                    round(recall,3),
                                                    round((y_pred==y_test).sum()/len(y_pred),3)))

# displaying combined histogram of body text length and percentage
plt.show()