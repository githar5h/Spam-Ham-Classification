# Spam/Ham classification using NLP
This simple python script sorts the data from a given database into spam or ham using Natural Language processing. The data is preprocessed and vectorized before being fed to the Random forest algorithm for supervised machine learning. The performace of the algorithm is then evaluated using the test-train split procedure and the results of the evaulation are printed, along with the combined histograms for message length and punctuation percentage in the database.

## Prerequisites
1. Python3 along with the following modules:  
   • re  
   • nltk  
   • numpy  
   • pandas  
   • string  
   • matplotlib.pyplot  
2. Appropriate database

## Working
1. All the modules that are going to be used in the script are imported.
2. The spam/ham database is specified.
3. The functions for the preprocessing of the data from the spam/ham database are constructed.
   This includes functions for removing punctuations and stopwords, and tokenizing and stemming the text.
4. All the above functions are applied to get clean text.
5. The cleaned data is vectorized using tfidf vectorization.
6. Lambda function is applied for determining text message lengths and the histogram for the same is plotted    as part of a combined histogram.
7. Function for determining punctuation percentage is created and the histogram for the same is plotted as      part of a combined histogram.
8. The variables for the test-train split model are specified.
9. Required data is fed to the RandomForestClassifier algorithm.
10. The precision, recall and accuracy values of the prediction are printed.
11. Combined histograms for text message length and punctuation percentage is shown.
