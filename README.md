## Spam/Ham classification using NLP

The python script works in the following way:

1. All the modules that are going to be used in the script are imported.
2. The spam/ham database is specified.
3. The functions for the preprocessing of the data from the spam/ham database are constructed.
   This includes functions for removing punctuations and stopwords, and tokenizing and stemming the text.
4. All the above functions are applied to get clean text.
5. The cleaned data is vectorized using tfidf vectorization.
6. Lambda function is applied for determining text message lengths and the histogram for the same is plotted as part of a combined histogram.
7. Function for determining punctuation percentage is created and the histogram for the same is plotted as part of a combined histogram.
7. The test-train split model for spam/ham prediction is used.
8. The precision, recall and accuracy values of the prediction are printed.
9. Combined histograms for text message length and punctuation percentage is shown.
