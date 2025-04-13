import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from naive_bayes import *            # contains the naive bayes classifier
from helper_functions import *      # contains the helper functions




# loading the training dataset
train = pd.read_csv("train.csv")          
# loading testing data for analysis of data preprocessing & feature extraction on model performance
test = pd.read_csv("analysis_test.csv")   



# Doing data preprocessing on train and test data
train_processed = []
test_processed = []
for email in train['email']:
    processed_email = process_email(email)
    train_processed.append(processed_email)


for email in test['email']:
    processed_email = process_email(email)
    test_processed.append(processed_email)


train['processed_email'] = train_processed
test['processed_email'] = test_processed



# Extracting unigram features from raw test and train datasets
vec1 = CountVectorizer(binary=True)
X_train1 = vec1.fit_transform(train['email'])
y_train1 = train['label'].values
X_test1 = vec1.transform(test['email'])
y_test1 = test['label'].values


# Extracting unigram features from processed test and train datasets
vec2 = CountVectorizer(binary=True)
X_train2= vec2.fit_transform(train['processed_email'])
y_train2 = train['label'].values
X_test2 = vec2.transform(test['processed_email'])
y_test2 = test['label'].values


# Extracting unigram and bigram features from processed test and train datasets
vec3 = CountVectorizer(binary=True, ngram_range=(1, 2))
X_train3= vec3.fit_transform(train['processed_email'])
y_train3 = train['label'].values
X_test3 = vec3.transform(test['processed_email'])
y_test3 = test['label'].values


#Splitting into raw training dataset into spam and ham datasets
ham1 = train.loc[train['label'] == 0, 'email']
spam1 = train.loc[train['label'] == 1, 'email']


# Splitting into processed training dataset into spam and ham datasets
ham2 = train.loc[train['label'] == 0,'processed_email']
spam2 = train.loc[train['label'] == 1,'processed_email']


# Identifying the 20 most frequent unigrams in raw spam and ham emails
ham_freq1 = most_frequent(ham1, (1, 1))
spam_freq1 = most_frequent(spam1, (1, 1))
plot_graph(ham_freq1, "20 Most Frequent unigrams in raw Ham emails")
plot_graph(spam_freq1, "20 Most Frequent unigrams in raw Spam emails")


# Identifying the 20 most frequent unigrams in processed spam and ham emails
ham_freq2 = most_frequent(ham2, (1,1))
spam_freq2 = most_frequent(spam2, (1,1))
plot_graph(ham_freq2, "20 Most Frequent unigrams in processed Ham emails")
plot_graph(spam_freq2, "20 Most Frequent unigrams in processed Spam emails")


# Identifying the 20 most frequent bigrams in processed spam and ham emails
ham_freq4 = most_frequent(ham2, (2, 2))
spam_freq4 = most_frequent(spam2, (2,2))
plot_graph(ham_freq4, "20 Most Frequent bigrams in  processed Ham emails")
plot_graph(spam_freq4, "20 Most Frequent bigrams in processed spam emails")




nb1 = naivebayes_classifier()
nb1.fit(X_train1, y_train1)
ypred1 = nb1.predict(X_test1)
acc1= calculate_accuracy(nb1,X_test1,y_test1)
print("Accuracy over raw test emails(unigrams), X_test1:", acc1)


nb2 = naivebayes_classifier()
nb2.fit(X_train2, y_train2)
acc2= calculate_accuracy(nb2,X_test2,y_test2)
print("Accuracy over processed test emails (unigrams), X_test2:", acc2)


nb3 = naivebayes_classifier()
nb3.fit(X_train3, y_train3)
acc3= calculate_accuracy(nb3,X_test3,y_test3)
print("Accuracy over processed test emails (unigrams, bigrams), X_test3:", acc3)


df = ['Raw emails (using unigrams), X_test1','Processed emails (using unigrams), X_test2','Processed emails (using uni & bigrams), X_test3']
plt.figure(figsize=(8, 5))
bars = plt.bar(df, [acc1, acc2, acc3], color=['lightblue', 'red', 'green'], width=0.2) 
plt.title('Impact of Data Processing & Feature Extraction on Accuracy')  
plt.xlabel('Different Types of Datasets')
plt.ylabel('Accuracy')
plt.xticks([])
plt.yticks(range(75, 90, 2))
plt.ylim(75, 90)
plt.grid(False)
plt.legend(bars, df)
plt.show()






