import pandas as pd
import matplotlib.pyplot as plt
import re                # for text manipulation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np         # for matrix manipulations and numerical calculations


# list of common stop words, used for data preprocessing

common_words = ['a','an','as','of','up','that','the','will','and',"for","escapenumber","from","one","has",
                "had","been","new","all","are","of","this","am", "is", "it", "to", 
                "by", "of", "on", "or", "so"]



# Function for data preprocessing like removal of stop words, lowercasing the email, retaining punctuation marks ! and ?

def process_email(email):
    try:
        email + ""
    except TypeError:
        email = ""
    email = email.lower()
    email_words = re.findall(r'\w+|[!?]', email)
    processed_email = []
    for word in email_words:
        if (word not in common_words and len(word) > 2) or word in {"!", "?"}:
            processed_email.append(word)
    
    final_email = " ".join(processed_email)
    return final_email


# Obtaining the 20 most frequent ngrams from set of emails. using token_patterns, we also count the punctuation marks
def most_frequent(emails, nrange):
    vec = CountVectorizer(ngram_range=nrange, token_pattern=r"(?u)\b\w+\b|[!?]") 
    X = vec.fit_transform(emails)
    m,n = X.shape
    words = X.sum(axis=0).A1
    
    freq = []
    for w, i in vec.vocabulary_.items():
        freq.append((w, words[i]))
    
    top_20 = []
    for j in range(20):
        maxf = max(freq, key=lambda item: item[1])  # getting the most frequent (word,frequency) tuple
        top_20.append(maxf)
        freq.remove(maxf)
    return top_20



# Function to plot the 20 most frequent unigrams and bigrams
def plot_graph(words, title):
    words = pd.DataFrame(words, columns=['word', 'count'])
    plt.figure(figsize=(14, 10))
    plt.barh(words['word'], words['count'])
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('Word', fontsize=14)
    plt.show()



# calculates the accuracy of predictions to check model performance
def calculate_accuracy(clf, X,y):
    ypred = clf.predict(X)
    accuracy = np.round((np.mean(ypred == y)*100),3)
    return accuracy