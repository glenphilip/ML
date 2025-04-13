import numpy as np

class naivebayes_classifier:
    def __init__(self): 
        self.prob_spam = None                   
        self.prob_spam_words = None
        self.prob_ham_words = None

    
    def fit(self, X_train, y_train): # training the model to learn class prior and probabilites of the words given a class
        m, n = X_train.shape
        Xtrain_spam, Xtrain_ham = [], []
        for i in range(m):
            if y_train[i] == 1:
                Xtrain_spam.append(X_train[i])
            else:
                Xtrain_ham.append(X_train[i])
        
        spam_words, ham_words = self.get_wordcount(X_train,Xtrain_spam, Xtrain_ham)
        spam_num = np.sum(y_train)
        spam_words, ham_words, spam_num, ham_num, total = self.laplace_smoothing(spam_words, ham_words, spam_num, m)
        
        self.prob_spam_words = spam_words / spam_num
        self.prob_ham_words = ham_words / ham_num
        self.prob_spam = spam_num / total

    def get_wordcount(self,X,X_spam, X_ham):       # getting the frequency of each word in all spam emails and all ham emails
        m,n = X.shape
        spam_words = np.zeros(n)
        ham_words = np.zeros(n)
        for email in X_spam:
            spam_words += email.toarray().flatten() # converting the sparse matrix into a numpy 1d array
        for email in X_ham:
            ham_words += email.toarray().flatten()   # converting the sparse matrix into a numpy 1d array
        return spam_words, ham_words

    def laplace_smoothing(self, spam_words, ham_words, spam_num, total):      # applying smoothing to deal with zero probabilities
        spam_words += 1
        ham_words += 1
        total = total + 4           # added 4 pseudo emails, one containing all words to spam and ham, one containing no words to spam & ham                
        spam_num = spam_num + 2           
        ham_num = total - spam_num
        return spam_words, ham_words, spam_num, ham_num, total

    def log_probability(self, email, class_word_probs, class_prob):       # calculates the log probabilities of a test given class
        email = email.toarray().flatten()   # converting the sparse matrix into a numpy 1d array
        log_prob_present = np.sum(email * np.log(class_word_probs))
        log_prob_absent = np.sum((1 - email) * np.log(1 - class_word_probs))
        log_prob = log_prob_present + log_prob_absent + np.log(class_prob)
        return log_prob

    def predict(self, X):           # predicting on the testing dataset
        m, n = X.shape
        prob_spam = self.prob_spam
        prob_ham = 1 - prob_spam
        log_prob_spam = np.zeros(m)
        log_prob_ham = np.zeros(m)
        
        for i in range(m):
            log_prob_spam[i] = self.log_probability(X[i], self.prob_spam_words, prob_spam)
            log_prob_ham[i] = self.log_probability(X[i], self.prob_ham_words, prob_ham)
    
        predictions = np.zeros(m)
        for i in range(m):
            if log_prob_spam[i] >= log_prob_ham[i]:         # comparing log probability for prediction
                predictions[i] = 1
            else:
                predictions[i] = 0
        return predictions
