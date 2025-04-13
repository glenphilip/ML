import pandas as pd                                                 # used for data handling
import os                                                           # used for file handling
from sklearn.feature_extraction.text import CountVectorizer         # used for feature extraction
from naive_bayes import *                                           # using the naivebayes_classifier
from helper_functions import *                                     # contains the helper functions




train = pd.read_csv("train.csv")


# does data Processesing of the train and test dataset
def process_dataset(train, test):                             
    train_processed = []
    test_processed = []
    for email in train['email']:
        processed_email = process_email(email)
        train_processed.append(processed_email)
    for email in test['email']:
        processed_email = process_email(email)
        test_processed.append(processed_email)

    return train_processed, test_processed 


# obtaining the predictions.csv file
def get_predictions(train):                                          
    emails = []
    for f in os.listdir("test"):
        if f.endswith(".txt"):
            file_path = os.path.join("test", f)
            with open(file_path, 'r', encoding='utf-8') as email:
                c = email.read().strip()
            emails.append({'email_name': f, 'email': c})

    test = pd.DataFrame(emails)             # test emails extracted into a dataframe

    train_processed, test_processed = process_dataset(train, test)        
    
    train['processed_email'] = train_processed
    test['processed_email'] = test_processed
    vec = CountVectorizer(binary=True, ngram_range=(1, 2))
    X_train= vec.fit_transform(train['processed_email'])
    y_train = train['label'].values
    X_test = vec.transform(test['processed_email'])  # Extracting unigram and bigram features from processed testing emails

    nb = naivebayes_classifier()
    nb.fit(X_train, y_train)              # training the naive bayes on the train dataset
    y_pred = nb.predict(X_test)           # predicting on testing dataset
    test_predictions = test[['email_name', 'email']]

    test_predictions["Predictions"] = y_pred

    test_predictions.to_csv("test_predictions.csv",index = False)  # Generating a .csv file of predictions
    return y_pred, test_predictions


predictions, test_pred = get_predictions(train)  # y_pred is the array of predictions
print("Predictions array is saved in the variable predictions")
print("Prediction of the test emails\n", test_pred )






