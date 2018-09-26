# Code made by Matheus Feliciano (mbdf @ cin.ufpe.br)

# Util
import sys
from bs4 import BeautifulSoup as bs
from optparse import OptionParser
from sklearn import metrics
from time import time
import json
import pandas as pd
import numpy as np
# Data loading and vectorizer libraries
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer

# Classifiers libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from sklearn.svm import SVR

names = [
    "XGBClassifier",  # 1
    "SVR",  # 2
    "Linear Regression",  # 3
    "RandomForestRegressor"  # 0
]

classifiers = [
    XGBClassifier(),
    SVR(kernel='linear', C=1e3),
    LinearRegression(),
    RandomForestRegressor()
]

op = OptionParser()

op.add_option("-c", "--classifier", dest="classifier_picked", default="decisiontree",
              help="The type of the classifier, being: \"randomforest\", \"xgb\", \"svr\", \"linearregression\".")

op.add_option("-t", "--text", action="store_true", dest="text_mode",
              help="classify text (unstructured) data")

op.add_option("--csv", action="store_true", dest="csv_mode",
              help="classify csv (structured) data")

op.add_option("--evaluate", action="store_true", dest="evaluation",
              help="Evaluates the classifier")

argv = sys.argv[1:]
(opt, args) = op.parse_args(argv)


def main():
    # loading subsets
    train_set = None
    test_set = None
    if opt.csv_mode:
        train_set = pd.read_csv("csv/csv_final_train.csv").fillna(0)
        test_set = pd.read_csv("csv/csv_final_test.csv").fillna(0)
    #elif opt.text_mode:
        #train_set = 
        #test_set = 

    if opt.evaluation:
        evaluate(train_set, test_set)
    else:
        # print data from test subsets
        print("Data set information:")
        print(3*' ', "%d documents (training set)" % (len(train_set)))
        print(3*' ', "%d documents  (test set)" % (len(test_set)))
        print("\n")
        # initialize vectorizer for BoW
        if opt.text_mode:
            vector = CountVectorizer()
                #stop_words=getStopWords()) check if its necessary
            X_train = vector.fit_transform(train_set.data).todense()
            Y_train = train_set.target
        else:
           X_train = train_set.Preço.values.copy() #preco de acordo com:
           Y_train = train_set.Quartos.values #quartos
        # choosing classifier
        classifier = None
        if opt.classifier_picked == "xgb":
            classifier = classifiers[0]
        elif opt.classifier_picked == "svr":
            classifier = classifiers[1]
        elif opt.classifier_picked == "linearregression":
            classifier = classifiers[2]
        elif opt.classifier_picked == "randomforest":
            classifier = classifiers[3]
        else:
            print(
                "Classifier type not valid, using default: randomforest, for help use: --help")
            classifier = classifiers[3]

        classifier = classifier.fit(X_train[:, np.newaxis], Y_train)

        X_test = test_set.Preço.values.copy()
        Y_test = test_set.Quartos.values
        print(test_set)
        print("Finished training...")
        predictions = []
        #adjust from here on
        predict = classifier.predict(X_test[:, np.newaxis])
        predictions.extend(predict)
        for predict in predictions:	
            print("Relevance: " + str(predict) + ", ")
        print(metrics.classification_report(Y_test, predict))
        #print("Accuracy: ", metrics.accuracy_score(Y_test, predict))
        return predictions
        
#change from here on
def evaluate(train_set, test_set):
    print("Starting evaluation...")
    t0 = time()
    if opt.text_mode:
            vector = CountVectorizer()
            #stop_words=getStopWords()) check if its necessary
            X_train = vector.fit_transform(train_set.data).todense()
            Y_train = train_set.target
    else:
        X_train = train_set.Preço.values.copy() 
        Y_train = train_set.Quartos.values 

    duration = time() - t0
    print("executed in %fs " % (duration))
    #change logic to treat structured and non-structured classifiers for each type of data
    for name, classifier in zip(names, classifiers):
        print(name, " classifier:")

        t0 = time()
        classifier.fit(X_train, Y_train)
        duration = t0 - time()
        print("Dataset trained in %fs " % (duration))

        print(
            4*" ", "Training score: {0:.1f}%".format(classifier.score(X_train, Y_train * 100)))

        X_test = vector.transform(test_set.data).todense()
        Y_test = test_set.target

        predict = classifier.predict(X_test)
        print(metrics.classification_report(
            Y_test, predict, target_names=test_set.target_names))
        print("Accuracy: ", metrics.accuracy_score(Y_test, predict))

def getStopWords():
    stopwords = ""
    with open("stopwords.txt", "r") as sw:
        stopwords = sw.read()
    return stopwords.split()

if __name__ == '__main__':
    main()