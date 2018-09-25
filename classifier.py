# Code made by Matheus Feliciano (mbdf @ cin.ufpe.br)

# Util
import sys
from bs4 import BeautifulSoup as bs
from optparse import OptionParser
from sklearn import metrics
from time import time
import json
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
    if opt.text_mode:
        train_set = load_files("txts/data/")
        test_set = load_files("txts/test/")
    else:
        train_set = load_files("csv/data/")
        test_set = load_files("csv/test/")

    if opt.evaluation:
        evaluate(train_set, test_set)
    else:
        # print data from test subsets
        print("Data set information:")
        print(3*' ', "%d documents (training set)" % (len(train_set.data)))
        print(3*' ', "%d documents  (test set)" % (len(test_set.data)))
        print("\n")
        # initialize vectorizer for BoW
        if opt.text_mode:
            vector = CountVectorizer()
                #stop_words=getStopWords()) check if its necessary
            X_train = vector.fit_transform(train_set.data).todense()
            Y_train = train_set.target
        else:
            #check how it is for structured

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

        classifier = classifier.fit(X_train, Y_train)
        print("Finished training...")
        predictions = []
        #adjust from here on
        with open(opt.json_mode, mode='r', errors='ignore') as j1:
        	if opt.json_mode:
        		json_data = json.loads(j1.read())
        		for page in json_data:
        			parsedData = parsingHTML(page["texto"])
        			bow = vector.transform([parsedData])
        			predict = classifier.predict(bow.toarray())
        			predictions.extend(predict)
        		
        		zeroes = 0
        		ones = 0
        		for predict in predictions:
        			if predict == 1:
        				ones += 1
        			elif predict == 0:
        				zeroes += 1
        				
        		print("Zeroes "+ str(zeroes))
        		print("Ones "+ str(ones))
        	return predictions
            
        with open(opt.file, mode='r', encoding="utf-8", errors='ignore') as data:
            input = data.read()
            bow = vector.transform([input])
            predict = classifier.predict(bow.toarray())
            predictions.extend(predict)
            for predict in predictions:	
            	print("Relevance: " + str(predict) + ", ")
            return predictions

def evaluate(train_set, test_set):
    print("Starting evaluation...")
    t0 = time()
    if opt.text_mode:
            vector = CountVectorizer()
            #stop_words=getStopWords()) check if its necessary
            X_train = vector.fit_transform(train_set.data).todense()
            Y_train = train_set.target
        else:
            #check how it is for structured

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