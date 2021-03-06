import logging
import os, sys
import inspect
import pandas as pd
import glob as g
import random
from sklearn.externals import joblib
import mlflow
from mlflow.sklearn import log_model

import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_log_error(y_true, y_pred):
    return np.mean(np.abs((np.log(y_true)-np.log(y_pred))))

def getStopWords():
    stopwords = ""
    with open("stopwords.txt", "r") as sw:
        stopwords = sw.read()
    return stopwords.split()

# Create a dataframe with the four feature variables
train_csv = pd.read_csv(sys.argv[1])
test_csv = pd.read_csv(sys.argv[2])

city = sys.argv[1].split("_")[0] 
print(city)

train_urls = g.glob(city.lower() + "_sample/txts/train/*.txt")
test_urls = g.glob(city.lower() + "_sample/txts/test/*.txt")

train_txt = []
test_txt = []

for txt in train_urls:
    with open(txt, "r", encoding="utf-8", errors="ignore") as t1:
        train_txt.append(t1.read())

for txt in test_urls:
    with open(txt, "r", encoding="utf-8", errors="ignore") as t1:
        test_txt.append(t1.read())

train_csv = train_csv.drop(['url'],axis=1)
test_csv = test_csv.drop(['url'],axis=1)

train_csv['type'] = train_csv['type'].map({'apart': 1, 'house': 0})
test_csv['type'] = test_csv['type'].map({'apart': 1, 'house': 0})

#remove the rent rows, leaving only 'sell' operation
train_csv['operation'] = train_csv['operation'].map({'sell': 1, 'rent': 0})
test_csv['operation'] = test_csv['operation'].map({'sell': 1, 'rent': 0})

train_csv = train_csv[train_csv.operation != 0]
test_csv = test_csv[test_csv.operation != 0]

#print(len(train_txt))
#print(len(test_txt))
#print(train_csv.count)
#print(test_csv.count)

#create the text column
#train_csv['txt'] = train_txt
#test_csv['txt'] = test_txt

#vectorizing text
vector = TfidfVectorizer(stop_words=getStopWords(), max_features=1000)
X_txt_train = vector.fit_transform(train_txt).todense()
X_txt_test = vector.transform(test_txt).todense()

#create df with vectorizer
train_txt_df = pd.DataFrame(X_txt_train, columns=vector.get_feature_names())
test_txt_df = pd.DataFrame(X_txt_test, columns=vector.get_feature_names())

#Joining the 2 DFs
train_csv = pd.concat([train_csv, train_txt_df])
test_csv = pd.concat([test_csv, test_txt_df])

#colocando valores 0 em palavras que não estão nos documentos. Evitando NaN
train_csv.fillna(0, inplace=True)
test_csv.fillna(0, inplace=True)

print(train_csv.count)

print("train vector size: " + str(X_txt_train.shape))
print("test vector size: " + str(X_txt_test.shape))

# Create a list of the feature column's names
features = train_csv.columns[1:]

# train_csv['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y = train_csv['price']
#print(y.count)
y_test_csv = test_csv['price']

train_sample = train_csv.sample(n=len(train_csv), random_state=2)
print(train_sample)

#train_sample = X_txt_train

# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train_sample))
print('Number of observations in the test data:',len(test_csv))
#print('TRAIN SAMPLES VALUES:', train_sample)

svm = SVR()
xgboost = xgb.XGBRegressor()

def rf_from_cfg(cfg, seed):
    """
        Creates a random forest regressor from sklearn and fits the given data on it.
        This is the function-call we try to optimize. Chosen values are stored in
        the configuration (cfg).

        Parameters:
        -----------
        cfg: Configuration
            configuration chosen by smac
        seed: int or RandomState
            used to initialize the rf's random generator

        Returns:
        -----------
        np.mean(rmses): float
            mean of root mean square errors of random-forest test predictions
            per cv-fold
    """
    rfr = RandomForestRegressor(
        n_estimators=cfg["num_trees"],
        criterion=cfg["criterion"],
        min_samples_split=cfg["min_samples_to_split"],
        min_samples_leaf=cfg["min_samples_in_leaf"],
        min_weight_fraction_leaf=cfg["min_weight_frac_leaf"],
        max_features=cfg["max_features"],
        max_leaf_nodes=cfg["max_leaf_nodes"],
        bootstrap=cfg["do_bootstrapping"],
        random_state=seed)

    def rmse(y, y_pred):
        return np.sqrt(np.mean((y_pred - y)**2))

      
    # Creating root mean square error for sklearns crossvalidation
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    #score = cross_val_score(rfr, train_sample.iloc[:,1:7].as_matrix(), train_sample.iloc[:,0].as_matrix(), cv=11, scoring=rmse_scorer)
    score = cross_val_score(rfr, train_sample.iloc[:, train_sample.columns != 'price'].as_matrix(), train_sample.iloc[:,0].as_matrix(), cv=11, scoring=rmse_scorer)
    #score = cross_val_score(rfr, train_sample, train_csv['price'], cv=11, scoring=rmse_scorer)
    return -1 * np.mean(score)  # Because cross_validation sign-flips the score

print("Running hyperparameter optimization...")
logger = logging.getLogger("RF-example")
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)  # Enable to show debug-output
logger.info("Running random forest example for SMAC. If you experience "
            "difficulties, try to decrease the memory-limit.")

# Build Configuration Space which defines all parameters and their ranges.
# To illustrate different parameter types,
# we use continuous, integer and categorical parameters.
cs = ConfigurationSpace()

# We can add single hyperparameters:
do_bootstrapping = CategoricalHyperparameter(
    "do_bootstrapping", ["true", "false"], default_value="true")
cs.add_hyperparameter(do_bootstrapping)

# Or we can add multiple hyperparameters at once:
max_depth = UniformIntegerHyperparameter("max_depth", 1, 10, default_value=5)
num_trees = UniformIntegerHyperparameter("num_trees", 10, 50, default_value=50)
max_features = UniformIntegerHyperparameter("max_features", 1, features.shape[0], default_value=1)
min_weight_frac_leaf = UniformFloatHyperparameter("min_weight_frac_leaf", 0.0, 0.5, default_value=0.0)
criterion = CategoricalHyperparameter("criterion", ["mse", "mae"], default_value="mse")
min_samples_to_split = UniformIntegerHyperparameter("min_samples_to_split", 2, 20, default_value=2)
min_samples_in_leaf = UniformIntegerHyperparameter("min_samples_in_leaf", 1, 20, default_value=1)
max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 10, 1000, default_value=100)

cs.add_hyperparameters([max_depth, num_trees, max_features, min_weight_frac_leaf, criterion, min_samples_to_split, min_samples_in_leaf, max_leaf_nodes])

# SMAC scenario oject
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternative runtime)
                     "runcount-limit": 50,  # maximum number of function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true",
                     "memory_limit": 3072,   # adapt this to reasonable value for your hardware
                     "shared_model": True,
                     "input_psmac_dirs": "smac3-output*"
                     })

# To optimize, we pass the function to the SMAC-object
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
            tae_runner=rf_from_cfg)

# Example call of the function with default values
# It returns: Status, Cost, Runtime, Additional Infos
def_value = smac.get_tae_runner().run(cs.get_default_configuration(), 1)[1]
print("Value for default configuration: %.2f" % (def_value))

# Start optimization
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

incumbent._values

mlflow.log_param("city",city)
mlflow.log_param("model","rf")
all_feats = ""
for elem in features:
    if all_feats == "":
        all_feats = elem
    else:
        all_feats = all_feats + "," + elem

mlflow.log_param("features", all_feats)
mlflow.log_param("max_depth", incumbent._values['max_depth'])
mlflow.log_param("num_trees", incumbent._values['num_trees'])
mlflow.log_param("num_trees", incumbent._values['num_trees'])
mlflow.log_param("min_samples_to_split", incumbent._values['min_samples_to_split'])
mlflow.log_param("min_samples_in_leaf", incumbent._values['min_samples_in_leaf'])
mlflow.log_param("max_leaf_nodes", incumbent._values['max_leaf_nodes'])

print("Evaluating the model...")

regr = RandomForestRegressor(max_depth=incumbent._values['max_depth'],
        n_estimators=incumbent._values['num_trees'], 
        random_state=0, 
        bootstrap=incumbent._values['num_trees'], min_samples_split=incumbent._values['min_samples_to_split'], min_samples_leaf=incumbent._values['min_samples_in_leaf'], max_leaf_nodes=incumbent._values['max_leaf_nodes'])

svm.fit(train_csv[features], y)
xgboost.fit(train_csv[features], y)
regr.fit(train_csv[features], y)
""" regr.fit(X_txt_train, y)
svm.fit(X_txt_train, y)
xgboost.fit(X_txt_train, y)
 """
# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
regr.predict(test_csv[features])
svm.predict(test_csv[features])
xgboost.predict(test_csv[features])
""" regr.predict(X_txt_test)
svm.predict(X_txt_test)
xgboost.predict(X_txt_test)
 """
#metrics for the csv RF
r_mle_rf = mean_log_error(regr.predict(test_csv[features]), y_test_csv)
r_mape_rf = mean_absolute_percentage_error(regr.predict(test_csv[features]), y_test_csv)
r_mae_rf  = np.median(abs(regr.predict(test_csv[features])-y_test_csv)/y_test_csv)
r_mse_rf = mean_squared_error(regr.predict(test_csv[features]), y_test_csv)
r_msle_rf = mean_squared_log_error(regr.predict(test_csv[features]), y_test_csv)
r_r2_rf = r2_score(regr.predict(test_csv[features]), y_test_csv)

#metrics for the csv SVM
r_mle_svm = mean_log_error(svm.predict(test_csv[features]), y_test_csv)
r_mape_svm = mean_absolute_percentage_error(svm.predict(test_csv[features]), y_test_csv)
r_mae_svm  = np.median(abs(svm.predict(test_csv[features])-y_test_csv)/y_test_csv)
r_mse_svm = mean_squared_error(svm.predict(test_csv[features]), y_test_csv)
r_msle_svm = mean_squared_log_error(svm.predict(test_csv[features]), y_test_csv)
r_r2_svm = r2_score(svm.predict(test_csv[features]), y_test_csv)

#metrics for the csv XGB
r_mle_xgb = mean_log_error(xgboost.predict(test_csv[features]), y_test_csv)
r_mape_xgb = mean_absolute_percentage_error(xgboost.predict(test_csv[features]), y_test_csv)
r_mae_xgb  = np.median(abs(xgboost.predict(test_csv[features])-y_test_csv)/y_test_csv)
r_mse_xgb = mean_squared_error(abs(xgboost.predict(test_csv[features])), y_test_csv)
r_msle_xgb = mean_squared_log_error(abs(xgboost.predict(test_csv[features])), y_test_csv)
r_r2_xgb = r2_score(abs(xgboost.predict(test_csv[features])), y_test_csv)

#metrics for txt RF
""" r_mae_rf = np.median(abs(regr.predict(X_txt_test)-y_test_csv)/y_test_csv)
r_mse_rf = mean_squared_error(regr.predict(X_txt_test), y_test_csv)
r_msle_rf = mean_squared_log_error(regr.predict(X_txt_test), y_test_csv)
r_r2_rf = r2_score(regr.predict(X_txt_test), y_test_csv)
 """
#metrics for txt SVM
""" r_mae_svm = np.median(abs(svm.predict(X_txt_test)-y_test_csv)/y_test_csv)
r_mse_svm = mean_squared_error(svm.predict(X_txt_test), y_test_csv)
r_msle_svm = mean_squared_log_error(svm.predict(X_txt_test), y_test_csv)
r_r2_svm = r2_score(svm.predict(X_txt_test), y_test_csv)
 """
#metrics for txt XGB
""" r_mae_xgb = np.median(abs(xgboost.predict(X_txt_test)-y_test_csv)/y_test_csv)
r_mse_xgb = mean_squared_error(abs(xgboost.predict(X_txt_test)), y_test_csv)
r_msle_xgb = mean_squared_log_error(abs(xgboost.predict(X_txt_test)), y_test_csv)
r_r2_xgb = r2_score(abs(xgboost.predict(X_txt_test)), y_test_csv)
 """
#printing metrics
#RF
print("RF:")
print("r_mle_rf:" + str(r_mle_rf))
print("r_mape_rf:" + str(r_mape_rf))
print("r_mae_rf: " + str(r_mae_rf))
print("r_mse_rf: " + str(r_mse_rf))
print("r_msle_rf: " + str(r_msle_rf))
print("r_r2_rf: " + str(r_r2_rf))
#SVM
print("SVM:")
print("r_mle_svm:" + str(r_mle_svm))
print("r_mape_svm" + str(r_mape_svm))
print("r_mae_svm: " + str(r_mae_svm))
print("r_mse_svm: " + str(r_mse_svm))
print("r_msle_svm: " + str(r_msle_svm))
print("r_r2_svm: " + str(r_r2_svm))
#XGB
print("XGB:")
print("r_mle_xgb:" + str(r_mle_xgb))
print("r_mape_xgb:" + str(r_mape_xgb))
print("r_mae_xgb: " + str(r_mae_xgb))
print("r_mse_xgb: " + str(r_mse_xgb))
print("r_msle_xgb: " + str(r_msle_xgb))
print("r_r2_xgb: " + str(r_r2_xgb))

#adding metrics to log
#RF
mlflow.log_metric("r_mle_rf", r_mle_rf)
mlflow.log_metric("r_mape_rf", r_mape_rf)
mlflow.log_metric("r_mae_rf",r_mae_rf)
mlflow.log_metric("r_mse_rf",r_mse_rf)
mlflow.log_metric("r_msle_rf",r_msle_rf)
mlflow.log_metric("r_r2_rf",r_r2_rf)
#SVM
mlflow.log_metric("r_mle_svm", r_mle_svm)
mlflow.log_metric("r_mape_svm", r_mape_svm)
mlflow.log_metric("r_mae_svm",r_mae_svm)
mlflow.log_metric("r_mse_svm",r_mse_svm)
mlflow.log_metric("r_msle_svm",r_msle_svm)
mlflow.log_metric("r_r2_svm",r_r2_svm)
#XGB
mlflow.log_metric("r_mle_xgb", r_mle_xgb)
mlflow.log_metric("r_mape_xgb", r_mape_xgb)
mlflow.log_metric("r_mae_xgb",r_mae_xgb)
mlflow.log_metric("r_mse_xgb",r_mse_xgb)
mlflow.log_metric("r_msle_xgb",r_msle_xgb)
mlflow.log_metric("r_r2_xgb",r_r2_xgb)

print("Building the final model...")
#result for csv
result = pd.concat([train_csv,test_csv])
#result for txt
#result_txt = np.concatenate((X_txt_train, X_txt_test))
y = result['price']
#csv
regr.fit(result[features], y)
svm.fit(result[features], y)
xgboost.fit(result[features], y)
#txt
#regr.fit(result_txt, y)
#svm.fit(result_txt, y)
#xgboost.fit(result_txt, y)

joblib.dump(regr,city+"_rf.pkl")
log_model(regr, "model_rf")

joblib.dump(svm,city+"_svm.pkl")
log_model(svm, "model_svm")

joblib.dump(xgboost,city+"_xgb.pkl")
log_model(xgboost, "model_xgb")

