import logging
import os, sys
import inspect
import pandas as pd
from sklearn.externals import joblib
import mlflow
from mlflow.sklearn import log_model

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

# Create a dataframe with the four feature variables
train_csv = pd.read_csv(sys.argv[1])
test_csv = pd.read_csv(sys.argv[2])

city = sys.argv[1].split("_")[0] 
print(city)

train_csv = train_csv.drop(['url'],axis=1)
test_csv = test_csv.drop(['url'],axis=1)

train_csv['type'] = train_csv['type'].map({'apart': 1, 'house': 0})
test_csv['type'] = test_csv['type'].map({'apart': 1, 'house': 0})

#remove the rent rows, leaving only 'sell' operation
train_csv['operation'] = train_csv['operation'].map({'sell': 1, 'rent': 0})
test_csv['operation'] = test_csv['operation'].map({'sell': 1, 'rent': 0})

train_csv = train_csv[train_csv.operation != 0]
test_csv = test_csv[test_csv.operation != 0]

# Create a list of the feature column's names
features = train_csv.columns[1:]

# train_csv['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y = train_csv['price']
y_test_csv = test_csv['price']

train_csv_sample = train_csv.sample(n=len(train_csv), random_state=2)

# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train_csv_sample))
print('Number of observations in the test data:',len(test_csv))
#print('TRAIN SAMPLES VALUES:', train_csv_sample)


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
    score = cross_val_score(rfr, train_csv_sample.iloc[:,1:7].as_matrix(), train_csv_sample.iloc[:,0].as_matrix(), cv=11, scoring=rmse_scorer)
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

regr.fit(train_csv[features], y)

# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
regr.predict(test_csv[features])

r_mae  = np.median(abs(regr.predict(test_csv[features])-y_test_csv)/y_test_csv)
print(r_mae)
mlflow.log_metric("r_mae",r_mae)

print("Building the final model...")
result = pd.concat([train_csv,test_csv])
y = result['price']
regr.fit(result[features], y)
joblib.dump(regr,city+".pkl")
log_model(regr, "model")

