import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
import pickle
import sys

from timeit import default_timer as timer
from utils.elapse import elapse_time
from utils.resources import print_resources
import utils.params as params
from utils.datapipeline import import_data
from utils.results import print_cm


print("******************")
print("  RANDOM FOREST")
print("******************")

## INTRO
start = timer() # start timer to calculate run time
GPUs=print_resources() # computation resources available
hparams = params.get_hparams() # parse BASH run-time hyperparameters (used throughout script below)
params.save_hparams(hparams) #create model folder and save hyperparameters list .txt
## SET "SAFE" HPARAMS for sklearn model
hparams.model_type='cn'
hparams.output_length='vec'

## IMPORT DATA
x_train, y_train, x_test, y_test, num_classes, cs_idx, input_shape, output_shape = import_data(hparams)

train_runs=x_train.shape[0]
test_runs=x_test.shape[0]
time_steps=x_train.shape[1]
num_features=x_train.shape[2]
num_agents=num_features//4
print('train runs:',train_runs)
print('test runs:',test_runs)
print('time steps:',time_steps)
print('num features:',num_features)
print('num agents:',num_agents)


## Reshape data for sklearn
# x_train_original=x_train # used to visualize input data
x_train = x_train.reshape(train_runs, -1)  # Reshape to (batch, time*feature)
x_test = x_test.reshape(test_runs, -1)  # Reshape to (batch, time*feature)
if hparams.output_type == 'mc': # col vector to [batch,] format; NOT for multilabel
    y_train=y_train.ravel()
    y_test=y_test.ravel()
print('\n*** DATA for SK ***')
print('xtrain shape:',x_train.shape)
print('xtest shape:',x_test.shape)
print('ytrain shape:',y_train.shape)
print('ytest shape:',y_test.shape)


## MODEL
# Random Forest MULTICLASS classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
if hparams.output_type == 'mc':
    # "Train" model 
    clf.fit(x_train, y_train)
    # Inference on test set
    y_pred = clf.predict(x_test)
    # Get feature importances
    feature_importances = clf.feature_importances_
    # Reshape feature importances to match the original data shape (excluding the batch size); original data shape as (batch, time, feature)
    reshaped_importances = feature_importances.reshape((time_steps, num_features))
    # Caluclate size of "fit"/"trained" model
    p = pickle.dumps(clf)
    print(f'Multi-CLASS model size: {sys.getsizeof(p)} bytes')
# Random Forest MULTILABEL classifier
if hparams.output_type == 'ml':
    # Create and "train" model 
    multi_target_forest = MultiOutputClassifier(clf, n_jobs=-1)
    multi_target_forest.fit(x_train, y_train)
    # Inference on test set
    y_pred = multi_target_forest.predict(x_test)
    # Get feature importances
    feature_importances = multi_target_forest.estimators_[0].feature_importances_ #[0]=1st attribute, [1]=2nd
    # Reshape feature importances to match the original data shape (excluding the batch size); original data shape as (batch, time, feature)
    reshaped_importances = feature_importances.reshape((time_steps, num_features))
    # Caluclate size of "fit"/"trained" model
    p = pickle.dumps(multi_target_forest)
    print(f'Multi-LABEL model size: {sys.getsizeof(p)} bytes')
# Evaluate the classifier
print('Accuracy:', accuracy_score(y_test, y_pred))
## PRINT ELAPSE TIME
elapse_time(start)

## RESULTS
class_names = ['Greedy', 'Greedy+', 'Auction', 'Auction+']
attribute_names = ["COMMS", "PRONAV"]
print_cm(hparams, y_test, y_pred, class_names, attribute_names)

# VISUALIZE FEATURE IMPORTANCE
num_subplot=math.ceil(math.sqrt(time_steps))
plt.figure(figsize=(40,30)) # width/height
# plt.suptitle('Random Forest Feature Importance vs. Time Step')
for i, importances in enumerate(reshaped_importances):
    plt.subplot(num_subplot,num_subplot, i + 1)
    plt.title(f'Time Step {i}')
    plt.bar(range(num_features), importances, align='center')
    plt.xlabel('Features: Pos(0-19) & Vel(20-39)')
    plt.ylabel('Feature Importance [weight]')
plt.savefig(hparams.model_dir + "Feature_Importance.png")


## PRINT ELAPSE TIME
elapse_time(start)
