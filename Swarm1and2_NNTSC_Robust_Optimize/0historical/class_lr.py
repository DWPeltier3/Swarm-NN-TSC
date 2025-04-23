import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module

from timeit import default_timer as timer
from utils.elapse import elapse_time
from utils.resources import print_resources
import utils.params as params
from utils.datapipeline import import_data

print("******************")
print("  LOG REGRESSION")
print("******************")

## INTRO
start = timer() # start timer to calculate run time
GPUs=print_resources() # computation resources available
hparams = params.get_hparams() # parse BASH run-time hyperparameters (used throughout script below)
params.save_hparams(hparams) #create model folder and save hyperparameters list .txt
## Updae params for sklearn model
hparams.model_type="lstm"
hparams.output_type='mc'
hparams.output_length="seq"

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
x_train_original=x_train
x_train = x_train.reshape(-1, 40)  # Reshape to (batch*time, feature)
x_test = x_test.reshape(-1, 40)  # Reshape to (batch*time, feature)
y_train=y_train.ravel()
y_test=y_test.ravel()
print('\n*** DATA for SK ***')
print('xtrain shape:',x_train.shape)
print('xtest shape:',x_test.shape)
print('ytrain shape:',y_train.shape)
print('ytest shape:',y_test.shape)


## MODEL
# Create and train the logistic regression model
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, max_iter=1000)
lr.fit(x_train, y_train)
# Evaluate the model
train_accuracy = lr.score(x_train, y_train)
test_accuracy = lr.score(x_test, y_test)
print(f'Training Accuracy: {train_accuracy*100:.2f}%')
print(f'Test Accuracy: {test_accuracy*100:.2f}%')


##  COEFFICIENT PLOT
# Get coefficients
coefficients = lr.coef_
# Plot coefficients
plt.figure(figsize=(10, 5))
plt.bar(range(coefficients.shape[1]), coefficients[0], align='center')
plt.title('Coefficient plot')
plt.savefig(hparams.model_dir + "Coefficient_plot_LogReg.png")


## CONFUSION MATRIX
# Predict labels for test set
y_pred = lr.predict(x_test)
# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(hparams.model_dir + "confusion_matrix_LogReg.png")


## TIME SERIES PLOTS (VISUALIZE PATTERNS)
# Select a random sample
sample_idx = np.random.randint(0, x_train_original.shape[0])
# Get data for that sample
sample_data = x_train_original[sample_idx]
# Plot positions and velocities over time for each agent
num_subplot=math.ceil(math.sqrt(num_agents))
plt.figure(figsize=(20,20))
for agent_idx in range(num_agents):
    plt.subplot(num_subplot,num_subplot, agent_idx + 1)
    plt.plot(sample_data[:, agent_idx], label='Px')
    plt.plot(sample_data[:, agent_idx+num_agents], label='Py')
    plt.plot(sample_data[:, agent_idx+2*num_agents], label='Vx')
    plt.plot(sample_data[:, agent_idx+3*num_agents], label='Vy')
    plt.legend()
    plt.title(f'Agent {agent_idx + 1}')
plt.savefig(hparams.model_dir + "Agent_feature_plots_LogReg.png")


##PCA
# Perform PCA 2D
pca = PCA(n_components=3)
pca_result = pca.fit_transform(x_train)
# Scatter plot of the first two principal components
plt.figure(figsize=(10, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_train, cmap='jet', alpha=0.5, marker=".")
plt.colorbar(label='Class label')
plt.title('2D PCA of data')
plt.savefig(hparams.model_dir + "PCA_2D_LogReg.png")

# Perform PCA 3D
pca = PCA(n_components=3)
pca_result = pca.fit_transform(x_train)
# Scatter plot of the first three principal components
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=y_train, cmap='jet', alpha=0.5, marker=".")
plt.colorbar(sc, label='Class label', orientation='horizontal')
ax.set_title('3D PCA of data')
plt.savefig(hparams.model_dir + "PCA_3D_LogReg.png")


## PRINT ELAPSE TIME
elapse_time(start)