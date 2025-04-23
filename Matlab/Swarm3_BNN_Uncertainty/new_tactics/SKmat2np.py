import argparse
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer

## Function for calculating elapse time
def elapse_time(start):
    end = timer()
    elapse=end-start

    hours, remainder = divmod(elapse, 3600)
    minutes, seconds = divmod(remainder, 60)

    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)

    # Return formatted time as a string
    return f"{hours}h, {minutes}m, {seconds}s"

## INTRO
start = timer()                 # start timer to calculate run time

## Initialize parser
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument('--decoy_motion', type=str, help='decoy motion')
parser.add_argument('--num_att', type=int, help='number of attackers (killers)')
parser.add_argument('--num_def', type=int, help='number of defenders')
parser.add_argument('--mat_list', nargs='+', help='list of .mat file names to combine; each is a class')
# Parse the arguments
args = parser.parse_args()

# Define input variables
decoy_motion = args.decoy_motion
num_att = args.num_att
num_def = args.num_def
mat_list = args.mat_list

## PRINT HEADER FOR LOG FILE
print(f'MATLAB .mat --> Numpy Converter\n'
      f'Num_Att: {num_att}\n'
      f'Num_Def: {num_def}\n'
      f'Defender Motion: {decoy_motion}\n'
      f'MAT Files: {mat_list}\n')

## IMPORT MATLAB matrix as np array
results_name = f"NEW_{num_att}v{num_def}_{decoy_motion}"    # updated later for NEW2 saving
mat_folder = f"/home/donald.peltier/swarm/data/mat_files/" # folder that contains .mat files
mat_path = os.path.join(mat_folder, results_name)
mat_list = [os.path.join(mat_path, mat_file) for mat_file in mat_list]

mat_data = []
# seedstart = []               # seedstart added in case want to use TEST SET ONLY
for mat_file in mat_list:
    mat = sio.loadmat(mat_file)
    mat_array=np.array(mat['data'])
    # print(f"\nMat File: {mat_file}\nMat array shape: {mat_array.shape}\n")
    mat_data.append(mat_array)
    # seedstart.append(np.array(mat['seedstart']))
combined_mat = np.vstack(mat_data)
# seedstart = np.vstack(seedstart)

## DETERMINE DIMENSIONS
num_runs = len(combined_mat)
runs = [len(mat) for mat in mat_data]
start_indices = [0] + np.cumsum(runs).tolist()[:-1] # Calculate the starting index for each class
num_feat = len(combined_mat[0][0][0][0])
time = [len(run[0][0]) for run in combined_mat]
max_time = max(time)
min_time = min(time)

print('num runs:', num_runs)
for i, run in enumerate(runs, start=1):
    print(f'run{i}:', run)
print('num features:', num_feat)
print('max time:', max_time)
print('min time:', min_time)


results_name = f"SK2_{num_att}v{num_def}_{decoy_motion}"


## PLOT TIME STATISTICS
# Calculate the mean
mean_time = statistics.mean(time)
# Calculate the variance
variance_value = statistics.variance(time)
# Calculate the median
median_value = statistics.median(time)
print(f"The mean is: {mean_time}")
print(f"The variance is: {variance_value}")
print(f"The median is: {median_value}")
# Create a histogram from the time data
plt.hist(time, bins=50, color='blue', edgecolor='black')
# Set the x-axis major ticks to 100 and minor ticks to 20
major_ticks = range(0, max(time) + 100, 100)
minor_ticks = range(0, max(time) + 20, 20)
# Configure major and minor ticks
plt.xticks(major_ticks)
plt.gca().set_xticks(minor_ticks, minor=True)
# Include grid lines for major ticks
plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
# Label the mean line
plt.axvline(mean_time, color='red', linestyle='dashed', linewidth=2)
plt.text(mean_time+2, plt.ylim()[1]*0.9, f'mean={mean_time:.0f}', color='red')
plt.text(mean_time+2, plt.ylim()[1]*0.7, f'variance={variance_value:.0f}', color='red')
plt.text(mean_time+2, plt.ylim()[1]*0.5, f'median={median_value:.0f}', color='red')
plt.xlabel('Time Steps')
plt.ylabel('Frequency')
plt.title('Histogram of Engagement Total Time Steps')
# Show the plot
plt.savefig(os.path.join(mat_path, f"Time_histogram_{results_name}.png"))

print(f"\nTime to import .mat files: {elapse_time(start)}")
new_timer = timer()

## CREATE PYTHON DATA ARRAY
# MIN TIME (truncate each run to "min time" length; prevents ragged array, all instances have same time length)
data = combined_mat[0,0][:,:min_time]   # first run
for run in range(1, num_runs):          # stack subsequent runs
    temp = combined_mat[run,0][:,:min_time]
    data = np.vstack((data, temp))
print('data shape:', data.shape)

print(f"\nTime to truncate to min time: {elapse_time(new_timer)}")
new_timer = timer()

## CREATE LABELS
labels = []
for i, run in enumerate(runs):
    labels.append(i * np.ones((run, 1), dtype=int))
label = np.vstack(labels)
print('label shape', label.shape)
for start_idx in start_indices:
    print(f"label sample\n {label[start_idx:start_idx+5]}\n")

print(f"\nTime to create labels: {elapse_time(new_timer)}")
new_timer = timer()

# ## CREATE TESTSET
# x_test = data
# y_test = label

## NOT NEEDED IF ONLY TESTSET ######################
## SPLIT DATA (TRAIN & TEST)...RANDOMLY SPLITS ("SHUFFLES")
test_percentage = 0.25
x_train_list, x_test_list, y_train_list, y_test_list = [], [], [], []

for i in range(len(runs)):
    start_idx = start_indices[i]
    end_idx = start_indices[i] + runs[i]
    x_train, x_test, y_train, y_test = train_test_split(
        data[start_idx:end_idx],
        label[start_idx:end_idx],
        test_size=test_percentage,
        random_state=0
    )
    x_train_list.append(x_train)
    x_test_list.append(x_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)

x_train = np.vstack(x_train_list)
x_test = np.vstack(x_test_list)
y_train = np.vstack(y_train_list)
y_test = np.vstack(y_test_list)

print('x train shape', x_train.shape)
print('y train shape', y_train.shape)
################################################
print('x test shape', x_test.shape)
print('y test shape', y_test.shape)

print(f"\nTime to split dataset (train & test): {elapse_time(new_timer)}")
new_timer = timer()


## NOT NEEDED IF ONLY TESTSET ######################
print('\nORIGINAL x train sample (first instance, first time step):\n', x_train[0,0])
print('\nORIGINAL y train sample:', y_train[0])

## SHUFFLE TRAIN DATA
x_train, y_train = shuffle(x_train, y_train, random_state=0)
print('\nSHUFFLE x train sample (first instance, first time step):\n', x_train[0,0])
print('\nSHUFFLE y train sample:', y_train[0])

## NORMALIZE DATA
# FIT to training data only, then transform both training and test data (p.70 HOML)

## Standard Scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
print('\nSCALED x train sample (first instance, first time step):\n', x_train[0,0])
print('\nmean:', scaler.mean_)
print('\nvariance:', scaler.var_)

# TRANSFORM test dataset
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
# print('\nx TEST SCALED example (first instance, first time step):\n', x_test[0,0])
################################################

print(f"\nTime to normalize dataset: {elapse_time(new_timer)}")

## SAVE DATASET
data_path = '/home/donald.peltier/swarm/data/'
filename = f'data_{results_name}.npz'
# filename = f'testset_data_{results_name}.npz' # Testset ONLY
full_path = os.path.join(data_path, filename)

# Save dataset as .npz
np.savez(full_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
# np.savez(full_path, x_test=x_test, y_test=y_test, seedstart=seedstart)  # Testset ONLY
print(f"saved dataset in {full_path}")


## NOT NEEDED IF ONLY TESTSET ######################
# Save mean/variance as .npz
mean_var_folder = 'mean_var/'
filename = f'mean_var_{results_name}.npz'
full_path = os.path.join(data_path, mean_var_folder, filename)
np.savez(full_path, mean=scaler.mean_, variance=scaler.var_)
print(f"saved mean/var in {full_path}")

# Save mean/variance as .mat (for matlab NN inference datapipeline)
filename=f'mean_var_{results_name}.mat'
full_path = os.path.join(data_path, mean_var_folder, filename)
sio.savemat(full_path, {'mean': scaler.mean_, 'variance': scaler.var_})
################################################

## PRINT TOTAL SCRIPT ELAPSE TIME
print(f"\nTotal Script Time: {elapse_time(start)}")


