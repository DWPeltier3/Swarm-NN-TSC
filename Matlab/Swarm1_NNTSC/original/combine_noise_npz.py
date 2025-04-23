import os
import glob
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

decoy_motion = "star"
num_att = 10
num_def = 10
noise_levels = range(0, 51) #end not included
data_file_pre=f"/home/donald.peltier/swarm/data/historical/noise/Noise1/data_10v10_r4800s_4cl_a10_noise"
text_folder=f"/home/donald.peltier/swarm/logs/datagen/Noise/1"

## DESCRIPTION
# Find overall min_time for all datasets.
# Load, unscale, and shorten each sub_dataset.
# Finally,combine all sub_datasets and then scale the combined dataset.

def find_time(data_file):
    ## PRINT HEADER FOR LOG FILE
    print(f'TIME for Data File: {data_file}...')
    ## Load dataset
    data=np.load(data_file)
    x_train = data['x_train']
    # x_test = data['x_test']
    ## DETERMINE DIMENSIONS
    # num_runs = x_train[1]+x_test[1]
    time = x_train.shape[1]
    print(f'{time}')
    return time


def extract_mean_variance(text_folder, noise):
    mean_var_folder="/home/donald.peltier/swarm/data/mean_var"
    
    if noise==0:
        original_filename=f"mean_var_{num_att}v{num_def}_r4800_c4_a10.npz"
        original_path = os.path.join(mean_var_folder, original_filename)
        original_scaler_data = np.load(original_path)
        mean = original_scaler_data['mean']
        variance = original_scaler_data['variance']
        return mean, variance
    
    # Search for the file ending with "_noise{noise}.txt"
    search_pattern = os.path.join(text_folder, f"*noise{noise}.txt")
    matching_files = glob.glob(search_pattern)
    if not matching_files:
        raise FileNotFoundError(f"No file found for noise level {noise} in {text_folder}")
    
    file_path = matching_files[0] # Assuming there's only one file that matches the pattern
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Extract mean values
        mean_lines = []
        start_collecting = False
        for line in lines:
            if line.startswith("mean:"):
                start_collecting = True
            if start_collecting:
                mean_lines.append(line.strip().split(':')[-1].strip())
            if line.startswith("variance:"):
                break
        mean_str = " ".join(mean_lines)
        mean = np.fromstring(mean_str.strip("[]"), sep=' ')
        # Extract variance values
        variance_lines = []
        start_collecting = False
        for line in lines:
            if line.startswith("variance:"):
                start_collecting = True
            if start_collecting:
                variance_lines.append(line.strip().split(':')[-1].strip())
            if line.startswith("x TEST SCALED"):
                break
        variance_str = " ".join(variance_lines)
        variance = np.fromstring(variance_str.strip("[]"), sep=' ')
    
    # Reshape original_mean and original_variance to match the last dimension of x_train and x_test
    # print(f"Mean: {mean}")
    # print(f"Variance: {variance}")
    mean = mean.reshape(1, 1, -1)
    variance = variance.reshape(1, 1, -1)
    # print(f"RESHAPED Mean: {mean}")
    # print(f"RESHAPED Variance: {variance}")
    
    # Save mean/variance as .npz
    mean_var_folder='/home/donald.peltier/swarm/data/mean_var'
    filename=f'mean_var_{num_att}v{num_def}_r4800_c4_a10_noise{noise}.npz'
    full_path = os.path.join(mean_var_folder, filename)
    np.savez(full_path, mean=mean, variance=variance)
    print(f"saved mean/var in {full_path}")
    
    return mean, variance


def create_dataset(noise, text_folder, data_file, combined_min_time):
    ## PRINT HEADER FOR LOG FILE
    print(f'\nLOAD & UNSCALE for NOISE: {noise}\n')

    ## LOAD DATASET
    data=np.load(data_file)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    ## EXTRACT ORIGINAL MEAN & VARIANCE
    original_mean, original_variance = extract_mean_variance(text_folder, noise)

    ##  UNSCALE DATASET 
    # Unscale the dataset using the original mean and variance
    print('\nSCALED X TRAIN sample (first instance, first time step):\n',x_train[0,0])
    x_train_unscaled = x_train * np.sqrt(original_variance) + original_mean
    x_test_unscaled = x_test * np.sqrt(original_variance) + original_mean
    print('\nUNSCALED X TRAIN sample (first instance, first time step):\n',x_train_unscaled[0,0])

    ## SHORTEN DATASET TO COMBINED MIN TIME
    x_train_unscaled = x_train_unscaled[:,:combined_min_time,:]
    x_test_unscaled = x_test_unscaled[:,:combined_min_time,:]

    return x_train_unscaled, y_train, x_test_unscaled, y_test


## FIND MIN TIME from all combined sub-datasets 
times = []
for noise in noise_levels:
    data_file = f"{data_file_pre}{noise}.npz"
    time = find_time(data_file)
    times.append(time)
combined_min_time = min(times)
print('combined min time:', combined_min_time)


## COMBINE ALL SUBDATASETS
# Initialize containers for combined data
x_trains, y_trains, x_tests, y_tests = [], [], [], []
# Run create_dataset for each sub dataset
for noise in noise_levels:
    data_file = f"{data_file_pre}{noise}.npz"
    x_train, y_train, x_test, y_test = create_dataset(noise, text_folder, data_file, combined_min_time)
    x_trains.append(x_train)
    y_trains.append(y_train)
    x_tests.append(x_test)
    y_tests.append(y_test)
# Combine all datasets
x_train = np.concatenate(x_trains, axis=0)
y_train = np.concatenate(y_trains, axis=0)
x_test = np.concatenate(x_tests, axis=0)
y_test = np.concatenate(y_tests, axis=0)
print('\nCOMBINED DATASET:')
print('x train shape', x_train.shape)
print('y train shape', y_train.shape)
print('x test shape', x_test.shape)
print('y test shape', y_test.shape)
print('\nORIGINAL x train sample (first instance, first time step):\n',x_train[0,0])
print('\nORIGINAL y train sample:',y_train[0])


## SHUFFLE TRAIN DATA
x_train, y_train = shuffle(x_train, y_train, random_state=0)
print('\nSHUFFLE x train sample (first instance, first time step):\n',x_train[0,0])
print('\nSHUFFLE y train sample:',y_train[0])


## NORMALIZE DATA
# FIT to training data only, then transform both training and test data (p.70 HOML)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
print('\nSCALED x train sample (first instance, first time step):\n',x_train[0,0])
# TRANSFORM test dataset
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
## Standard Scaler attributes: print (and save) for use during inference
print('\nmean:',scaler.mean_)
print('\nvariance:',scaler.var_)


## SAVE DATASET & MEAN/VARIANCE

motion_abrev_to_dataname = {
        "star": "r4800_c4_a10",
        "semi": "semi",
        "str": "straight",
        "per": "perpendicular",
        "split": "split",
    }

data_path='/home/donald.peltier/swarm/data/'
filename=f'data_{num_att}v{num_def}_{motion_abrev_to_dataname.get(decoy_motion)}_combined_noise.npz'
full_path = os.path.join(data_path, filename)

# Save dataset as .npz
np.savez(full_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
print(f"saved dataset in {full_path}")

# Save mean/variance as .npz
mean_var_folder='mean_var/'
filename=f'mean_var_{num_att}v{num_def}_{motion_abrev_to_dataname.get(decoy_motion)}_combined_noise.npz'
full_path = os.path.join(data_path, mean_var_folder, filename)
np.savez(full_path, mean=scaler.mean_, variance=scaler.var_)
print(f"saved mean/var in {full_path}")

# Save mean/variance as .mat (for matlab NN inference datapipeline)
filename=f'mean_var_{num_att}v{num_def}_{motion_abrev_to_dataname.get(decoy_motion)}_combined_noise.mat'
full_path = os.path.join(data_path, mean_var_folder, filename)
sio.savemat(full_path, {'mean': scaler.mean_, 'variance': scaler.var_})