import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statistics
import argparse
import os

# Initialize parser
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument('--scaling_factor', type=float, required=True,
                    help='The scaling factor for the computation.')
# Parse the arguments
args = parser.parse_args()
# Use args.scaling_factor as needed in your script
scaling_factor = args.scaling_factor
print(f"Scaling factor: {scaling_factor}")

## IMPORT MATLAB matrix as np array
data_folder='/home/donald.peltier/swarm/data/mat_files/10v10_r4800_c4_a10/' # folder that contains .mat files
data1=data_folder+'data_g.mat'
data2=data_folder+'data_gp.mat'
data3=data_folder+'data_a.mat'
data4=data_folder+'data_ap.mat'
mat1=sio.loadmat(data1)
mat2=sio.loadmat(data2)
mat3=sio.loadmat(data3)
mat4=sio.loadmat(data4)
mat1=np.array(mat1['data'])
mat2=np.array(mat2['data'])
mat3=np.array(mat3['data'])
mat4=np.array(mat4['data'])
mat=np.vstack((mat1,mat2,mat3,mat4))

## DETERMINE DIMENSIONS
num_runs = len(mat)
run1 = len(mat1) #number of class 1
run2 = len(mat2) #number of class 2
run3 = len(mat3) #number of class 3
run4 = len(mat4) #number of class 4
#class starting index
c1si=0
c2si=run1
c3si=run1+run2
c4si=run1+run2+run3
num_feat=len(mat[0][0][0][0])
time=[]
for run in range(num_runs):
    time.append(len(mat[run][0][0]))
max_time = max(time)
min_time = min(time)
print('num runs:',num_runs)
print('run1:',run1)
print('run2:',run2)
print('run3:',run3)
print('run4:',run4)
print('num features:',num_feat)
# print('time lengths sample:',time[0:10]) #gets too long with large number of runs
print('max time:',max_time)
print('min time:',min_time)

## PLOT TIME STATISTICS
# Calculate the mean
mean_time = statistics.mean(time)
# Calculate the variance
variance_value = statistics.variance(time)
print(f"The mean is: {mean_time}")
print(f"The variance is: {variance_value}")
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
plt.xlabel('Time Steps')
plt.ylabel('Frequency')
plt.title('Histogram of Engagement Total Time Steps')
# Show the plot
plt.savefig("/home/donald.peltier/swarm/code/matlab/" + "Time_histogram.png")

## CREATE PYTHON DATA ARRAY
# MIN TIME (truncate each run to "min time" length; prevents ragged array, all instances have same time length)
data=mat[0,0][:,:min_time] #first run
for run in range(1,num_runs): #stack subsuquent runs
    temp=mat[run,0][:,:min_time]
    data=np.vstack((data,temp))
print('data shape:',data.shape)

## CREATE LABELS
# TODO: make this "not hard coded"; # labels = # of matlab imports
label = np.vstack((np.zeros((run1,1),dtype=int), np.ones((run2,1),dtype=int), 2*np.ones((run3,1),dtype=int), 3*np.ones((run4,1),dtype=int))) #np.ones default type is float64
print('label shape', label.shape)
print('label sample', label[c1si:c1si+5],'\n', label[c2si:c2si+5],'\n', label[c3si:c3si+5],'\n', label[c4si:c4si+5])

## ADD NOISE TO DATA (MEASUREMENTS)
# Define your noise scales:
# scaling_factor=0.02 # noise percentage of measurement
noise_scale_position = 40*scaling_factor  # position exist 0-40
noise_scale_velocity = 1*scaling_factor  # velocity exist 0-1
print(f"pos_scale {noise_scale_position} and vel_scale {noise_scale_velocity}")
np.random.seed(0) # define seed for Numpy random

# def add_noise(data, noise_scale_position, noise_scale_velocity):
#     # Assuming the first 20 features are position (Px, Py), and the next 20 are velocity
#     position_features = data[..., :20]  # Replace 20 with the correct number of position features
#     velocity_features = data[..., 20:]  # Replace 20 with the correct starting index for velocity features
#     # Generate noise
#     position_noise = np.random.normal(0, noise_scale_position, position_features.shape)
#     velocity_noise = np.random.normal(0, noise_scale_velocity, velocity_features.shape)
#     # Check the noise mean for a single feature across all time steps in the first batch
#     feature_index = 0  # Feature index to check
#     mean_feature_noise = np.mean(position_noise[0, :, feature_index])
#     print(f"Mean of position noise for batch 0, feature {feature_index} across all time steps:", mean_feature_noise)
#     # Add noise to features
#     noisy_position_features = position_features + position_noise
#     noisy_velocity_features = velocity_features + velocity_noise
#     # Combine the features back into the original data structure
#     noisy_data = np.concatenate((noisy_position_features, noisy_velocity_features), axis=-1)
#     return noisy_data
# # Add noise to your data
# data = add_noise(data, noise_scale_position, noise_scale_velocity)

def add_noise(data, noise_scale_position, noise_scale_velocity):
    # Parameters
    num_batches = data.shape[0]  # Number of batches in your data
    num_timesteps = data.shape[1]  # Number of time steps in your data
    num_position_features = data.shape[2]//2  # Number of position features
    num_velocity_features = data.shape[2]//2  # Number of velocity features
    total_features = num_position_features + num_velocity_features
    # print(f"batches {num_batches}, timesteps {num_timesteps}, pos_feat {num_position_features}, vel_feat {num_velocity_features}, total_feat {total_features}")
    # Generate noise for each feature and batch
    noise_tensor = np.zeros((num_batches, num_timesteps, total_features))
    for batch in range(num_batches):
        for feature in range(total_features):
            if feature < num_position_features:
                # Generate position noise for this feature across all time steps
                noise_vector = np.random.normal(0, noise_scale_position, num_timesteps)
            else:
                # Generate velocity noise for this feature across all time steps
                noise_vector = np.random.normal(0, noise_scale_velocity, num_timesteps)
            # Subtract the mean to ensure zero mean
            noise_vector -= np.mean(noise_vector)
            # Assign this noise vector to the appropriate place in the noise tensor
            noise_tensor[batch, :, feature] = noise_vector
    # Add the noise tensor to the original data
    noisy_data = data + noise_tensor
    # Now you can check if the mean is zero for any batch and feature across time
    batch_index = 0
    feature_index = 0  # Change as needed
    # print(f"Shape of noise tensor {noise_tensor.shape}")
    # print(f"Sample of Px noise [B0,F0]: {noise_tensor[batch_index, :, feature_index]}")
    print(f"Mean noise for batch {batch_index}, feature {feature_index}:", np.mean(noise_tensor[batch_index, :, feature_index]))
    # print(f"Noisy data sample for batch {batch_index}, feature {feature_index}:", noisy_data[batch_index, :, feature_index])
    return noisy_data
# Add noise to your data
data = add_noise(data, noise_scale_position, noise_scale_velocity)

## SPLIT DATA (TRAIN & TEST)
test_percentage=0.25
x_c1train, x_c1test, y_c1train, y_c1test = train_test_split(data[:c2si], label[:c2si], test_size=test_percentage, random_state=0) #split each category separately (equal representation during training and testing)
x_c2train, x_c2test, y_c2train, y_c2test = train_test_split(data[c2si:c3si], label[c2si:c3si], test_size=test_percentage, random_state=0)
x_c3train, x_c3test, y_c3train, y_c3test = train_test_split(data[c3si:c4si], label[c3si:c4si], test_size=test_percentage, random_state=0)
x_c4train, x_c4test, y_c4train, y_c4test = train_test_split(data[c4si:], label[c4si:], test_size=test_percentage, random_state=0)

x_train = np.vstack((x_c1train,x_c2train,x_c3train,x_c4train)) #recombine datasets
x_test = np.vstack((x_c1test,x_c2test,x_c3test,x_c4test))
y_train = np.vstack((y_c1train,y_c2train,y_c3train,y_c4train))
y_test = np.vstack((y_c1test,y_c2test,y_c3test,y_c4test))

print('x train shape', x_train.shape)
print('y train shape', y_train.shape)
print('x test shape', x_test.shape)
print('y test shape', y_test.shape)

print('\nx TRAIN sample (first instance, firt time step):\n',x_train[0,0])
print('y train sample:',y_train[0])

## SHUFFLE TRAIN DATA
x_train, y_train = shuffle(x_train, y_train, random_state=0)
print('\nSHUFFLE')
print('\nx train sample:\n',x_train[0,0])
print('\ny train sample:',y_train[0])

## NORMALIZE DATA
# FIT to training data only, then transform both training and test data (p.70 HOML)
# Comment/Uncomment lines to switch between "Standard Scaler" and "MinMax Scaler"
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
print('\nx TRAIN SCALED example (first instance, firt time step):\n',x_train[0,0])
## Standard Scaler attributes
print('\nmean:',scaler.mean_)
print('\nvariance:',scaler.var_)

x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
print('\nx TEST SCALED example (first instance, firt time step):\n',x_test[0,0])

## SAVE DATASET
post=int(round(scaling_factor*100))
path='/home/donald.peltier/swarm/data/historical/noise/Noise1'
# DvA: #def v #att; 4cl=4 classes; r=# runs; s=scaled; a10=acceleration 10 steps; rs=random start, ns=normal start
filename=f'data_10v10_r4800s_4cl_a10_noise{post}.npz'
# filename='data_100v100_r4800s_4cl_a10_ns.npz'
full_path = os.path.join(path, filename)
np.savez(full_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
print(f"saved {post} data in {full_path}")