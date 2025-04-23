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
parser.add_argument('--swarm_size', type=int, required=True,
                    help='swarm size')
# Parse the arguments
args = parser.parse_args()
swarm_size = args.swarm_size

## IMPORT MATLAB matrix as np array
data_folder=f"/home/donald.peltier/swarm/data/mat_files/{swarm_size}v{swarm_size}_r4800_c4_a10/" # folder that contains .mat files; varying swarm size
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
plt.savefig("/home/donald.peltier/swarm/code/matlab/" + f"Time_histogram_{swarm_size}.png")


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


## SPLIT DATA (TRAIN & TEST)...RANDOMLY SPLITS ("SHUFFLES")
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

print('\nORIGINAL x train sample (first instance, firt time step):\n',x_train[0,0])
print('\nORIGINAL y train sample:',y_train[0])


## SHUFFLE TRAIN DATA
x_train, y_train = shuffle(x_train, y_train, random_state=0)
print('\nSHUFFLE x train sample (first instance, firt time step):\n',x_train[0,0])
print('\nSHUFFLE y train sample:',y_train[0])


## NORMALIZE DATA
# FIT to training data only, then transform both training and test data (p.70 HOML)
# Comment/Uncomment lines to switch between "Standard Scaler" and "MinMax Scaler"

## Standard Scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
print('\nSCALED x train sample (first instance, firt time step):\n',x_train[0,0])
print('\nmean:',scaler.mean_)
print('\nvariance:',scaler.var_)

### MinMax Scaler
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
# print('\nSCALED x train sample (first instance, firt time step):\n',x_train[0,0])
# print('\nNum Features:',scaler.n_features_in_)
# print('\nData Min:',scaler.data_min_)
# print('\nData Max:',scaler.data_max_)

# TRANSFORM test dataset
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
# print('\nx TEST SCALED example (first instance, firt time step):\n',x_test[0,0])


## SAVE DATASET & MEAN/VARIANCE
data_path='/home/donald.peltier/swarm/data/'
# DvA: #def v #att; 4cl=4 classes; r=# runs; s=scaled; a10=acceleration 10 steps; rs=random start, ns=normal start
filename=f'data_{swarm_size}v{swarm_size}_r4800s_c4_a10.npz'
full_path = os.path.join(data_path, filename)
# Save dataset as .npz
np.savez(full_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
print(f"saved dataset in {full_path}")
# Save mean/variance as .npz
filename=f'mean_var_{swarm_size}v{swarm_size}_r4800s_c4_a10.npz'
full_path = os.path.join(data_path, filename)
np.savez(full_path, mean=scaler.mean_, variance=scaler.var_)
print(f"saved mean/var in {full_path}")
# Save mean/variance as .mat (for matlab NN inference datapipeline)
mean_var_folder='mean_var/'
filename=f'mean_var_{swarm_size}v{swarm_size}_r4800s_c4_a10.mat'
full_path = os.path.join(data_path, mean_var_folder, filename)
sio.savemat(full_path, {'mean': scaler.mean_, 'variance': scaler.var_})