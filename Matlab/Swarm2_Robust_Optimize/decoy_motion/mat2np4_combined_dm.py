import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

swarm_size = 10
input_num_runs = 10000*4 # used for naming files in Second experiment with 10k runs per tactic per motion
# decoy_motions=["straight", "perpendicular", "split", "semi", "r4800_c4_a10"]
decoy_motions=["star", "str", "perL", "perR", "semi"]

## DESCRIPTION
# Finds overall min time for all motions in decoy_motion list.
# Then creates sub_dataset for each motion in the list.
# Finally, combines all sub_datasets BEFORE scaling the combined dataset.

def find_min_time(swarm_size, decoy_motion):
    ## PRINT HEADER FOR LOG FILE
    print(f'FIND MIN TIME\n'
        f'Swarm Size: {swarm_size}\n'
        f'Decoy Motion: {decoy_motion}\n'
        f'Velocity = random 0.05-0.4\n')
    
    # Define the mapping from motion abbreviations to .mat folders

    # First experiment with 1200 runs per tactic per motion
    # motion_to_folder = {
    #     "star": "10v10_r4800_c4_a10",
    #     "semi": "10v10_semi",
    #     "str": "10v10_str",
    #     "perL": "10v10_perL",
    #     "perR": "10v10_perR",
    # }

    # Second experiment with 10k runs per tactic per motion
    motion_to_folder = {
        "star": "10v10_star_r40000",
        "semi": "10v10_semi_r40000",
        "str": "10v10_str_r40000",
        "perL": "10v10_perL_r40000",
        "perR": "10v10_perR_r40000",
    }
    mat_folder = motion_to_folder.get(decoy_motion)

    ## IMPORT MATLAB matrix as np array
    data_folder=f"/home/donald.peltier/swarm/data/mat_files" # directory of folders that contain .mat files
    mat_path = f"{data_folder}/{mat_folder}/" # folder that contains .mat files for specific motion
    data1=mat_path+'data_g.mat'
    data2=mat_path+'data_gp.mat'
    data3=mat_path+'data_a.mat'
    data4=mat_path+'data_ap.mat'
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
    time=[]
    for run in range(num_runs):
        time.append(len(mat[run][0][0]))
    min_time = min(time)
    print(f'num runs: {num_runs}\n'
          f'min time: {min_time}\n\n')

    return min_time


def create_dataset(swarm_size, decoy_motion, combined_min_time):
    ## PRINT HEADER FOR LOG FILE
    print(f'\nMATLAB .mat --> Numpy Converter\n'
        f'Swarm Size: {swarm_size}\n'
        f'Decoy Motion: {decoy_motion}\n'
        f'Velocity = random 0.05-0.4\n')

    # Define the mapping from motion abbreviations to .mat folders
    
    # First experiment with 1200 runs per tactic per motion
    # motion_to_folder = {
    #     "star": "10v10_r4800_c4_a10",
    #     "semi": "10v10_semi",
    #     "str": "10v10_str",
    #     "perL": "10v10_perL",
    #     "perR": "10v10_perR",
    # }

    # Second experiment with 10k runs per tactic per motion
    motion_to_folder = {
        "star": "10v10_star_r40000",
        "semi": "10v10_semi_r40000",
        "str": "10v10_str_r40000",
        "perL": "10v10_perL_r40000",
        "perR": "10v10_perR_r40000",
    }
    mat_folder = motion_to_folder.get(decoy_motion)

    ## IMPORT MATLAB matrix as np array
    data_folder=f"/home/donald.peltier/swarm/data/mat_files" # directory of folders that contain .mat files
    mat_path = f"{data_folder}/{mat_folder}/" # folder that contains .mat files for specific motion
    data1=mat_path+'data_g.mat'
    data2=mat_path+'data_gp.mat'
    data3=mat_path+'data_a.mat'
    data4=mat_path+'data_ap.mat'
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
    # print('min time:',min_time)

    ## CREATE PYTHON DATA ARRAY
    # MIN TIME (truncate each run to "min time" length; prevents ragged array, all instances have same time length)
    data=mat[0,0][:,:combined_min_time] #first run
    for run in range(1,num_runs): #stack subsuquent runs
        temp=mat[run,0][:,:combined_min_time]
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

    # print('\nx TRAIN sample (first instance, firt time step):\n',x_train[0,0])
    # print('y train sample:',y_train[0])

    return x_train, y_train, x_test, y_test


## FIND MIN TIME from all combined sub-datasets
print("FIND MIN TIME")
min_times = []
for motion in decoy_motions:
    min_time = find_min_time(swarm_size, motion)
    min_times.append(min_time)
combined_min_time = min(min_times)
print('combined min time:', combined_min_time)


## COMBINE ALL SUBDATASETS
print("COMBINE DATASETS")
# Initialize containers for combined data
x_trains, y_trains, x_tests, y_tests = [], [], [], []
# Run create_dataset for each motion
for motion in decoy_motions:
    x_train, y_train, x_test, y_test = create_dataset(swarm_size, motion, combined_min_time)
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
print('\nORIGINAL x train sample (first instance, firt time step):\n',x_train[0,0])
print('\nORIGINAL y train sample:',y_train[0])


## SHUFFLE TRAIN DATA
x_train, y_train = shuffle(x_train, y_train, random_state=0)
print('\nSHUFFLE x train sample (first instance, firt time step):\n',x_train[0,0])
print('\nSHUFFLE y train sample:',y_train[0])


## NORMALIZE DATA
# FIT to training data only, then transform both training and test data (p.70 HOML)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
print('\nSCALED x train sample (first instance, firt time step):\n',x_train[0,0])
# TRANSFORM test dataset
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
# print('\nx TEST SCALED example (first instance, firt time step):\n',x_test[0,0])
## Standard Scaler attributes: print (and save) for use during inference
print('\nmean:',scaler.mean_)
print('\nvariance:',scaler.var_)


## SAVE DATASET & MEAN/VARIANCE
data_path='/home/donald.peltier/swarm/data/'
# filename=f'data_{swarm_size}v{swarm_size}_combined.npz' # First experiment with 1200 runs per tactic per motion
filename=f'data_{swarm_size}v{swarm_size}_combined_r{input_num_runs}.npz' # Second experiment with 10k runs per tactic per motion
full_path = os.path.join(data_path, filename)

# Save dataset as .npz
np.savez(full_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
print(f"saved dataset in {full_path}")

# Save mean/variance as .npz
mean_var_folder='mean_var/'
# filename=f'mean_var_{swarm_size}v{swarm_size}_combined.npz' # First experiment with 1200 runs per tactic per motion
filename=f'mean_var_{swarm_size}v{swarm_size}_combined_r{input_num_runs}.npz' # Second experiment with 10k runs per tactic per motion
full_path = os.path.join(data_path, mean_var_folder, filename)
np.savez(full_path, mean=scaler.mean_, variance=scaler.var_)
print(f"saved mean/var in {full_path}")

# Save mean/variance as .mat (for matlab NN inference datapipeline)
# filename=f'mean_var_{swarm_size}v{swarm_size}_combined.mat' # First experiment with 1200 runs per tactic per motion
filename=f'mean_var_{swarm_size}v{swarm_size}_combined_r{input_num_runs}.mat' # Second experiment with 10k runs per tactic per motion
full_path = os.path.join(data_path, mean_var_folder, filename)
sio.savemat(full_path, {'mean': scaler.mean_, 'variance': scaler.var_})