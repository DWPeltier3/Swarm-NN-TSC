import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
import seaborn as sns
from utils.savevariables import append_to_csv
import math
import os

## IMPORT DATA
def import_data(hparams):
    
    ## Extract parameters
    num_att=hparams.num_att
    model_type = hparams.model_type  # fc=fully connect, cn=CNN, fcn=FCN, res=ResNet, lstm=LSTM, tr=transformer
    window = hparams.window
    output_type = hparams.output_type  # mc=multiclass, ml=multilabel, mh=multihead
    output_length = hparams.output_length  # vec=vector, seq=sequence
    mean_var_file = hparams.mean_var_file
    if hparams.num_def==-1: # signals use of combined decoy number dataset
        num_def="1C10"  # update 2nd number as required if multiple combDN datasets exist (i.e. 1C10, 1C15)
        print(f"Using COMBINED DECOY NUMBER Dataset; num_def used for labels = {num_def}")
    else:
        num_def=hparams.num_def

    ## LOAD DATA (single dataset)  ***************************************
    # Define the mapping from motion abbreviations to data filenames
    motion_abrev_to_dataname = {
        "star": "r4800_c4_a10",
        "semi": "semi",
        "str": "str",
        "perL": "perL",
        "perR": "perR",
        "comb": "combDM"
    }
    
    data_filename = f"data_{num_att}v{num_def}_{motion_abrev_to_dataname.get(hparams.motion)}.npz"  # normal use
    data_path = f"{hparams.data_folder}/{data_filename}"  # normal use
    # data_path = f"{hparams.data_folder}"                # for specific datasets; noise, broad, etc.
    data=np.load(data_path)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
 

    ## RESCALE DATASET IF MEAN/VAR FILEPATH PROVIDED  ***************************************
    if mean_var_file is not None:
        print('\nRESCALE DATASET using MEAN/VAR FILEPATH PROVIDED  **********************************')
        print('\nOriginal x train sample (first instance, first time step):\n', x_train[0, 0])

        # Load the mean and variance used to scale the original dataset
        mean_var_folder="/home/donald.peltier/swarm/data/mean_var"
        original_filename=f"mean_var_{num_att}v{num_def}_{motion_abrev_to_dataname.get(hparams.motion)}.npz"
        original_path = os.path.join(mean_var_folder, original_filename)
        original_scaler_data = np.load(original_path)
        original_mean = original_scaler_data['mean']
        original_variance = original_scaler_data['variance']
        print(f'\nOriginal scaling: {original_filename}')
        # Load the mean and variance to be used for rescaling (must be same scaling used to train NN inputs)
        new_path = mean_var_file
        new_scaler_data = np.load(new_path)
        new_mean = new_scaler_data['mean']
        new_variance = new_scaler_data['variance']
        print(f'New scaling: {mean_var_file}')

        # Unscale the dataset using the original mean and variance
        x_train_unscaled = x_train * np.sqrt(original_variance) + original_mean
        x_test_unscaled = x_test * np.sqrt(original_variance) + original_mean
        print('\nUnscaled x train sample (first instance, first time step):\n', x_train_unscaled[0, 0])
        # Rescale the dataset using the new mean and variance
        x_train_rescaled = (x_train_unscaled - new_mean) / np.sqrt(new_variance)
        x_test_rescaled = (x_test_unscaled - new_mean) / np.sqrt(new_variance)
        # Verify the rescaled data
        print('\nRescaled x train sample (first instance, first time step):\n', x_train_rescaled[0, 0])
        x_train=x_train_rescaled
        x_test=x_test_rescaled


    ## VELOCITY or POSITION ONLY  ***************************************
    v_idx = num_att * 2
    if hparams.features == 'v':
        print('\n*** VELOCITY ONLY ***')
        x_train = x_train[:, :, v_idx:]
        x_test = x_test[:, :, v_idx:]
        feature_names = ['Vx', 'Vy']
    elif hparams.features == 'p':
        print('\n*** POSITION ONLY ***')
        x_train = x_train[:, :, :v_idx]
        x_test = x_test[:, :, :v_idx]
        feature_names = ['Px', 'Py']
    else:
        feature_names = ['Px', 'Py', 'Vx', 'Vy']
    hparams.feature_names = feature_names

    ## CHARACTERIZE DATA
    time_steps=x_train.shape[1]
    num_features=x_train.shape[2]
    num_features_per = len(hparams.feature_names)
    num_classes=len(hparams.class_names); hparams.num_classes=num_classes
    num_attributes=len(hparams.attribute_names)

    ## VISUALIZE DATA
    hparams.num_features_per = num_features_per
    print('\n*** DATA ***')
    print(f"DECOY MOTION: {hparams.motion}")
    print('xtrain shape:',x_train.shape)
    print('xtest shape:',x_test.shape)
    print('ytrain shape:',y_train.shape)
    print('ytest shape:',y_test.shape)
    print('num attackers:',num_att)
    print('num features:',num_features)
    print('num features per agent:',num_features_per)
    print('feature names:',feature_names)
    print('num attribtues:',num_attributes)
    print('attributes:',hparams.attribute_names)
    print('num classes:',num_classes)
    print('classes:',hparams.class_names)
    print('xtrain sample (1st instance, 1st time step, all features)\n',x_train[0,0,:num_features])
    print('ytrain sample (first instance)',y_train[0])

    ## REDUCE OBSERVATION WINDOW, IF REQUIRED  ***************************************
    if window==-1 or window>time_steps: # if window = -1 (use entire window) or invalid window (too large>min_time): uses entire observation window (min_time) for all runs
        window=time_steps
    hparams.window=window
    x_train=x_train[:,:window]
    x_test=x_test[:,:window]
    print('\n*** REDUCED OBSERVATION WINDOW ***')
    print('time steps available:',time_steps)
    print('window used:',window)
    print('xtrain shape:',x_train.shape)
    print('xtest shape:',x_test.shape)

    ## PLOT QUANTITIES TO RELATE TO CAM (only pass ONE instance VELOCITY feautures)  ***************************************
    if model_type == 'fcn':
        instance=6
        if hparams.features == 'v':
            plotting_function(hparams,x_train[instance])
        elif hparams.features == 'pv':
            plotting_function(hparams, x_train[instance,:,v_idx:])

    ## TIME SERIES PLOTS (VISUALIZE PATTERNS)  ***************************************
    '''
    # Plots all agents' features vs. time window for one class
    # Select sample
    # sample_idx = np.random.randint(0, x_train.shape[0]) # if want a random sample
    sample_idx = 0 # first sample
    sample_data = x_train[sample_idx] # Get data for that sample
    # Plot positions and velocities over time for each agent
    num_subplot=math.ceil(math.sqrt(num_att))
    plt.figure(figsize=(20,20))
    for agent_idx in range(num_att):
        plt.subplot(num_subplot,num_subplot, agent_idx + 1)
        for feature in range(num_features_per):
            plt.plot(sample_data[:, agent_idx+feature*num_att], label=feature_names[feature])
        plt.xlabel('Time Step')
        plt.ylabel('Feature Value [normalized]')
        plt.legend()
        plt.title(f'Agent {agent_idx + 1}')
    plt.savefig(hparams.model_dir + "Agent_feature_plots.png")
    '''
    # Plots one agents' features vs. time window for all classes
    # Find the unique classes and their first index
    unique_classes, unique_indices = np.unique(y_train, return_index=True)
    agent_idx=0
    # Create a subplot for each class to visualize features for Agent_idx
    plt.figure(figsize=(10, 10))  # 2x2 plots
    num_subplot=math.ceil(math.sqrt(num_classes))
    for i, idx in enumerate(unique_indices): #idx=first train instance of each class
        sample_data = x_train[idx]  # Get data for that sample
        print(f"Feature plot {i} {hparams.class_names[i]} = idx {idx}")
        plt.subplot(num_subplot,num_subplot, i + 1) #square matrix of subplots
        for feature in range(num_features_per):
            plt.plot(sample_data[:, agent_idx+feature*num_att], label=feature_names[feature])
        plt.xlabel('Time Step')
        plt.ylabel('Feature Value [normalized]')
        plt.legend()
        plt.title(f'{hparams.class_names[i]}')
    plt.savefig(hparams.model_dir + "Agent_feature_plots_per_class.png")


    ## PCA  ***************************************
    # Reshape data to be 2D: (num_samples * num_timesteps, num_features)
    x_pca = x_train.reshape(-1, num_features)
    # must have label for every timestep (same as "sequence output")
    train_temp=np.zeros((y_train.shape[0], window), dtype=np.int8)
    for c in range(num_classes):
        train_temp[y_train[:,0]==c]=c
    y_pca=train_temp.ravel() # reshape labels to be 1D: (num_samples * num_timesteps,)
    print('\n*** DATA for PCA ***')
    print('x_pca shape:',x_pca.shape)
    print('y_pca shape:',y_pca.shape)

    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(x_pca)

    # Normalize labels to map to the colormap
    norm = plt.Normalize(y_pca.min(), y_pca.max())
    # Create a custom legend
    unique_labels = np.unique(y_pca)
    handles = [Patch(color=plt.cm.jet(norm(label)), label=f"{hparams.class_names[label]}") for label in unique_labels]
    # 2D Scatter plot of the first two principal components
    plt.figure(figsize=(10, 5))
    scatter=plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_pca, cmap='jet', norm=norm, alpha=0.5, marker=".")
    plt.legend(handles=handles, title="Classes")
    plt.title('2D Principle Component Analysis of Input Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(hparams.model_dir + "PCA_12.png")
    # 2D Scatter plot of principal components 1 & 3
    plt.figure(figsize=(10, 5))
    scatter=plt.scatter(pca_result[:, 0], pca_result[:, 2], c=y_pca, cmap='jet', norm=norm, alpha=0.5, marker=".")
    plt.legend(handles=handles, title="Classes")
    plt.title('Principle Component Analysis of Input Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 3')
    plt.savefig(hparams.model_dir + "PCA_13.png")
    # 3D Scatter plot of the first three principal components
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=y_pca, cmap='jet', norm=norm, alpha=0.5, marker=".")
    plt.legend(handles=handles, title="Classes")
    ax.set_title('3D Principle Component Analysis of Input Data')
    plt.savefig(hparams.model_dir + "PCA_3D.png")
    
    # Print explained variance ratio for each component
    explained_variance = pca.explained_variance_ratio_
    print('\n*** Explained Variance by Each PCA Component ***')
    for i, variance in enumerate(explained_variance):
        print(f"Principal Component {i + 1}: {variance:.4f} variance")
    # Get PCA loadings (weight of each input on each PC)
    loadings = pca.components_
    # Heatmap to visualize loadings
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(loadings, annot=True, cmap="coolwarm", xticklabels=False, yticklabels=[f'PC{i+1}' for i in range(loadings.shape[0])])
    # Rotate the text inside the heatmap by 90 degrees
    for text in heatmap.texts:
        text.set_rotation(90)
    # Manually set x-tick labels for grouped features
    group_positions = [5, 15, 25, 35]  # Midpoints of the 10-element groups for labeling
    plt.xticks(group_positions, feature_names, fontsize=10)
    # Add vertical lines to separate groups
    for sep_pos in [10, 20, 30]:
        plt.axvline(x=sep_pos, color='black', linewidth=2)
    plt.title("Each Input's Contribution to Each PCA Dimension")
    plt.xlabel("Original Features")
    plt.ylabel("Principal Components")
    plt.savefig(hparams.model_dir + "PCA_Loading_Heatmap.png")


    ## TEST SET: DETERMINE NUM DATA CLASSES AND NUM RUNS (HELPS SHOW PORTION OF TEST RESULTS AT END)
    num_runs=y_test.shape[0]
    rpc=num_runs//num_classes # runs per class
    print('\n*** START INDEX FOR EACH TEST CLASS ***')
    print('num test classes:',num_classes)
    print('num test runs:',num_runs)
    # ****** added to fix COMB_DN test set results visulization
        # if DN=1:10, then typical test set of 1200 will be repeated 10 times (resulting in 12,000 test instances)
        # setting rpc=300 ensures seeing 5 inference examples from each of 4 classes for DN=1 at end of "class.py"
        # to see other DN (for example DN=8), set cs_idx0 = (DN-1)*1200 below and switch comments below
    # rpc=300  # runs per class; 1200 test set / 4 classes
    # cs_idx0=(8-1)*1200  # set to visualise specific DN (for example DN=8); = (DN-1)*1200
    print('runs per class:',rpc)

    cs_idx=[] # class start index
    for c in range(num_classes):
        cs_idx.append(c*rpc)
        # cs_idx.append(c*rpc+cs_idx0)
    print(f'test set class start indices: {cs_idx}')


    ## IF MULTILABEL (and output length=vector), RESTRUCTURE LABELS
    # check for errors in requested output_length
    if model_type!='lstm' and model_type!='tr': # only 'lstm' and 'tr' can output sequences
        output_length='vec'
    if (output_type=='ml' or output_type=='mh') and output_length!='seq':
        if output_type == 'mh': # multihead must preserve multiclass labels
            y_train_class=y_train
            y_test_class=y_test
        # add column for second label
        y_train=np.insert(y_train,1,0,axis=1)
        y_test=np.insert(y_test,1,0,axis=1)
        # update labels: columns are classes
        # 1st col=Comms(Auction), 2nd col=ProNav
        y_train[y_train[:,0]==0]=[0,0] # Greedy
        y_train[y_train[:,0]==1]=[0,1] # Greedy+
        y_train[y_train[:,0]==2]=[1,0] # Auction
        y_train[y_train[:,0]==3]=[1,1] # Auction+
        y_test[y_test[:,0]==0]=[0,0] # same as above
        y_test[y_test[:,0]==1]=[0,1]
        y_test[y_test[:,0]==2]=[1,0]
        y_test[y_test[:,0]==3]=[1,1]
        print('\n*** MULTILABEL ***')
        print('ytrain shape:',y_train.shape)
        print('ytest shape:',y_test.shape)
        print('ytrain sample (first instance)',y_train[0,:])
        print('ytest sample (first instance)',y_test[0,:])
        if output_type == 'mh':
            print('\n*** MULTIHEAD MULTICLASS ***')
            print('ytrain_class shape:',y_train_class.shape)
            print('ytest_class shape:',y_test_class.shape)
            print('ytrain_class sample (first instance)',y_train_class[0,:])
            print('ytest_class sample (first instance)',y_test_class[0,:])
            y_train=[y_train_class,y_train] # combine class & attribute labels
            y_test=[y_test_class,y_test]
            print('\n*** MULTIHEAD COMBINED LABELS ***')
            print(f'ytrain shape: class {y_train[0].shape} and attr {y_train[1].shape}')
            print(f'ytest shape: class {y_test[0].shape} and attr {y_test[1].shape}')

    ## IF SEQUENCE OUTPUT, RESTRUCTURE LABELS
    if output_length == 'seq':
        if output_type=='mc' or output_type=='mh': # multiclass
            train_temp=np.zeros((y_train.shape[0], window), dtype=np.int8)
            test_temp=np.zeros((y_test.shape[0], window), dtype=np.int8)
            for c in range(num_classes):
                train_temp[y_train[:,0]==c]=c
                test_temp[y_test[:,0]==c]=c
            if output_type == 'mh':
                y_train_class=train_temp
                y_test_class=test_temp
        if output_type=='ml' or output_type=='mh': # multilabel
            train_temp=np.zeros((y_train.shape[0], window, num_attributes), dtype=np.int8)
            test_temp=np.zeros((y_test.shape[0], window, num_attributes), dtype=np.int8)
            train_temp[y_train[:,0]==0]=[0,0] # Greedy
            train_temp[y_train[:,0]==1]=[0,1] # Greedy+
            train_temp[y_train[:,0]==2]=[1,0] # Auction
            train_temp[y_train[:,0]==3]=[1,1] # Auction+
            test_temp[y_test[:,0]==0]=[0,0] # same as above
            test_temp[y_test[:,0]==1]=[0,1]
            test_temp[y_test[:,0]==2]=[1,0]
            test_temp[y_test[:,0]==3]=[1,1]
        y_train=train_temp
        y_test=test_temp
        print('\n*** SEQUENCE OUTPUTS ***')
        print('ytrain shape:',y_train.shape)
        print('ytest shape:',y_test.shape)
        print('ytrain sample (first instance)\n',y_train[0,:])
        print('ytest sample (first instance)\n',y_test[0,:])
        if output_type == 'mh':
            print('\n*** MULTIHEAD MULTICLASS ***')
            print('ytrain_class shape:',y_train_class.shape)
            print('ytest_class shape:',y_test_class.shape)
            print('ytrain_class sample (first instance)',y_train_class[0,:])
            print('ytest_class sample (first instance)',y_test_class[0,:])
            y_train=[y_train_class,y_train] # combine class & attribute labels
            y_test=[y_test_class,y_test]
            print('\n*** MULTIHEAD COMBINED LABELS ***')
            print(f'ytrain shape: class {y_train[0].shape} and attr {y_train[1].shape}')
            print(f'ytest shape: class {y_test[0].shape} and attr {y_test[1].shape}')

    ## RESHAPE DATA IF "FULLY CONNECTED" NN
    if  model_type == 'fc':
        # flatten timeseries data into 1 column per run for fully connected model
        x_train=x_train.reshape(x_train.shape[0],-1) # Reshape to (batch, time*feature)
        x_test=x_test.reshape(x_test.shape[0],-1) # Reshape to (batch, time*feature)
        print('\n*** FULLY CONNECTED DATA ***')
        print('xtrain shape:',x_train.shape)
        print('xtest shape:',x_test.shape)

    ## SIZE MODEL INPUTS AND OUTPUTS
    # inputs
    input_shape=x_train.shape[1:] # time steps and features
    # outputs
    if output_type == 'mc': # multiclass
        output_shape=len(np.unique(y_train)) # number of unique labels (auto flattens to find unique labels)
    elif output_type == 'ml': # multilabel
        output_shape=y_train.shape[-1] # number of binary labels; [-1]=last dimension (works for vector & sequence)
    elif output_type == 'mh': # multihead
        output_shape=[len(np.unique(y_train[0])),y_train[1].shape[-1]]
    print('\n*** SIZE MODEL INPUTS & OUTPUTS ***')
    print('input shape: ', input_shape)
    print('output shape: ', output_shape)

    return x_train, y_train, x_test, y_test, cs_idx, input_shape, output_shape





## CREATE DATASET
def get_dataset(hparams, x_train, y_train, x_test, y_test):

    batch_size = hparams.batch_size
    validation_split=hparams.val_split
    num_val_samples = round(x_train.shape[0]*validation_split)
    print(f'Val Split {validation_split} and # Val Samples {num_val_samples}')

    # Reserve "num_val_samples" for validation
    x_val = x_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]

    if hparams.output_type!='mh':
        y_val = y_train[-num_val_samples:]
        y_train = y_train[:-num_val_samples]
        train_dataset=tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        val_dataset=tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
        test_dataset=tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    else: # multihead classifier (2 outputs)
        y_val=[]
        for head in range(len(y_train)):
            y_val.append(y_train[head][-num_val_samples:])
            y_train[head] = y_train[head][:-num_val_samples]
        train_dataset=tf.data.Dataset.from_tensor_slices((x_train, {'output_class':y_train[0], 'output_attr':y_train[1]})).batch(batch_size)
        val_dataset=tf.data.Dataset.from_tensor_slices((x_val, {'output_class':y_val[0], 'output_attr':y_val[1]})).batch(batch_size)
        test_dataset=tf.data.Dataset.from_tensor_slices((x_test, {'output_class':y_test[0], 'output_attr':y_test[1]})).batch(batch_size)
    
    # removes "auto sharding" warning
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = train_dataset.with_options(options)
    val_dataset = val_dataset.with_options(options)
    test_dataset = test_dataset.with_options(options)

    # autotune prefetching
    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(autotune)
    val_dataset = val_dataset.prefetch(autotune)
    # test_dataset = val_dataset.prefetch(autotune) # NO JOY: added in attempt to alieve WARNING:tensorflow:triggered tf.function retracing.

    # cache and shuffle
    train_dataset = train_dataset.cache().shuffle(train_dataset.cardinality())
    val_dataset = val_dataset.cache().shuffle(val_dataset.cardinality())

    return (train_dataset, val_dataset, test_dataset)





def calculate_rmse(agent_quantities, average_quantities):
    # Calculate RMSE between each agent's quantity [#a,features] and the average swarm quantity [1,f]
    return np.sqrt(np.mean((agent_quantities - average_quantities) ** 2, axis=0))

def plotting_function(hparams, x_data):
    '''Given a single instance (engagement), calculates, stores, and plots several quantities:
    Δ Velocity, Δ Acceleration, RMSE Vel, Accel, and Heading 
    '''

    ## Reshape velocities for one instance assuming the format is [time, agents * features_per]
    velocities = x_data.reshape(hparams.window, hparams.num_att, 2) #[t,#a,VxVy]

    ## Compute Heading Angles
    heading_angles = np.arctan2(velocities[..., 1], velocities[..., 0])  #[t, #a]
    average_heading_angle = np.mean(heading_angles, axis=1)  #[t]

    ## Compute changes (derivatives)
    # Calculate the change in velocity components (slope=accel) between each time step and magnitude
    delta_velocities = np.diff(velocities, axis=0) #[t-1,#a,2]
    delta_velocities_magnitude = np.linalg.norm(delta_velocities, axis=2) #[t-1,#a]
    # Calculate the change in acceleration components (accel slope) between each time step and magnitude
    delta_accelerations = np.diff(delta_velocities, axis=0) #[t-1,#a,2]
    delta_accelerations_magnitude = np.linalg.norm(delta_accelerations, axis=2) #[t-1,#a]
    # Calculate the change in heading components (slope=accel) between each time step and magnitude
    delta_heading = np.diff(heading_angles, axis=0) #[t-1,#a]
    delta_heading_magnitude = np.absolute(delta_heading) #[t-1,#a]
    # Sum the magnitude of the changes for all agents at each time step
    total_delta_velocities_magnitude = np.sum(delta_velocities_magnitude, axis=1) #[t-1]
    total_delta_accelerations_magnitude = np.sum(delta_accelerations_magnitude, axis=1) #[t-1]
    total_delta_heading = np.sum(delta_heading_magnitude, axis=1) #[t-1]
    
    ## Compute RMSE
    velocity_rmse = []
    acceleration_rmse = []
    avg_vel_over_time = []
    avg_accel_over_time = []
    heading_angle_rmse = []
    for t in range(hparams.window):
        # Extract velocities and accelerations at time step t and reshape [num_att, VxVy]
        vel = velocities[t, :].reshape(hparams.num_att, 2)
        # Calculate averages at time t: [1, VxVy]
        average_velocity = np.mean(vel, axis=0, keepdims=True)
        # Compute RMSE: Velocity, Acceleration, Heading
        rmse = calculate_rmse(vel, average_velocity) # [f]
        velocity_rmse.append(np.mean(rmse)) #[1]
        avg_vel_over_time.append(np.mean(average_velocity)) #[1]

        if t < hparams.window - 1: # accel has one less time step
            accel = delta_velocities[t, :].reshape(hparams.num_att, 2)
            average_acceleration = np.mean(accel, axis=0, keepdims=True)
            rmse = calculate_rmse(accel, average_acceleration) # [f]
            acceleration_rmse.append(np.mean(rmse)) #[1]
            avg_accel_over_time.append(np.mean(average_acceleration)) #[1]

        rmse = calculate_rmse(heading_angles[t], average_heading_angle[t]) # [1]
        heading_angle_rmse.append(np.mean(rmse)) #[1]
   
    ## Save variables for post processing
    append_to_csv(total_delta_velocities_magnitude, hparams.model_dir + "variables.csv", "Total Velocity Del Mag Data")
    append_to_csv(total_delta_accelerations_magnitude, hparams.model_dir + "variables.csv", "\nTotal Acceleration Del Mag Data")
    append_to_csv(total_delta_heading, hparams.model_dir + "variables.csv", "\nTotal Heading Del Mag Data")
    append_to_csv(avg_vel_over_time, hparams.model_dir + "variables.csv", "\nAverage Velocity")
    append_to_csv(avg_accel_over_time, hparams.model_dir + "variables.csv", "\nAverage Acceleration")
    append_to_csv(average_heading_angle, hparams.model_dir + "variables.csv", "\nAverage Heading")
    append_to_csv(velocity_rmse, hparams.model_dir + "variables.csv", "\nVelocity RMSE Over Time")
    append_to_csv(acceleration_rmse, hparams.model_dir + "variables.csv", "\nAcceleration RMSE Over Time")
    append_to_csv(heading_angle_rmse, hparams.model_dir + "variables.csv", "\nHeading RMSE Over Time")

    ## PLOTTING
    # Plot each agent Δ VELOCITY magnitude (=acceleration)
    plt.figure(figsize=(10, 6))
    for agent in range(hparams.num_att):
        plt.plot(delta_velocities_magnitude[:, agent], label=f'Agent {agent+1} ΔV')
    plt.xlabel('Time Step')
    plt.ylabel('Δ Velocity Magnitude')
    plt.title('Δ Velocity Magnitude For Each Agent vs. Time')
    # plt.legend()
    plt.savefig(hparams.model_dir + "1Velocity_delta_over_Time.png")
    # Plot TOTAL Δ VELOCITY magnitude (=total accel)
    plt.figure(figsize=(10, 6))
    plt.plot(total_delta_velocities_magnitude, label='Total ΔV', color='black')
    plt.xlabel('Time Step')
    plt.ylabel('Total Δ Velocity Magnitude')
    plt.title('Total Δ Velocity Magnitude vs. Time')
    plt.legend()
    plt.savefig(hparams.model_dir + "1Velocity_delta_Total_over_Time.png")
    # Plot each agent Δ ACCELERATION magnitude
    plt.figure(figsize=(10, 6))
    for agent in range(hparams.num_att):
        plt.plot(delta_accelerations_magnitude[:, agent], label=f'Agent {agent+1} ΔV')
    plt.xlabel('Time Step')
    plt.ylabel('Δ Acceleration Magnitude')
    plt.title('Δ Acceleration Magnitude For Each Agent vs. Time')
    # plt.legend()
    plt.savefig(hparams.model_dir + "1Accel_delta_over_Time.png")
    # Plot TOTAL Δ ACCELERATION magnitude
    plt.figure(figsize=(10, 6))
    plt.plot(total_delta_accelerations_magnitude, label='Total ΔA', color='black')
    plt.xlabel('Time Step')
    plt.ylabel('Total Δ Acceleration Magnitude')
    plt.title('Total Δ Acceleration Magnitude vs. Time')
    plt.legend()
    plt.savefig(hparams.model_dir + "1Accel_delta_Total_over_Time.png")
    # Plot TOTAL Δ HEADING magnitude
    plt.figure(figsize=(10, 6))
    plt.plot(total_delta_heading, label='Total ΔH', color='black')
    plt.xlabel('Time Step')
    plt.ylabel('Total Δ Heading Magnitude')
    plt.title('Total Δ Heading Magnitude vs. Time')
    plt.legend()
    plt.savefig(hparams.model_dir + "1Head_delta_Total_over_Time.png")
    # Velocity RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(velocity_rmse, label='Velocity RMSE')
    plt.xlabel('Time Step')
    plt.ylabel('RMSE')
    plt.title('Velocity RMSE (Agent - Average) vs. Time')
    plt.legend()
    plt.savefig(hparams.model_dir + "1Velocity_RMSE_over_Time.png")
    # Acceleration RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(acceleration_rmse, label='Acceleration RMSE')
    plt.xlabel('Time Step')
    plt.ylabel('RMSE')
    plt.title('Acceleration RMSE (Agent - Average) vs. Time')
    plt.legend()
    plt.savefig(hparams.model_dir + "1Acceleration_RMSE_over_Time.png")
    # Heading RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(heading_angle_rmse, label='Heading RMSE')
    plt.xlabel('Time Step')
    plt.ylabel('RMSE')
    plt.title('Heading RMSE (Agent - Average) vs. Time')
    plt.legend()
    plt.savefig(hparams.model_dir + "1Heading_RMSE_over_Time.png")