#!/bin/bash
#SBATCH --job-name=swarm-class
#SBATCH --output=/home/donald.peltier/swarm/logs/historical/noise/compare/predict_CNmhFULL_noise_0_combscaled/predict_noise_%a.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=beards
#SBATCH --array=0-50

# Setup Environment
source /etc/profile
module load lang/miniconda3/23.1.0
module load app/graphviz/8.0.5
source activate swarm

## DESCRIPTION: runs engagement for changing noise levels
# ensure #SBATCH --array=0-50 (lower and upper noise levels)

# Define parameters
mode="predict"                  # Mode of operation: "train" or "predict"
trained_model="/home/donald.peltier/swarm/model/historical/noise/compare/08-15_14-29-47_train_cnmh-1_star_10v10_noise0_combscaled/model.keras"  # Path to the trained model
# Special Use: for specific datasets, by name (must change comments datapipeline.py line 39)
data_folder="/home/donald.peltier/swarm/data/historical/noise/Noise1/data_10v10_r4800s_4cl_a10_noise${SLURM_ARRAY_TASK_ID}.npz"  # Folder containing the noise datasets
num_att=10                      # Number of attackers
num_def=10                     # Number of defenders: -1 uses "combined decoy number" dataset (1 to 10 decoys) but can be combined with combined motion dataset
motion="star"                   # Decoy motion used: ("star, str, per, split, semi, comb"); dictionary in "datapipeline" maps motion abbreviation to full motion name; "comb" uses all motions
window=-1                       # Window size: -1 uses full window
features="pv"                   # Features to be used: 'pv'=position & velocity, 'p'=position only, 'v'=velocity only
model_type="cn"                 # Type of model: 'fc'=fully connect, 'cn'=CNN, 'fcn'=FCN, 'res'=ResNet, 'lstm'=long short term memory, 'tr'=transformer
tuned=f                         # Tuned model used: if 'True', will train using tuned parameters (see "model.py"); "False" for CNNex
output_type="mh"                # Output type: 'mc'=multiclass, 'ml'=multilabel, 'mh'=multihead, which combines mc and ml
output_length="vec"             # Output length: 'vec'=vector (final only), 'seq'=sequence (every time step) **** only "lstm" or "tr" can have "seq" output
dropout=0.1                     # Dropout rate
kernel_initializer="he_normal"  # Kernel initializer: "glorot_uniform/normal" "he_normal"
kernel_regularizer="none"       # Kernel regularizer: "none" "l1" "l2" "l1_l2"
optimizer="adam"                # Optimizer to be used
initial_learning_rate=0.0001    # Initial learning rate
callback_list="checkpoint,early_stopping,csv_log"  # List of callbacks: "checkpoint,early_stopping,csv_log"
patience=50                     # Patience for early stopping (min val loss)
num_epochs=1000                 # Number of epochs
batch_size=50                   # Batch size
val_split=0.2                   # Validation split (% of training set; training set is 75% of total set)
# tune_type="r"                   # only used for tuning
# tune_epochs=1000                # only used for tuning
model_dir="/home/donald.peltier/swarm/model/historical/noise/compare/predict_CNmhFULL_noise_0_combscaled/${mode}_noise_${SLURM_ARRAY_TASK_ID}/"  # Directory for saving the model dynamically named
mean_var_file="/home/donald.peltier/swarm/data/mean_var/mean_var_10v10_r4800_c4_a10_combined_noise.npz"  # Mean/var file path for rescaling the dataset; defaults to "none"

# Conditionally include arguments (if you comment out lines above it will use defaults defined in "params.py")
args=""
[ ! -z "$mode" ] && args+="--mode=$mode "
[ ! -z "$tuned" ] && args+="--tuned=$tuned "
[ ! -z "$trained_model" ] && args+="--trained_model=$trained_model "
[ ! -z "$data_folder" ] && args+="--data_folder=$data_folder "
[ ! -z "$num_att" ] && args+="--num_att=$num_att "
[ ! -z "$num_def" ] && args+="--num_def=$num_def "
[ ! -z "$motion" ] && args+="--motion=$motion "
[ ! -z "$window" ] && args+="--window=$window "
[ ! -z "$features" ] && args+="--features=$features "
[ ! -z "$model_type" ] && args+="--model_type=$model_type "
[ ! -z "$output_type" ] && args+="--output_type=$output_type "
[ ! -z "$output_length" ] && args+="--output_length=$output_length "
[ ! -z "$dropout" ] && args+="--dropout=$dropout "
[ ! -z "$kernel_initializer" ] && args+="--kernel_initializer=$kernel_initializer "
[ ! -z "$kernel_regularizer" ] && args+="--kernel_regularizer=$kernel_regularizer "
[ ! -z "$optimizer" ] && args+="--optimizer=$optimizer "
[ ! -z "$initial_learning_rate" ] && args+="--initial_learning_rate=$initial_learning_rate "
[ ! -z "$callback_list" ] && args+="--callback_list=$callback_list "
[ ! -z "$patience" ] && args+="--patience=$patience "
[ ! -z "$num_epochs" ] && args+="--num_epochs=$num_epochs "
[ ! -z "$batch_size" ] && args+="--batch_size=$batch_size "
[ ! -z "$val_split" ] && args+="--val_split=$val_split "
[ ! -z "$tune_type" ] && args+="--val_split=$val_split "
[ ! -z "$tune_epochs" ] && args+="--val_split=$val_split "
[ ! -z "$model_dir" ] && args+="--model_dir=$model_dir "
[ ! -z "$mean_var_file" ] && args+="--mean_var_file=$mean_var_file "

# Execute the command
python class.py $args