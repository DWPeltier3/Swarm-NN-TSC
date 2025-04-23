#!/bin/bash
#SBATCH --job-name=swarm-class
#SBATCH --output=/home/donald.peltier/swarm/logs/decoy_motion/cnEXmc20_scaled10v10comb/predict/predict_comb%a.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=beards
#SBATCH --array=1-6

# Setup Environment
. /etc/profile
module load lang/miniconda3/23.1.0
module load app/graphviz/8.0.5
source activate swarm

## DESCRIPTION: runs engagement for changing variables
# ensure #SBATCH --array=1-"count of changing variable entries"

# Variables
motions=("star" "semi" "str" "perL" "perR" "comb")
motion=${motions[$SLURM_ARRAY_TASK_ID-1]}

# Define parameters
mode="predict"                  # Mode of operation: "train" or "predict"
trained_model="/home/donald.peltier/swarm/model/decoy_motion/cnEXmc20_10v10_scaled10v10comb/08-12_15-57-59_train_comb/model.keras"  # Path to the trained model
data_folder="/home/donald.peltier/swarm/data"  # Folder containing the datasets, mat files, and mean/var files
num_att=10                      # Number of attackers
num_def=10                      # Number of defenders: -1 uses "combined decoy number" dataset (1 to 10 decoys)
# motion="star"                   # Decoy motion used: ("star, str, per, split, semi, comb"); dictionary in "datapipeline" maps motion abbreviation to full motion name; "comb" uses all motions
window=20                       # Window size: -1 uses full window
features="pv"                   # Features to be used: 'pv'=position & velocity, 'p'=position only, 'v'=velocity only
model_type="cn"                 # Type of model: 'fc'=fully connect, 'cn'=CNN, 'fcn'=FCN, 'res'=ResNet, 'lstm'=long short term memory, 'tr'=transformer
tuned=f                         # Tuned model used: if 'True', will train using tuned parameters (see "model.py")
output_type="mc"                # Output type: 'mc'=multiclass, 'ml'=multilabel, 'mh'=multihead, which combines mc and ml
output_length="vec"             # Output length: 'vec'=vector (final only), 'seq'=sequence (every time step) **** only "lstm" or "tr" can have "seq" output
dropout=0.1                     # Dropout rate
kernel_initializer="he_normal"  # Kernel initializer: "glorot_uniform/normal" "he_normal"
kernel_regularizer="none"       # Kernel regularizer: "none" "l1" "l2" "l1_l2"
optimizer="adam"                # Optimizer to be used
initial_learning_rate=0.0001    # Initial learning rate
callback_list="checkpoint,early_stopping,csv_log"  # List of callbacks: "checkpoint, early_stopping, csv_log"
patience=50                     # Patience for early stopping
num_epochs=1000                 # Number of epochs
batch_size=50                   # Batch size
val_split=0.2                   # Validation split (% of training set; training set is 75% of total set)
# model_dir="/home/donald.peltier/swarm/model/decoy_motion/cnmc20_scaled10v10comb/$(date +%m-%d_%H-%M-%S)_${mode}_${model_type}${output_type}${window}_${motion}_${num_att}v${num_def}/"  # Directory for saving the model dynamically named
model_dir="/home/donald.peltier/swarm/model/decoy_motion/cnEXmc20_10v10_scaled10v10comb/$(date +%m-%d_%H-%M-%S)_${mode}_comb_${motion}/"  # Directory for saving the model dynamically named
mean_var_file="/home/donald.peltier/swarm/data/mean_var/mean_var_10v10_combined.npz"  # Mean/var file path for rescaling the dataset; defaults to "none"

# Conditionally include arguments (if comment out lines above will use defaults defined in "params.py")
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
[ ! -z "$model_dir" ] && args+="--model_dir=$model_dir "
[ ! -z "$mean_var_file" ] && args+="--mean_var_file=$mean_var_file "

# Execute the command
python class.py $args