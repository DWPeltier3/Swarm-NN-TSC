#!/bin/bash
#SBATCH --job-name=swarm-class
#SBATCH --output=/home/donald.peltier/swarm/logs/CNmc20_10v10_star_%j_BNN200k_PredictNEW4.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=beards

# SLURM setup
echo "Current Date and Time: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM GPUs: $SLURM_GPUS_ON_NODE"
echo "SLURM CPUs: $SLURM_CPUS_PER_TASK"

# Environment setup
source /etc/profile
module load lang/miniconda3/23.1.0
source activate swarm

# Define parameters
mode="predict"   # Mode of operation: "train" or "predict"

# TRAINED MODEL: path to the trained model checkpoint (model.ckpt)
    # if Mode=Train and bnn=True, uncomment and include DNN.ckpt for BNN MOPED (else MOPED false)
    # if Mode=Predict, uncomment and include DNN or BNN.ckpt 
# trained_model="/home/donald.peltier/swarm/model/Swarm3/1_Models/01-03_14-37-10_train_BNNf_cnmc20_star_10v10_5kDNN/model.ckpt"
# trained_model="/home/donald.peltier/swarm/model/Swarm3/1_Models/01-16_13-53-41_train_BNNf_cnmc20_star_10v10_40kDNN/model.ckpt"
# trained_model="/home/donald.peltier/swarm/model/Swarm3/CombDM/01-16_14-09-20_train_BNNf_cnmc20_combDM40k_10v10/model.ckpt"
# trained_model="/home/donald.peltier/swarm/model/Swarm3/1_Models/01-16_13-35-36_train_BNNt_cnmc20_star_10v10_5kBNN/model.ckpt"
# trained_model="/home/donald.peltier/swarm/model/Swarm3/1_Models/01-17_09-41-28_train_BNNt_cnmc20_star_10v10_40kBNN/model.ckpt"
trained_model="/home/donald.peltier/swarm/model/Swarm3/CombDM/01-28_13-30-57_train_BNNt_cnmc20_combDM40k_10v10/model.ckpt"

# DATA FOLDER
# data_folder="/home/donald.peltier/swarm/data"                                 # Normal Use: Folder containing the datasets, mat files, and mean/var files
# data_folder="/home/donald.peltier/swarm/data/data_10v10_r4800_c4_a10.npz"     # Special Use: for specific datasets, by name (must change comments datapipeline.py line 39)
# data_folder="/home/donald.peltier/swarm/data/data_10v10_star_r40000.npz"
data_folder="/home/donald.peltier/swarm/data/data_NEW4_10v10_star.npz"
# data_folder="/home/donald.peltier/swarm/data/data_10v10_combDM_r40000.npz"
# data_folder="/home/donald.peltier/swarm/data/data_NEW_10v10_star.npz"
# data_folder="/home/donald.peltier/swarm/data/data_NEW2_10v10_star.npz"
# data_folder="/home/donald.peltier/swarm/data/data_NEW3_10v10_star.npz"

# CLASS NAMES
# class_names="Greedy Greedy+ Auction Auction+"                                             # default; multiclass names separated by a space
# class_names="Greedy Greedy+ Auction Auction+ HVU HVU+ Star Left Down"                     # NEW/3: includes HVU, HVU+, and Attacker motions (start at V=constant); New3 AM initial V=0
# class_names="Greedy Greedy+ Auction Auction+ SK_G SK_G+ SK_A SK_A+ HVU HVU+ Left Down"    # NEW2: adds station keeping (SKv1) to New
class_names="Greedy Greedy+ Auction Auction+ CA_G CA_G+ CA_A CA_A+ Left Down"             # NEW4: v1 (left & down), v3 SK = CA (collision avoidance)

num_att=10                      # Number of attackers
num_def=10                      # Number of defenders: -1 uses "combined decoy number" dataset (1 to 10 decoys) but can be combined with combined motion dataset
motion="star"                   # Defender motion used: ("star, str, perL, perR, semi, comb"); dictionary in "datapipeline" maps motion abbreviation to full motion name; "comb" uses all motions
window=20                       # Window size: -1 uses full window
features="pv"                   # Features to be used: 'pv'=position & velocity, 'p'=position only, 'v'=velocity only
model_type="cn"                 # Type of model: 'fc'=fully connect, 'cn'=CNN, 'fcn'=FCN, 'res'=ResNet, 'lstm'=long short term memory, 'tr'=transformer
bnn=t                           # Bayesian Neural Network Flag: default (False) trains deterministic NN, True=BNN
num_monte_carlo=20              # Bayesian Inference: number of Monte Carlo predictions for entire test set
num_instances_visualize=5       # Number of each class to visualize results for
tuned=t                         # Tuned model used: if 'True', will train using tuned parameters (see "model.py"); "False" for CNNex
output_type="mc"                # Output type: 'mc'=multiclass, 'ml'=multilabel, 'mh'=multihead, which combines mc and ml
output_length="vec"             # Output length: 'vec'=vector (final only), 'seq'=sequence (every time step) **** only "lstm" or "tr" can have "seq" output
dropout=0.1                     # Dropout rate
initial_learning_rate=0.0001    # Initial learning rate
callback_list="checkpoint,early_stopping"  # List of callbacks: "checkpoint,early_stopping" --csv_log is handled separately
patience=500                    # Patience for early stopping (min val loss)
num_epochs=5000                 # Number of epochs; defult 1000
batch_size=50                   # Batch size; defult 50
val_split=0.2                   # Validation split (% of training set; training set is 75% of total set); default 0.2
model_dir="/home/donald.peltier/swarm/model/$(date +%m-%d_%H-%M-%S)_${mode}_BNN${bnn}_${model_type}${output_type}${window}_${motion}_${num_att}v${num_def}/"  # Directory for saving the model dynamically named

# mean_var_NN_DS="/home/donald.peltier/swarm/data/mean_var/mean_var_10v10_r4800_c4_a10.npz"         # Mean/var file path for dataset used to train NN; defaults to "none"
# mean_var_NN_DS="/home/donald.peltier/swarm/data/mean_var/mean_var_10v10_star_r40000.npz"
mean_var_NN_DS="/home/donald.peltier/swarm/data/mean_var/mean_var_10v10_combined_r40000.npz"

# mean_var_current_DS="/home/donald.peltier/swarm/data/mean_var/mean_var_10v10_star_r40000.npz"     # Mean/var file path for current dataset; defaults to "none"
mean_var_current_DS="/home/donald.peltier/swarm/data/mean_var/mean_var_NEW4_10v10_star.npz"
# mean_var_current_DS="/home/donald.peltier/swarm/data/mean_var/mean_var_NEW_10v10_star.npz"
# mean_var_current_DS="/home/donald.peltier/swarm/data/mean_var/mean_var_NEW2_10v10_star.npz"
# mean_var_current_DS="/home/donald.peltier/swarm/data/mean_var/mean_var_NEW3_10v10_star.npz"

# Conditionally include arguments (if you comment out lines above it will use defaults defined in "params.py")
args=""
[ ! -z "$mode" ] && args+="--mode=$mode "
[ ! -z "$tuned" ] && args+="--tuned=$tuned "
[ ! -z "$trained_model" ] && args+="--trained_model=$trained_model "
[ ! -z "$data_folder" ] && args+="--data_folder=$data_folder "
[ ! -z "$class_names" ] && args+="--class_names $class_names "
[ ! -z "$num_att" ] && args+="--num_att=$num_att "
[ ! -z "$num_def" ] && args+="--num_def=$num_def "
[ ! -z "$motion" ] && args+="--motion=$motion "
[ ! -z "$window" ] && args+="--window=$window "
[ ! -z "$features" ] && args+="--features=$features "
[ ! -z "$model_type" ] && args+="--model_type=$model_type "
[ ! -z "$bnn" ] && args+="--bnn=$bnn "
[ ! -z "$num_monte_carlo" ] && args+="--num_monte_carlo=$num_monte_carlo "
[ ! -z "$num_instances_visualize" ] && args+="--num_instances_visualize=$num_instances_visualize "
[ ! -z "$output_type" ] && args+="--output_type=$output_type "
[ ! -z "$output_length" ] && args+="--output_length=$output_length "
[ ! -z "$dropout" ] && args+="--dropout=$dropout "
[ ! -z "$initial_learning_rate" ] && args+="--initial_learning_rate=$initial_learning_rate "
[ ! -z "$callback_list" ] && args+="--callback_list=$callback_list "
[ ! -z "$patience" ] && args+="--patience=$patience "
[ ! -z "$num_epochs" ] && args+="--num_epochs=$num_epochs "
[ ! -z "$batch_size" ] && args+="--batch_size=$batch_size "
[ ! -z "$val_split" ] && args+="--val_split=$val_split "
[ ! -z "$model_dir" ] && args+="--model_dir=$model_dir "
[ ! -z "$mean_var_NN_DS" ] && args+="--mean_var_NN_DS=$mean_var_NN_DS "
[ ! -z "$mean_var_current_DS" ] && args+="--mean_var_current_DS=$mean_var_current_DS "

# Execute the command
srun python class.py $args