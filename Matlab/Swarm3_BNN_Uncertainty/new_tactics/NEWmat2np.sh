#!/bin/bash
#SBATCH --job-name=swarm_mat2np
#SBATCH --output=/home/donald.peltier/swarm/logs/NEW4_mat2np_%j.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
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
source ~/.bashrc
module use /share/modules/base
module load compile/gcc/7.2.0
module load app/matlab/R2023b
module load lang/miniconda3/23.1.0
source activate swarm

# Variables
decoy_motion="star"
num_att=10
num_def=10

# List of .mat files to combine
# mat_list=("data_g.mat" "data_gp.mat" "data_a.mat" "data_ap.mat" "data_hvu.mat" "data_hvup.mat" "data_am_star.mat" "data_am_left.mat" "data_am_down.mat")
# mat_list=("data_g.mat" "data_gp.mat" "data_a.mat" "data_ap.mat" "data_hvu.mat" "data_hvup.mat" "data_am_star2.mat" "data_am_left2.mat" "data_am_down2.mat")
# mat_list=("data_g.mat" "data_gp.mat" "data_a.mat" "data_ap.mat" "data_skg.mat" "data_skgp.mat" "data_ska.mat" "data_skap.mat" "data_hvu.mat" "data_hvup.mat" "data_am_left.mat" "data_am_down.mat")
mat_list=("data_g.mat" "data_gp.mat" "data_a.mat" "data_ap.mat" "data_skg3.mat" "data_skgp3.mat" "data_ska3.mat" "data_skap3.mat" "data_am_left.mat" "data_am_down.mat")
# mat_list=("data_skg.mat" "data_skgp.mat" "data_ska.mat" "data_skap.mat")
# mat_list=("data_skg2.mat" "data_skgp2.mat" "data_ska2.mat" "data_skap2.mat")
# mat_list=("data_skg3.mat" "data_skgp3.mat" "data_ska3.mat" "data_skap3.mat")

# Using time command to measure the execution time for MATLAB script
# time matlab -nodisplay -nojvm -nosplash -r "NEWdatagen; exit;"

# Running the Python script with properly formatted mat_list
python NEWmat2np.py \
--decoy_motion "$decoy_motion" \
--num_att "$num_att" \
--num_def "$num_def" \
--mat_list "${mat_list[@]}"