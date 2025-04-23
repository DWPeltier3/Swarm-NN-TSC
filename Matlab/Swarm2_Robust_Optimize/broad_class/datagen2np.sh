#!/bin/bash
#SBATCH --job-name=swarm_BROADdatagen2np
#SBATCH --output=/home/donald.peltier/swarm/logs/datagen/BROADdatagen2np_%j.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=beards

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
mat_list=("data_obsap.mat" "data_hvug.mat" "data_ap.mat")

# Using time command to measure the execution time for MATLAB script
time matlab -nodisplay -nojvm -nosplash -r "BROADdatagen; exit;"

# Running the Python script with properly formatted mat_list
python BROADmat2np.py \
--decoy_motion "$decoy_motion" \
--num_att "$num_att" \
--num_def "$num_def" \
--mat_list "${mat_list[@]}"