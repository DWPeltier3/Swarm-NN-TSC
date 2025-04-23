#!/bin/bash
#SBATCH --job-name=swarm_DataGenAndNp
#SBATCH --output=/home/donald.peltier/swarm/logs/datagen/decoy_number/datagen2np_perR_10v%a.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=beards
#SBATCH --array=1-9

source /etc/profile
source ~/.bashrc
module use /share/modules/base
module load compile/gcc/7.2.0
module load app/matlab/R2023b
module load lang/miniconda3/23.1.0
source activate swarm

# Variables
decoy_motion="perR"
num_att=10
num_defs=({1..9}) # add at top: --array=1-9 and datagen2np_%a.txt
num_def=${num_defs[$SLURM_ARRAY_TASK_ID-1]} 
# num_def=10

# Using time command to measure the execution time for MATLAB script
time matlab -nodisplay -nojvm -nosplash -r "\
datagenarray($num_att, $num_def, '$decoy_motion'); exit;"

# Run the Python script
python mat2np4.py \
--decoy_motion $decoy_motion \
--num_att $num_att \
--num_def $num_def
