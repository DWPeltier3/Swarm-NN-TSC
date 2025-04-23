#!/bin/bash
#SBATCH --job-name=swarm_savedata
#SBATCH --output=/home/donald.peltier/swarm/logs/swarm-NEWdatagen_SK3.txt
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

# RUN MATLAB SCRIPT
# Using time command to measure the execution time
# if using plot function, do NOT include "-nojvm" flag
time matlab -nodisplay -nojvm -nosplash -r "NEWdatagen"