#!/bin/bash
#SBATCH --job-name=swarm_DataGenAndNp
#SBATCH --output=/home/donald.peltier/swarm/logs/datagen/decoy_motion/datagen2np_%a.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=beards
#SBATCH --array=1-5

source /etc/profile
source ~/.bashrc
module use /share/modules/base
module load compile/gcc/7.2.0
module load app/matlab/R2023b
module load lang/miniconda3/23.1.0
source activate swarm


# Variables
decoy_motions=("star" "str" "perL" "perR" "semi")
decoy_motion="${decoy_motions[$SLURM_ARRAY_TASK_ID-1]}"

# Run MATLAB script
time matlab -nodisplay -nojvm -nosplash -r "\
datagenarray('$decoy_motion'); exit;"

# Run Python script
python mat2np4.py \
--decoy_motion $decoy_motion
