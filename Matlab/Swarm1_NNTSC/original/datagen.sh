#!/bin/bash
#SBATCH --job-name=swarm_savedata
#SBATCH --output=/home/donald.peltier/swarm/logs/swarm-datagen.txt
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
matlab -nodisplay -nojvm -nosplash -r "datagen"