#!/bin/bash
#SBATCH --job-name=swarm-extract_data
#SBATCH --output=/home/donald.peltier/swarm/logs/extracted_data/extract.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00

# Setup Environment
sourc  /etc/profile
module load lang/miniconda3/23.1.0
source activate swarm

# Run program
python extractnoise.py \
