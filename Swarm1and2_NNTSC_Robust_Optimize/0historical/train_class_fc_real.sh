#!/bin/bash
#SBATCH --job-name=swarm_class
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/donald.peltier/swarm/logs/swarm-cfc%j.txt
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/4.10.3

source activate cs4321_peltier

python class_fc_real.py \
--model_dir="/home/donald.peltier/swarm/model/swarm_cfc_$(date +%Y-%m-%d_%H-%M-%S)/" \
--num_epochs=1000 \
--batch_size=25 \
--data_path="/home/donald.peltier/swarm/data/data_7v10_r4800s_4cl.npz" \
--real_data_path="/home/donald.peltier/swarm/data/data_7v10_r4s_nps.npz" \
--window=-1 # -1 uses full window