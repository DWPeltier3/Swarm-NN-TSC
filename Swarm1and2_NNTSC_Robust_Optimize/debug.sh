#!/bin/bash
#SBATCH --job-name=swarm_class
#SBATCH --output=../logs/debug/swarm-cfc-debug%j.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/23.1.0
module load app/graphviz/8.0.5

source activate swarm

python -m debugpy --wait-for-client --listen 0.0.0.0:54321 \
class_tune.py \
--mode="train" \
--trained_model="/home/donald.peltier/swarm/model/swarm_class09-08_14-30/model.keras" \
--model_dir="/home/donald.peltier/swarm/model/swarm_class$(date +%m-%d_%H-%M)/" \
--data_path="/home/donald.peltier/swarm/data/data_10v10_r4800s_4cl_a10.npz" \
--window=20 \
--model_type="fc" \
--output_type="mc" \
--output_length="vec" \
--dropout=0.2 \
--kernel_initializer="he_normal" \
--kernel_regularizer="none" \
--optimizer="adam" \
--initial_learning_rate=0.0001 \
--callback_list="checkpoint, early_stopping, csv_log" \
--patience=50 \
--num_epochs=100 \
--batch_size=50 \
--train_val_split=0.2 \

## NOTES
# mode = 'train' or 'predict'
# window = -1 uses full window
# model_type = 'fc'=fully connect, 'cn'=CNN, 'fcn'=FCN, 'res'=ResNet, 'lstm'=long short term memory, 'tr'=transformer
# output_type = 'mc'=multiclass, 'ml'=multilabel, 'mh'=multihead
# output_length = 'vec'=vector (final only), 'seq'=sequence (every time step) **
#                                              ** only "lstm" or "tr" can have "seq" output
# kernel_initializer = "glorot_normal" "he_uniform/normal"
# kernel_regularizer = "none" "l1" "l2" "l1_l2"