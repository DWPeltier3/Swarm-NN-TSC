#!/bin/bash
#SBATCH --job-name=odm
#SBATCH --output=/home/donald.peltier/swarm/logs/odm_10002_CNmc20_10v1C10_combDM_semi10v8_IPcontSQP_AccelON.txt
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

# Parameters
COMP_SYS="h"        # computer system: h = hamming, m = macbook

DEF_NUM=8;         # number of defenders (agents that can NOT kill)
DEF_ACCEL_FLAG=1    # 0 or 1: include defender acceleration during initial trajectory generation
DEF_Spread=0.0001   # defender initial dispersion distance about starting point (default=5) 0.0001

BEST_IT=0           # 0 or 1: whether to use best initial trajectory (1) or defined Def_motion (0)
DEF_MOTION="semi"   # defender motion: star, str, perL, perR, semi, zero
SEED=10002           # random seed (for reproducibility): use seed > 1200 (so not part of train/test sets), 1202, 10002
MODELNAME="CNmc20_10v1C10_combDM"  # model names: CNmc20_10v1C10_combDM, CNmc20_10v1C10_perL, CNmc20, CNmc20_10v1C10_semi, CNmc20_r40k, CNmcFull, LSTMMmc20, LSTMMmcFull

OPT_OBJ="y"         # optimization objective: y = max true_pred_all, t =  min tf, p = max probmin, o = max 1class prob
TRAJ_APPROX="s"     # trajectory approximation: "s" = spline, "b" = Bernstein
FMIN_ALGO="interior-point"     # algorithm used by fmincon: 'interior-point', 'sqp'
PolyCon="[-1 -1; -1 5; 1 6; 6 1; 6 -1; -1 -1]"            # QUAD DIAMOND
# PolyCon="[-1 -1; -1 5; 5 8; 5 1; -1 -1]"                  # Quad

scale_accel=1;      # acceleration constraint scale factor
scale_pos=10;       # position constraint scale factor

# path to .mat file from previous run (x = x0) for continuation
load_previous="/home/donald.peltier/swarm/model/matlab/results/10002_CNmc20_10v1C10_combDM_10v8_Best0_semi_Algo_sqp_Traj_s_Obj_y_2024-10-31_13-44/10002_CNmc20_10v1C10_combDM_10v8_Best0_semi_Algo_sqp_Traj_s_Obj_y_2024-10-31_13-44.mat"
# load_previous=""   # pass empty string to turn off continuation
# /home/donald.peltier/swarm/model/matlab/results/STPvsDN_DM/combDM_NN/predict_semi/odm_10002_CNmc20_10v1C10_combDM_semi10v10_IP2.txt

# RUN MATLAB SCRIPT
# Using time command to measure the execution time
# do NOT include "-nojvm" flag if using plot function
# addpath to "keras_importer" if loading NN converted from TensorFlow
time matlab -nodisplay -nosplash -r "\
addpath('/home/donald.peltier/swarm/code/matlab/keras_importer'); \
odm('$COMP_SYS', $DEF_NUM, $DEF_ACCEL_FLAG, $DEF_Spread, \
$BEST_IT, '$DEF_MOTION', $SEED, '$MODELNAME', \
'$OPT_OBJ', '$TRAJ_APPROX', '$FMIN_ALGO', \
$PolyCon, $scale_accel, $scale_pos, \
'$load_previous');
exit;"