%% Function caller used to run odm.m function

clc; clear all; close all;

% # Parameters
COMP_SYS        ="m";           % computer system: h = hamming, m = macbook

DEF_NUM         =9;            % number of defenders (agents that can NOT kill)
DEF_ACCEL_FLAG  =0;             % 0 or 1: include defender acceleration during initial trajectory generation
% DEF_ACCEL_STEPS =10;            % defender steps to reach max velocity; 1-10 (less=more accel)
Def_spread      =0.0001;             % defender initial dispersion distance about starting point (default=5) 0.0001

BEST_IT         =1;             % 0 or 1: whether to use best initial trajectory (1) or defined Def_motion (0)
DEF_MOTION      ="semi";        % defender motion: star, semi, str, perL, perR, zero
SEED            =10002;          % random seed (for reproducibility): use seed > 1200 (so not part of train/test sets)
MODELNAME       ="CNmc20_10v1C10_combDM";      % model names: CNmc20, CNmc20_10v1C10_perL, CNmc20_10v1C10_combDM, CNmc20_10v1C10_semi, CNmc20_r40k, CNmcFull, LSTMMmc20, LSTMMmcFull

OPT_OBJ         ="y";           % optimization objective: y = max true_pred_all, t =  min tf, p = max probmin, o = max 1class prob
TRAJ_APPROX     ="s";           % trajectory approximation: "s" = spline, "b" = Bernstein
FMIN_ALGO       = 'sqp';        % algorithm used by fmincon: 'interior-point', 'sqp'
% PolyCon         =[-2 0; -2 4; 5 5; 5 1; 4 -1; 2 -3; 0 -3; -2 0];  % HVU20 1201
% PolyCon         =[-1 -2; -1 12; 6 15; 12 15; 12 2; -1 -2];        % HVU-1 1201
% PolyCon         =[-1 -1; -1 5; 5 8; 5 1; -1 -1];                  % HVU20 1202
PolyCon         =[-1 -1; -1 5; 1 6; 6 1; 6 -1; -1 -1];                  % QUAD DIAMOND
% PolyCon         =[-1 -2; -1 12; 6 15; 12 15; 12 2; -1 -2];        % HVU-1 1202
% PolyCon         =[-2 -1; -2 4; 6 5; 6 1; 4 -1; -2 -1];            % HVU20 1203
% PolyCon         =[-3 -2; -3 10; 6 15; 12 15; 12 2; 9 -2; -3 -2];  % HVU-1 1203
% PolyCon         =[-2 -1; -2 4; 6 5; 6 1; 4 -1; -2 -1];            % HVU20 1204
% PolyCon         =[-1 -2; -1 8; 6 10; 12 10; 12 5; 6 0; -1 -2];    % HVU-1 1204
% PolyCon         =[-2 -1; -2 2; 0 2; 1 4; 6 5; 4 -1; -2 -1];       % HVU20 1205
% PolyCon         =[-3 -2; -3 10; 6 15; 12 15; 12 2; 9 -2; -3 -2];  % HVU-1 1205

% PolyCon         =[1 -1; -1 -1; -1 1; 5 7; 7 5; 1 -1];             % HVU20 NE
% PolyCon         =[1 -1; -1 -1; -7 5; -5 7; 1 1; 1 -1];            % HVU20 NW
% PolyCon         =[1 -1; -5 -7; -7 -5; -1 1; 1 1; 1 -1];           % HVU20 SW
% PolyCon         =[-1 -1; -1 1; 1 1; 7 -5; 5 -7; -1 -1];            % HVU20 SE

scale_accel     = 1;            % acceleration constraint scale factor
scale_pos       = 10;           % position constraint scale factor

% load_previous   = "/Users/DWPeltier3/Library/CloudStorage/OneDrive-NavalPostgraduateSchool/1Documents/Research/swarm/Matlab/results/1202_CNmc20_10v10_Best1_star_Accel10v11_0_Traj_s_Obj_y_PxC_8_As_1_Ps_10/1202_CNmc20_10v10_Best1_star_Accel10v11_0_Traj_s_Obj_y_PxC_8_As_1_Ps_10.mat";
load_previous   = "";

odm(COMP_SYS, DEF_NUM, DEF_ACCEL_FLAG, Def_spread,...
    BEST_IT, DEF_MOTION, SEED, MODELNAME,...
    OPT_OBJ, TRAJ_APPROX, FMIN_ALGO,...
    PolyCon, scale_accel, scale_pos,...
    load_previous);