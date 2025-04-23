%% Collect & Save Data
% Use this script to run 4 algorithms below using inputs
% specified "seedrange" number of times and collect defender 
% data (position, velocity)

clc; clear all; close all;

%% Number of runs per class & save flag
seedrange           = 1200;     % # of samples (runs) to collect per algorithm
savemat             = 1;        % T/F: "true" will save all runs into a MATLAB .mat files

%% COMPUTATION SYSTEM
comp_sys            = "h";      % "m" = mac, "h" = hamming: sets folder paths and video setting

%% ENGAGEMENT SIM PARAMETERS       ***************************************
Att_num             = 10;       % number of attackers (agents that kill)
Def_num             = 10;       % number of defenders (agents that can NOT kill)
Att_vel_max         = 1;        % attacker max velocity constraint
Def_vel_max         = 0.4;      % defender max velocity constraint
Def_vel_min         = 0.0001;   % defender min velocity; non-zero prevents dividing by zero when calculating acceleration
Att_accel_steps     = 10;       % attacker acceleration steps
Def_accel_flag      = 0;        % T/F: include defender acceleration during initial trajectory generation
Def_accel_steps     = 10;       % defender acceleration steps
Att_kill_range      = 1;        % attacker weapons range
Def_motion          = "star";   % defender motion profile: star, str, perL, perR, semi, zero
do_plot_flag        = 0;        % T/F: plot initial trajectories during datagen
save_movie_flag     = 0;        % T/F: save initial trajectories moving plot during datagen
Def_final_fraction  = 0;        % final proportion of defenders remaining
seed                = 1205;     % random seed (for reproducability): use seed>1200 (so not part of train/test sets)
Att_obs_dist        = 5;        % for observation only: observe distance
repulsion_strength  = 0.2;      % for observation only: repel strength; effects min distance when closing fast

%% RESULTS & PLOTTING       **********************************************
results_name    = sprintf('Broad_%dv%d_%s',...
                  Att_num, Def_num, Def_motion)

if comp_sys     == "m"      % Macbook
    results     =  "/Users/DWPeltier3/Library/CloudStorage/OneDrive-NavalPostgraduateSchool/1Documents/Research/swarm/Matlab/results";
elseif comp_sys == "h"      % Hamming
    results     =  "/home/donald.peltier/swarm/model/matlab/results";
end

results_folder  = fullfile(results, results_name);
if ~exist(results_folder, 'dir')                % Check if the results folder exists
    mkdir(results_folder);                      % If the folder does not exist, create it
end

%% STORE VARIABLES FOR FUNCTION PASSING       ****************************
info.comp_sys           = comp_sys;
info.Att_num            = Att_num;
info.Def_num            = Def_num;
info.Att_vel_max        = Att_vel_max;
info.Def_vel_max        = Def_vel_max;
info.Def_vel_min        = Def_vel_min;
info.Att_accel_steps    = Att_accel_steps;
info.Def_accel_flag     = Def_accel_flag;
info.Def_accel_steps    = Def_accel_steps;
info.Att_kill_range     = Att_kill_range;
info.Def_motion         = Def_motion;
info.do_plot_flag       = do_plot_flag;
info.save_movie_flag    = save_movie_flag;
info.Def_final_fraction = Def_final_fraction;
info.seed               = seed;
info.results_folder     = results_folder;
info.Att_obs_dist       = Att_obs_dist;  % New variable for safe distance
info.repulsion_strength = repulsion_strength;  % New variable for repulsion strength

%% TEST FUNCTIONS
% HVU_G_A_PV(info);
% AP_A_PV(info);
% OBS_AP_A_PV(info);

%% Save Data Flag and File Name
savemat=true; %"true" will save all runs into a MATLAB .mat files

%% Run simulation multiple times and gather training data and labels
% init cells to hold data for each class
data_obsap={};
data_hvug={};
data_ap={};

% run simulations and record data
for seed=1:seedrange
    
    info.seed = seed;

    %% Observe Only Auction ProNav
    obsap=OBS_AP_A_PV(info);
    %% HVU Only GREEDY (Pursuit)
    hvug=HVU_G_A_PV(info);
    %% Attack All Friendlies Auction ProNav
    ap=AP_A_PV(info);

    %% Append run data to data matricies
    data_obsap=cat(1,data_obsap,{obsap});
    data_hvug=cat(1,data_hvug,{hvug});
    data_ap=cat(1,data_ap,{ap});

end

%% Save data matricies
if savemat
    if comp_sys     == "m"      % Macbook
        mat_folder     =  "/Users/DWPeltier3/Library/CloudStorage/OneDrive-NavalPostgraduateSchool/1Documents/Research/swarm/data/mat_files";
    elseif comp_sys == "h"      % Hamming
        mat_folder     =  "/home/donald.peltier/swarm/data/mat_files";
    end
    % Specify .mat files folder path
    mat_path  = fullfile(mat_folder, results_name);
    % Check if the folder exists, if not, create it
    if ~exist(mat_path, 'dir')
        mkdir(mat_path);
    end
    % Save each class' data with name 'data' for ease of Numpy conversion
    data = data_obsap;
    save(fullfile(mat_path, 'data_obsap.mat'), 'data')
    data = data_hvug;
    save(fullfile(mat_path, 'data_hvug.mat'), 'data')
    data = data_ap;
    save(fullfile(mat_path, 'data_ap.mat'), 'data')
end
%{%}