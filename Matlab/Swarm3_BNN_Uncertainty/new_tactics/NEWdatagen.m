%% Collect & Save Data
% Use this script to run algorithms below using inputs
% specified and collect data (position, velocity) for agents 
% being classified

clc; clear all; close all;

%% Number of runs per class & save flag
seedstart           = 1;        % default=1; can adjust to only capture test set
seedend             = 10000;    % # of samples (runs) to collect per algorithm
savemat             = 1;        % T/F: "true" will save all runs into a MATLAB .mat files

%% COMPUTATION SYSTEM
comp_sys            = "h";      % "m" = mac, "h" = hamming: sets folder paths and video settings

%% ENGAGEMENT SIM PARAMETERS       ***************************************
Att_num             = 10;       % number of attackers (agents that kill)
Def_num             = 10;       % number of defenders (agents that can NOT kill)
Att_vel_max         = 1;        % attacker max velocity constraint
Def_vel_max         = 0.4;      % defender max velocity constraint
Def_vel_min         = 0.05;     % defender min velocity; 0.05 & 0.0001; non-zero prevents dividing by zero when calculating acceleration
Att_accel_steps     = 10;       % attacker acceleration steps
Def_accel_flag      = 0;        % T/F: include defender acceleration during initial trajectory generation
Def_accel_steps     = 10;       % defender acceleration steps
Att_kill_range      = 1;        % NOT USED: attacker weapons range
Def_motion          = "star";   % defender motion: star, str, perL, perR, semi, zero
Att_motion          = "star";   % attacker motion: star, str, left, down, perL, perR, semi, zero
do_plot_flag        = 0;        % T/F: plot initial trajectories during datagen (required and auto added for "save_movie_flag")
save_movie_flag     = 0;        % T/F: save initial trajectories moving plot during datagen
Def_final_fraction  = 0;        % final proportion of defenders remaining
test_seed           = 7501;     % random seed (for reproducability): use seed>1200 (so not part of train/test sets)
sim_time_steps      = 20;       % max number of simulation time steps
Att_obs_dist        = 10;       % for observation only: observe distance
repulsion_strength  = 0.1;      % for observation only: repel strength; effects min distance when closing fast
min_dist            = 1;        % minimum allowable distance between attackers
repel_gain          = 0.5;        % weight for repulsion (0-1; 0=no collision avoidance, 1=full unit vecotr, >1=more important than targeting)

%% RESULTS & PLOTTING       **********************************************
results_name    = sprintf('NEW_%dv%d_%s',...
                  Att_num, Def_num, Def_motion);

if comp_sys     == "m"      % Macbook
    results     =  "/Users/DWPeltier3/Library/CloudStorage/OneDrive-NavalPostgraduateSchool/1Documents/0Research_Code/swarm/Matlab/results";
elseif comp_sys == "h"      % Hamming
    results     =  "/home/donald.peltier/swarm/model/matlab/results/Swarm3";
end

results_folder  = fullfile(results, results_name);
if ~exist(results_folder, 'dir')    % Check if the results folder exists
    mkdir(results_folder);          % If the folder does not exist, create it
end

if save_movie_flag      % if save movie flag true...
    do_plot_flag = 1;   % ...plotting flag auto sets to true
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
info.Att_motion         = Att_motion;           % New: allows different attacker motions
info.do_plot_flag       = do_plot_flag;
info.save_movie_flag    = save_movie_flag;
info.Def_final_fraction = Def_final_fraction;
info.seed               = test_seed;            % for testing & debugging
info.results_folder     = results_folder;
info.sim_time_steps     = sim_time_steps;
info.Att_obs_dist       = Att_obs_dist;         % New variable for safe distance
info.repulsion_strength = repulsion_strength;   % New variable for repulsion strength
info.min_dist           = min_dist;
info.repel_gain         = repel_gain;

%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TESTING FUNCTION CALLS (uses "info.seed=test_seed")

% G_A_PV(info);
% GP_A_PV(info);
% A_A_PV(info);
% AP_A_PV(info);
% APv2_A_PV(info);

% hvu  = HVU_A_PV(info);
% hvup = HVUP_A_PV(info);

% OBS_G_A_PV(info);
% OBS_GP_A_PV(info);
% OBS_A_A_PV(info);
% OBS_AP_A_PV(info);

% info.Att_motion = "star"; am_star = AM_A_PV(info);
% info.Att_motion = "left"; am_left = AM_A_PV(info);
% info.Att_motion = "down"; am_down = AM_A_PV(info);

SK_G_A_PV(info)
SK_GP_A_PV(info)
SK_A_A_PV(info)
SK_AP_A_PV(info)
%}

%{%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DATA GENERATION
%% init cells to hold data for each class

% data_g={};
% data_gp={};
% data_a={};
% data_ap={};

% data_hvu={};
% data_hvup={};

% data_obsg={};
% data_obsgp={};
% data_obsa={};
% data_obsap={};

% data_am_star={};
% data_am_left={};
% data_am_down={};

data_skg={};
data_skgp={};
data_ska={};
data_skap={};


%% run simulations and record data
for seed=seedstart:seedend
    
    info.seed = seed; % store current seed in "info" struct for passing to functions

    %% ORIGINAL 4 tactics
    % g  = G_A_PV(info);
    % gp = GP_A_PV(info);
    % a  = A_A_PV(info);
    % ap = AP_A_PV(info);

    %% HVU Only
    % hvu  = HVU_A_PV(info);
    % hvup = HVUP_A_PV(info);

    %% OBSERVE Only
    % obsg  = OBS_G_A_PV(info);
    % obsgp = OBS_GP_A_PV(info);
    % obsa  = OBS_A_A_PV(info);
    % obsap = OBS_AP_A_PV(info);

    %% CONSTANT ATTACKER MOTION
    % info.Att_motion = "star"; am_star = AM_A_PV(info);
    % info.Att_motion = "left"; am_left = AM_A_PV(info);
    % info.Att_motion = "down"; am_down = AM_A_PV(info);

    %% STATION KEEPING
    skg  = SK_G_A_PV(info);
    skgp = SK_GP_A_PV(info);
    ska  = SK_A_A_PV(info);
    skap = SK_AP_A_PV(info);


    %% APPEND RUN DATA

    % data_g  = cat(1,data_g,{g});
    % data_gp = cat(1,data_gp,{gp});
    % data_a  = cat(1,data_a,{a});
    % data_ap = cat(1,data_ap,{ap});

    % data_hvu  = cat(1,data_hvu,{hvu});
    % data_hvup = cat(1,data_hvup,{hvup});
    
    % data_obsg  = cat(1,data_obsg,{obsg});
    % data_obsgp = cat(1,data_obsgp,{obsgp});
    % data_obsa  = cat(1,data_obsa,{obsa});
    % data_obsap = cat(1,data_obsap,{obsap});
    
    % data_am_star=cat(1,data_am_star,{am_star});
    % data_am_left=cat(1,data_am_left,{am_left});
    % data_am_down=cat(1,data_am_down,{am_down});

    data_skg  = cat(1,data_skg,{skg});
    data_skgp = cat(1,data_skgp,{skgp});
    data_ska  = cat(1,data_ska,{ska});
    data_skap = cat(1,data_skap,{skap});

end

%% Save data matricies
if savemat

    if comp_sys     == "m"      % Macbook
        mat_folder     =  "/Users/DWPeltier3/Library/CloudStorage/OneDrive-NavalPostgraduateSchool/1Documents/0Research_Code/swarm/data/mat_files";
    elseif comp_sys == "h"      % Hamming
        mat_folder     =  "/home/donald.peltier/swarm/data/mat_files";
    end
    % Specify .mat files folder path
    mat_path  = fullfile(mat_folder, results_name);
    % Check if the folder exists, if not, create it
    if ~exist(mat_path, 'dir')
        mkdir(mat_path);
    end

    %% Save each class' data with name 'data' for ease of Numpy conversion
    
    % data = data_g;  save(fullfile(mat_path, 'data_g.mat'),  'data');
    % data = data_gp; save(fullfile(mat_path, 'data_gp.mat'), 'data');
    % data = data_a;  save(fullfile(mat_path, 'data_a.mat'),  'data');
    % data = data_ap; save(fullfile(mat_path, 'data_ap.mat'), 'data');

    % data = data_hvu;  save(fullfile(mat_path, 'data_hvu.mat'),  'data')
    % data = data_hvup; save(fullfile(mat_path, 'data_hvup.mat'), 'data')

    % data = data_obsg;  save(fullfile(mat_path, 'data_obsg.mat'),  'data')
    % data = data_obsgp; save(fullfile(mat_path, 'data_obsgp.mat'), 'data')
    % data = data_obsa;  save(fullfile(mat_path, 'data_obsa.mat'),  'data')
    % data = data_obsap; save(fullfile(mat_path, 'data_obsap.mat'), 'data')

    % data = data_am_star; save(fullfile(mat_path, 'data_am_star.mat'), 'data')
    % data = data_am_left; save(fullfile(mat_path, 'data_am_left.mat'), 'data')
    % data = data_am_down; save(fullfile(mat_path, 'data_am_down.mat'), 'data')
    
    % data = data_am_star; save(fullfile(mat_path, 'data_am_star2.mat'), 'data')
    % data = data_am_left; save(fullfile(mat_path, 'data_am_left2.mat'), 'data')
    % data = data_am_down; save(fullfile(mat_path, 'data_am_down2.mat'), 'data')

    % data = data_skg;  save(fullfile(mat_path, 'data_skg.mat'),  'data');
    % data = data_skgp; save(fullfile(mat_path, 'data_skgp.mat'), 'data');
    % data = data_ska;  save(fullfile(mat_path, 'data_ska.mat'),  'data');
    % data = data_skap; save(fullfile(mat_path, 'data_skap.mat'), 'data');

    % data = data_skg;  save(fullfile(mat_path, 'data_skg2.mat'),  'data');
    % data = data_skgp; save(fullfile(mat_path, 'data_skgp2.mat'), 'data');
    % data = data_ska;  save(fullfile(mat_path, 'data_ska2.mat'),  'data');
    % data = data_skap; save(fullfile(mat_path, 'data_skap2.mat'), 'data');

    data = data_skg;  save(fullfile(mat_path, 'data_skg3.mat'),  'data');
    data = data_skgp; save(fullfile(mat_path, 'data_skgp3.mat'), 'data');
    data = data_ska;  save(fullfile(mat_path, 'data_ska3.mat'),  'data');
    data = data_skap; save(fullfile(mat_path, 'data_skap3.mat'), 'data');
    
end
