%% Collect & Save Data
% Use this script to run 4 algorithms below using inputs
% specified "seedrange" number of times and collect defender 
% data (position, velocity)

clear all;

%% Number of runs per class
seedrange=1200; % # of samples (runs) to collect (per algorithm)

%% Simulation Inputs
N_attacker=10;     % number of attackers
N_defender=10;     % number of defenders (agents that kill; "attackers" for ONR)
Defender_v_max=1;   % defender velocity maximum
do_plot=false;
final_fraction=0;   % final proportion of attackers remaining vs. #defenders
accel=10;           % defender acceleration steps
kill_range=1;       % defender weapons range (kill attacker range)
rand_start=false;   % attackers start in random position

%% Save Data Flag and File Name
savemat=true; %"true" will save all runs into a MATLAB .mat file

%% Run simulation multiple times and gather training data and labels
% init cells to hold data for each class
data_g={};
data_gp={};
data_a={};
data_ap={};


% run simulations and record data
for seed=1:seedrange

    %% GREEDY (Pursuit)
    g=Greedy_D_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range,rand_start);

    %% GREEDY ProNav
    gp=GreedyPro_D_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range,rand_start);

    %% Auction (Pursuit)
    a=Auction_D_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range,rand_start);

    %% Auction ProNav
    ap=AuctionPro_D_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range,rand_start);

    %% Append run data to data matricies
    data_g=cat(1,data_g,{g});
    data_gp=cat(1,data_gp,{gp});
    data_a=cat(1,data_a,{a});
    data_ap=cat(1,data_ap,{ap});
end

%% Save data matricies
if savemat
    % Specify folder path
    folderPath = '/home/donald.peltier/swarm/data/mat_files/10v10_r4800_c4_a10/'; % Replace with your desired folder path

    % Check if the folder exists, if not, create it
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end

    data = data_g; % Save each class' data to 'data' for ease of Numpy conversion
    save([folderPath 'data_g.mat'], 'data')
    data = data_gp;
    save([folderPath 'data_gp.mat'], 'data')
    data = data_a;
    save([folderPath 'data_a.mat'], 'data')
    data = data_ap;
    save([folderPath 'data_ap.mat'], 'data')
end