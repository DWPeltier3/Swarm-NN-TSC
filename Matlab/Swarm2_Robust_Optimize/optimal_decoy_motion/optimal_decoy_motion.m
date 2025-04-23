%% OPTIMAL DECOY MOTION
%
% Runs simulation involving attackers killing decoys.
% Then uses attacker & decoy initial trajectories as initial conditions for
% an optimization problem which optimizes decoy trajectories to improve
% attacker classification via neural network.

clc; clear all; close all;

%% PARAMETERS       ******************************************************

%% COMPUTATION SYSTEM
comp_sys            = "m";      % "m" = mac, "h" = hamming: sets folder paths

%% ENGAGEMENT SIM PARAMETERS
tactic_name         = "g";      % attacker tactic name (simulation prefix): g, gp, a, ap (greedy, greedy+, auction, auction+)
Att_num             = 10;       % number of attackers (agents that kill)
Def_num             = 10;       % number of defenders (agents that can NOT kill)
Att_vel_max         = 1;        % attacker max velocity constraint
Def_vel_max         = 0.4;      % defender max velocity constraint
Def_vel_min         = 0.0001;   % defender min velocity; non-zero prevents dividing by zero when calculating acceleration
Att_accel_steps     = 10;       % attacker acceleration steps
Def_accel_flag      = 1;        % T/F: include defender acceleration during initial trajectory generation
Def_accel_steps     = 10;       % defender acceleration steps
Att_kill_range      = 1;        % attacker weapons range
Best_IT             = 1;        % T/F: whether to use best initial trajectory or definded Def_motion
Def_motion          = "star";   % defender motion profile: star, str, perL, perR, semi, zero
do_plot             = 0;        % T/F: plot and save initial trajectories during datagen
Def_final_fraction  = 0;        % final proportion of defenders remaining
seed                = 1201;     % random seed (for reproducability): use seed>1200 (so not part of train/test sets)

%% NN PARAMETERS
modelname       = "CNmc20"; % CNmc20, CNmcFull, LSTMMmc20, LSTMMmcFull  NOTE: CN are CNNEXcomb10v10
meanvarname     = "mean_var_10v10_combined";

if comp_sys         == "m"  % Macbook
    modelfolder     = "/Users/DWPeltier3/Library/CloudStorage/OneDrive-NavalPostgraduateSchool/1Documents/Research/swarm/models/";
    meanvarfolder   = "/Users/DWPeltier3/Library/CloudStorage/OneDrive-NavalPostgraduateSchool/1Documents/Research/swarm/data/mean_var/";
elseif comp_sys     == "h"  % Hamming
    modelfolder     = "/home/donald.peltier/swarm/model/matlab/";
    meanvarfolder   = "/home/donald.peltier/swarm/data/mean_var/";
end

modelpath       = modelfolder + modelname + ".mat";     % NN model
meanvarpath     = meanvarfolder + meanvarname + ".mat"; % mean & variance for scaling NN inputs (from training dataset)

%% OPTIMIZATION PARAMETERS
opt_obj     = "p";      % what to minimize/maximize: t = tf, p = probmin, o = other
traj_approx = "b";      % trajectory approximation: "s" = spline, "b" = Bernstein
N           = 40;       % Bernsein Polynomial order: 15, 40
Def_vFlag   = 1;        % T/F: 1: limits Vmax>Vmag>Vmin;  0: limits Vmax>Vmag
ProbMin     = 0.3;      % min probability for NN TRUE outputs (NN_label[all classes]>=ProbMin)


%% INITIALIZE Neural Network        **************************************

%% Load and analyze the NN
load(modelpath, "net")                          % net = loaded trained neural network
tnn             = net.Layers(1).MinLength;      % NN #time steps
fnn             = net.Layers(1).InputSize;      % NN #features
sim_time_steps  = tnn;                          % ensures number of simulation time steps = NN input requirement
% sim_time_steps  = 50;                           % if ~= tnn; number of simulation time steps (ex:3 = 0,1,2)

%% Load mean and variance from NN training data IOT scale inference data
params          = load(meanvarpath);
mean_values     = params.mean;                  % load mean and variance
variance_values = params.variance;
std_values      = sqrt(variance_values);        % Convert variance to standard deviation


%% STORE VARIABLES FOR FUNCTION PASSING       ****************************

%% Simulation specific
% data.sim                = sim;
% data.true_label_idx     = true_label_idx;
data.Att_num            = Att_num;
data.Def_num            = Def_num;
data.Att_vel_max        = Att_vel_max;
data.Def_vel_max        = Def_vel_max;
data.Def_vel_min        = Def_vel_min;
data.Att_accel_steps    = Att_accel_steps;
data.Def_accel_flag     = Def_accel_flag;
data.Def_accel_steps    = Def_accel_steps;
data.Att_kill_range     = Att_kill_range;
% data.Def_motion         = Def_motion;
data.Def_final_fraction = Def_final_fraction;
data.seed               = seed;
data.sim_time_steps     = sim_time_steps;

%% NN Specific
data.net                = net;
data.mean_values        = mean_values;
data.std_values         = std_values;

%% Optimization specific
data.opt_obj            = opt_obj;
data.traj_approx        = traj_approx;
data.Def_vFlag          = Def_vFlag;
data.ProbMin            = ProbMin;
% data.tf                 = tf;
% data.Cx0                = Cx(1,:); % keep Initial Position constant and vary other coefficients
% data.Cy0                = Cy(1,:); % shouldn't = BN\DefPos? (actually, C=DefPos)


%% FIND BEST INITIAL MOTION (OPTIONAL)       *****************************

if Best_IT
    Def_motion = findBestInitialMotion(data);
end

%% RESULTS & PLOTTING       **********************************************

results_name    = sprintf('%s_%d_%s_ProbMin%d_%dv%d_%s_Accel%dv%d_%d_Traj_%s_Obj_%s',...
                  tactic_name, seed, modelname, ProbMin*100, Att_num, Def_num, Def_motion,...
                  Att_accel_steps, Def_accel_steps, Def_accel_flag, traj_approx, opt_obj);

if comp_sys     == "m"      % Macbook
    results     = "/Users/DWPeltier3/Library/CloudStorage/OneDrive-NavalPostgraduateSchool/1Documents/Research/swarm/Matlab/results";
elseif comp_sys == "h"      % Hamming
    results     = "/home/donald.peltier/swarm/model/matlab/results";
end

results_folder  = fullfile(results, results_name);
if ~exist(results_folder, 'dir')                % Check if the results folder exists
    mkdir(results_folder);                      % If the folder does not exist, create it
end

diary(fullfile(results_folder, 'output.txt'));  % records all command window output to results folder

plot_width      = 600;                          % ensures able to see all plots easily
plot_height     = 350;                          % ensures able to see all plots easily


%% GENERATE INITIAL CONDITIONS       *************************************

%% Define the simulation function handles & true label index based on tactic_name
[sim_IC, sim, true_label_idx] = tacticSimAndLabel(tactic_name);

%% Generate IC: states [Batch=1,Time,Feature]: Pax,Pay,Vax,Vay...Pdx,Pdy,Vdx,Vdy
states = sim_IC(Att_num, Def_num, Att_vel_max, Def_vel_max,...
                Def_vel_min , Att_accel_steps, Def_accel_flag, Def_accel_steps, Att_kill_range,...
                Def_motion, Def_final_fraction, do_plot, results_folder, seed, sim_time_steps);

%% Parse "states" into features
states           = squeeze(states); % remove batch dimension
states_per_agent = length(states(1,:))/(Att_num+Def_num);

AttackersPos_x  = states(:,1:Att_num);
AttackersPos_y  = states(:,1*Att_num+1:2*Att_num);
AttackersVel_x  = states(:,2*Att_num+1:3*Att_num);
AttackersVel_y  = states(:,3*Att_num+1:4*Att_num);

def_idx         = states_per_agent*Att_num + 1; % first defender feature index

DefendersPos_x  = states(:,def_idx:def_idx + Def_num-1);
DefendersPos_y  = states(:,def_idx+1*Def_num:def_idx+2*Def_num-1);
DefendersVel_x  = states(:,def_idx+2*Def_num:def_idx+3*Def_num-1);
DefendersVel_y  = states(:,def_idx+3*Def_num:def_idx+4*Def_num-1);

%% If defenders are killed (used "FULL" time), replace "NaN" positions
    % (must have non-NaN initial conditions to start optimization)
    % Use last alive constant velocity to avoid sudden acceleration
DefendersPos_x  = fill_nans_assume_constant_velocity(DefendersPos_x);
DefendersPos_y  = fill_nans_assume_constant_velocity(DefendersPos_y);


%% CALCULATE OPTIMIZATION VARIABLES       ********************************

tf                  = sim_time_steps-1;
time_steps_normed   = linspace(0,1,sim_time_steps); % 20 or 55 time steps 0 to 1

if traj_approx == "b"       % Bernstein
    % calculate Bernstein matrix and Differential matrix for trajectory smoothing
    BN      = bernsteinMatrix(N,time_steps_normed);
    Dm      = Diff_elev(N,1)';
    % Convert defender position time series into Bernstein coefficients
    Cx      = BN\DefendersPos_x;
    Cy      = BN\DefendersPos_y;
    % save to "data" structure (here vs. below for "if/else" efficiency)
    data.N  = N;
    data.BN = BN;
    data.Dm = Dm;

elseif traj_approx == "s"   % Spline
    Cx      = DefendersPos_x;
    Cy      = DefendersPos_y;
end


%% STORE VARIABLES FOR FUNCTION PASSING       ****************************

data.sim                = sim;
data.true_label_idx     = true_label_idx;
data.Def_motion         = Def_motion;
data.tf                 = tf;
data.Cx0                = Cx(1,:); % keep Initial Position constant and vary other coefficients
data.Cy0                = Cy(1,:); % shouldn't = BN\DefPos? (actually, C=DefPos)


%% TEST NN INFERENCE ON IC DATA       ************************************

A_states = cat(2,AttackersPos_x,AttackersPos_y,AttackersVel_x,AttackersVel_y);
A_states_scaled = (A_states - mean_values) ./ std_values;
[NN_Prediction_1_Class,~] = predict(net, A_states_scaled)

D_ic= cat(3,DefendersPos_x',...         % initial decoy trajectory
            DefendersPos_y',...
            DefendersVel_x',...
            DefendersVel_y');
All_Classes_True_Predictions = truePredictAllClasses(D_ic, data)   % initial NN true predictions


%% RUN OPTIMIZATION       ************************************************

%% Prepare optimization variable initial conditions (exclude initial position b/c constant)
x1      = reshape(Cx(2:end,:),[],1);        % reshape to column vector
x2      = reshape(Cy(2:end,:),[],1);
x1_len  = length(x1); data.x1_len=x1_len;
x2_len  = length(x2); data.x2_len=x2_len;

%% Set optimization variable Upper and Lower bounds (ub &  lb)
if data.opt_obj == "t"                      % variable tf
    x0  = [x1;x2;tf];
    lb  = [-inf*ones(x1_len + x2_len,1); 0];
    ub  = [ inf*ones(x1_len + x2_len,1); tf+eps];

elseif data.opt_obj == "p"                  % variable ProbMin
    x0  = [x1;x2;ProbMin];
    lb  = [-inf*ones(x1_len + x2_len,1); ProbMin-eps];
    ub  = [ inf*ones(x1_len + x2_len,1); 1];

elseif data.opt_obj == "o"                  % fixed tf & ProbMin
    x0  = [x1;x2];
    lb  = -inf*ones(x1_len + x2_len,1);
    ub  =  inf*ones(x1_len + x2_len,1);
end

%% Cost and Constraints
cost    = @(x)costFunc(x,data);
c       = @(x)myconstraints(x,data);
A       = [];
b       = [];
Aeq     = [];
beq     = [];

%% Define "fmincon" options
options = optimoptions(@fmincon,...
                       'Algorithm','interior-point',...
                       'FiniteDifferenceType', 'central',...
                       'ConstraintTolerance',1e-6,...
                       'Display','iter',...
                       'FiniteDifferenceStepSize', 1e-9,...
                       'MaxFunctionEvaluations',1e6,...
                       'MaxIterations',100,...
                       'OptimalityTolerance',1e-12,...
                       'StepTolerance',1e-10,...
                       'FunctionTolerance',1e-12,...
                       'EnableFeasibilityMode',true,...
                       'SubproblemAlgorithm' , 'cg',...
                       'InitBarrierParam',1e-1);
%% Additional options commented out for potential future use
% 'Algorithm','interior-point',...
% 'FiniteDifferenceType', 'central',...     %'forward', 'central'
% 'ConstraintTolerance',1e-3...             % 1e-3, 1e-6
% 'Display','iter',...                      %'iter', 'none', 'off', 'final'
% 'FiniteDifferenceStepSize', 1e-9,...      % 1e-9, 1e-12
% 'MaxFunctionEvaluations',1e6,...
% 'MaxIterations',100,...                   % 100, 10
% 'OptimalityTolerance',1e-12,...           % 1e-8, 1e-12
% 'StepTolerance',1e-10,...                 % 1e-4, 1e-10
% 'FunctionTolerance',1e-12,...             % 1e-8, 1e-12
% 'EnableFeasibilityMode',true,...          % true, false
% 'SubproblemAlgorithm' , 'cg',...          % use if feasibility mode used
% 'InitBarrierParam',1e-1);                 % 1e-2, 1e-1

%% Check if optimization has already been completed (if yes load results)
results_matfile = fullfile(results_folder, [results_name, '.mat']);

if isfile(results_matfile)
    %% LOAD
    fprintf("File %s already exists. Skipping fmincon & loading 'data'.\n", results_name);
    load(results_matfile, 'data');
    Px=data.Px;    Py=data.Py;
    Vx=data.Vx;    Vy=data.Vy;
    Ax=data.Ax;    Ay=data.Ay;

else
    %% RUN & SAVE
    fprintf("Running Optimization\n");
    tic
    [x_opt,Jout,exitflag,outputb] = fmincon(cost,x0,A,b,Aeq,beq,lb,ub,c,options);
    time2opt=toc
    data.time2opt = time2opt;
    fprintf("Exit Flag:  %d \n", exitflag)
    fprintf("Final Cost: J = %.4f \n", Jout)
    if exitflag ~= 1 && exitflag ~= 2 && exitflag ~= -2
        % Ensure "best feasible" solution is not empty
        if isfield(outputb, 'bestfeasible') && ~isempty(outputb.bestfeasible)
            disp('USED BEST FEASIBLE: TRY AGAIN WITH SubproblemAlgorithm = cg')
            x_opt = outputb.bestfeasible.x;
            disp(outputb.bestfeasible)
        else
            disp("BEST FEASIBLE EMPTY: Using fmincon output")
        end
    elseif exitflag == -2
        if isfield(outputb, 'bestfeasible') && isempty(outputb.bestfeasible)
            disp('BEST FEASIBLE EMPTY')
        end
        disp('No feasible point found...PLOTS ARE JUST FOR SHOW')
        % disp('No feasible point found...ending script')
        % return
    end

    %% Save optimized variables to 'data'
    x = x_opt;
    if data.opt_obj == "t"      % variable tf
        tf = x(end);
    elseif data.opt_obj == "p"  % variable ProbMin
        ProbMin = x(end);
    elseif data.opt_obj == "o"  % fixed tf & ProbMin
    end
    data.x=x;
    data.tf=tf;
    data.ProbMin=ProbMin;

    %% convert optimization variables to defender PVA and add to "data"
    [Px,Py,Vx,Vy,Ax,Ay] = Opt2PVA(x,data);
    data.Px=Px;     data.Py=Py;
    data.Vx=Vx;     data.Vy=Vy;
    data.Ax=Ax;     data.Ay=Ay;

    %% Save "data": paramaters & optimized PVA
    if exitflag ~= -2
        save(results_matfile, 'data');
        fprintf("Saving results ('data') as filename %s .\n", results_matfile);
    end
end


%% Display results        ************************************************

%% Visualize overall engagemnt IC
figure(1);  hold on; sgtitle('Engagement IC Overview');
goA     = plot(AttackersPos_x, AttackersPos_y,'r');
goD     = plot(DefendersPos_x, DefendersPos_y, 'b');
legend([goA(1),goD(1)],'Attacker','Defender','Location', 'best');  % Automatically place the legend in the best location
hold off;
set(gcf, 'Position', [0, 500, plot_width, plot_height]); % [left, bottom, width, height]
saveas(gcf, fullfile(results_folder, '1_engagement_IC_overview.png'));

%% Visualize optimal vs IC decoy motion
figure(2); hold on; sgtitle('Optimal vs. Initial (Defender Motions)');
goD     = plot(DefendersPos_x, DefendersPos_y,'ro');    % initial decoy motion
goT     = plot(Px, Py, 'b');                            % optimal decoy motion
legend([goT(1),goD(1)],'Optimal','Initial','Location', 'best');
hold off;
set(gcf, 'Position', [700, 500, plot_width, plot_height]); % [left, bottom, width, height]
saveas(gcf, fullfile(results_folder, '2_optimal_vs_initial_defender_motions.png'));

%% Visualize optimal decoy Velocity Magnitude
Vn          = vecnorm(cat(3,Vx,Vy),2,3);                % norm of optimal velocity
Vn_ic       = vecnorm(cat(3,DefendersVel_x,DefendersVel_y),2,3); % norm of IC velocity
plot_lim    = size(Vn,1);                               % required b/c spline size(V)<Vic
figure(3); hold on; sgtitle('Optimal Defender Velocity');
plot(time_steps_normed(1:plot_lim),Vn);
plot(time_steps_normed(1:plot_lim),Vn_ic(1:plot_lim,:),':k');
xlabel('Time [normalized 0 to 1]');
ylabel('Velocity');
hold off;
set(gcf, 'Position', [0, 0, plot_width, plot_height]); % [left, bottom, width, height]
saveas(gcf, fullfile(results_folder, '3_optimal_defender_velocity.png'));

%% Visualize optimal decoy Acceleration Magnitude
An          = vecnorm(cat(3,Ax,Ay),2,3);                % norm of optimal acceleration
plot_lim    = size(An,1);                               % required b/c spline size(A)<Aic
figure(4);  hold on; sgtitle('Optimal Defender Acceleration');
plot(time_steps_normed(1:plot_lim),An);
xlabel('Time [normalized 0 to 1]');
hold off;
set(gcf, 'Position', [700, 0, plot_width, plot_height]); % [left, bottom, width, height]
saveas(gcf, fullfile(results_folder, '4_optimal_defender_acceleration.png'));

%% Compare optimal vs initial classifications
fprintf("Compare optimal vs initial classifications AND evaluate overall improvement.\n");
if data.traj_approx == "s"
    Vx = [Vx;Vx(end,:)]; % will never use last Vx, but required for "CAT" below
    Vy = [Vy;Vy(end,:)];
end
D_opt       = cat(3,Px',Py',Vx',Vy');               % optimized decoy trajectory
D_ic        = cat(3,DefendersPos_x',...             % initial decoy trajectory
                    DefendersPos_y',...
                    DefendersVel_x',...
                    DefendersVel_y');
y_true_o    = truePredictAllClasses(D_opt, data)    % optimized NN true predictions
y_true_i    = truePredictAllClasses(D_ic, data)     % initial NN true predictions
delta       = y_true_o - y_true_i                   % individual improvements
delta_total = sum(delta)                            % overall improvement

diary off;


%*************************************************************************
%*************************    FUNCTIONS   ********************************
%*************************************************************************


%% COST FUNCTION       ***************************************************
function J = costFunc(x,data)
% returns cost based on "optimization objective"; t = min tf, p = max ProbMin, o = other

if data.opt_obj == "t"
    J = x(end);
elseif data.opt_obj == "p"
    J = -x(end);
elseif data.opt_obj == "o"
    % unpack "data" variables required for calculations below
    sim                 = data.sim;
    true_label_idx      = data.true_label_idx;
    Att_num             = data.Att_num;
    Def_num             = data.Def_num;
    Att_vel_max         = data.Att_vel_max;
    Att_accel_steps     = data.Att_accel_steps;
    Att_kill_range      = data.Att_kill_range;
    Def_final_fraction  = data.Def_final_fraction;
    seed                = data.seed;
    sim_time_steps      = data.sim_time_steps;
    net                 = data.net;
    % convert optimization variables into defender positions and velocities
    [Px,Py,Vx,Vy,~,~] = Opt2PVA(x,data);
    if data.traj_approx == "s"
        Vx = [Vx;Vx(end,:)]; % will never use last Vx, but required for "CAT" below
        Vy = [Vy;Vy(end,:)];
    end
    % run sim using optimized defender positions to generate NN inputs = Attacker states
    D_states = cat(3,Px',Py',Vx',Vy');  % [agent, time, feature Px=1 Py=2 Vx=3 Vy=4]
    A_states = sim(D_states, Att_num, Def_num, Att_vel_max,...
        Def_final_fraction, seed, Att_accel_steps, Att_kill_range, sim_time_steps);
    A_states_scaled = SqueezeScaleStates(A_states, data);
    % COST = TRUE PREDICTION PROBABILITY FOR 1 CLASS
    label = predict(net, A_states_scaled);  % [label,score]...label is probability vector
    J = -double(label(true_label_idx));     % predicted probability for true label, fmincon requires "double" type
end
end


%% CONSTRAINT FUNCTION       *********************************************
function [C,Ceq] = myconstraints(x,data)

% unpack "data" variables required for calculations below
Def_vel_min     = data.Def_vel_min;
Def_vel_max     = data.Def_vel_max;
Def_accel_max   = Def_vel_max/data.Def_accel_steps;
if data.opt_obj == "p"  % variable ProbMin
    ProbMin     = x(end);
else
    ProbMin     = data.ProbMin;
end

% convert optimization variables into defender positions, velocities, and accelerations
[Px,Py,Vx,Vy,Ax,Ay] = Opt2PVA(x,data);

% Kinematic Constraints: velocity and acceleration magnitudes
V_magnitude_squared = Vx.^2 + Vy.^2;
A_magnitude_squared = Ax.^2 + Ay.^2;
V_magnitude_squared = V_magnitude_squared(:);           % convert to column vector
A_magnitude_squared = A_magnitude_squared(:);           % convert to column vector

if data.Def_vFlag == false
    C_kin = [A_magnitude_squared - Def_accel_max^2;...  % Acceleration constraint
             V_magnitude_squared - Def_vel_max^2];      % Max velocity constraint
else
    C_kin = [A_magnitude_squared - Def_accel_max^2;...  % Acceleration constraint
             V_magnitude_squared - Def_vel_max^2;...    % Max velocity constraint
            -V_magnitude_squared + Def_vel_min^2];      % Min velocity constraint
end

% NN Prediciton Constraint: ALL FOUR SIM_A_PV --> NN True Label >= ProbMin
if data.traj_approx == "s"
    Vx = [Vx;Vx(end,:)]; % will never use last Vx, but required for "CAT" below
    Vy = [Vy;Vy(end,:)];
end
D_states    = cat(3,Px',Py',Vx',Vy');                   % [agent, time, feature Px=1 Py=2 Vx=3 Vy=4]
y_true      = truePredictAllClasses(D_states, data);    % 4 true prediction probabilities (double type)
C_prob      = ProbMin - y_true';                        % column vector of constraints

% Combine all inequality constraints
C   = [C_kin; C_prob];
Ceq = [];
end


%% TACTIC SIM & LABEL       *********************************************
function [sim_IC, sim, true_label_idx] = tacticSimAndLabel(tactic_name)
    if strcmp(tactic_name, 'g')
        sim_IC          = @Greedy_AD_PV;        % handle for IC datagen function
        sim             = @Greedy_A_PV;         % handle for NN datagen function
        true_label_idx  = 1;                    % true probability index (want to maximize)
        % true_label_idx = 2;                   % false probability index (want to spoof)
    elseif strcmp(tactic_name, 'gp')
        sim_IC          = @GreedyPro_AD_PV;
        sim             = @GreedyPro_A_PV;
        true_label_idx  = 2;
    elseif strcmp(tactic_name, 'a')
        sim_IC          = @Auction_AD_PV;
        sim             = @Auction_A_PV;
        true_label_idx  = 3;
    elseif strcmp(tactic_name, 'ap')
        sim_IC          = @AuctionPro_AD_PV;
        sim             = @AuctionPro_A_PV;
        true_label_idx  = 4;
    else
        error('Invalid simulation name');
    end
end



%% SCALE & SQUEEZE NN INPUTS       ***************************************
function scaled_states = SqueezeScaleStates(states, data)
% input is of dimensions [Batch, Time, Feature]
% output is of dimensions [Time, Feature] & scaled
mean_values     = data.mean_values;
std_values      = data.std_values;
states = squeeze(states); %remove batch dimension for inference (B=1 instance): [1,T,F] -> [T,F]
scaled_states = (states - mean_values) ./ std_values;
end


%% BEZIER STUFF       ***************************************************
function Dm = Diff(N,tf )
% derivative of a Bezier curve
% INPUT
% N: number of nodes
% OUTPUT
% Dm{N}: differentiation matrix for bez curves of order N (N+1 ctrl points)
% Notes:
% If Cp are the control points of bez, then the control points of bezdot are Cpdot = Cp*Dm
% To compute bezdot with N ctrl points, degree elevation must be performed 
Dm = -[N/tf*eye(N); zeros(1,N)]+[zeros(1,N);N/tf*eye(N)];
end
function Telev = deg_elev(N) %********************************************
% INPUT
% N order of Bezier curve
% OUTPUT
% Telev{N}: Transformation matrix from Nth order (N+1 control points) to
% (N+1)th order (N+2 control points)
% If Cp is of order N-1, then Cp*Telev{N-1} is of order N
% see Equation (12+1) in https://pdfs.semanticscholar.org/f4a2/b9def119bd9524d3e8ebd3c52865806dab5a.pdf
% Paper: A simple matrix form for degree reduction of Be´zier curves using ChebyshevBernstein basis transformations
if N < 5 
  es='ERROR: The approximation order should be at least 5';
  disp(es); Dm = [];
return
end
for i = 1:1:N
    Telev{i} = zeros(i+2,i+1);
    for j = 1:1:i+1
        Telev{i}(j,j) = i+1-(j-1);
        Telev{i}(j+1,j) = 1+(j-1);
    end
    Telev{i} = 1/(i+1)*Telev{i}';
end
end
function Dm = Diff_elev(N,tf) %*******************************************
% derivative of a Bezier curve
% INPUT
% N: number of nodes, tf: final time
% OUTPUT
% Dm{N}: differentiation matrix for bez curves of order N (N+1 ctrl points)
% The differentiation matrix is (N+1)by(N+1), ie. differently from Diff,
% this matrix gives a derivative of the same order of the curve to be
% differentiated
% Notes:
% If Cp are the control points of bez, then the control points of bezdot are Cpdot = Cp*Dm
Dm = Diff(N,tf);
Telev = deg_elev(N);
Dm = Dm*Telev{N-1};
end


%% NN TRUE OUTPUT PREDICTIONS GIVEN DECOY MOTION       *******************
function [y_true] = truePredictAllClasses(D_states, data)
% Decoy motion will result in 4 different attacker motions (b/c 4 different
% tactics, or classes). The true predictions from these 4 different
% attacker motions is what we want to maximize: one decoy motion that
% causes all four tactics to reveal themselves with highest probability.

% unpack "data" variables required for calculations below
Att_num             = data.Att_num;
Def_num             = data.Def_num;
Att_vel_max         = data.Att_vel_max;
Def_final_fraction  = data.Def_final_fraction;
seed                = data.seed;
Att_accel_steps     = data.Att_accel_steps;
Att_kill_range      = data.Att_kill_range;
sim_time_steps      = data.sim_time_steps;
net                 = data.net;

% Greedy
A_states        = Greedy_A_PV(D_states, Att_num, Def_num, Att_vel_max,...
                    Def_final_fraction, seed, Att_accel_steps, Att_kill_range, sim_time_steps);
A_states_scaled = SqueezeScaleStates(A_states, data);
label           = predict(net, A_states_scaled);
y1              = label(1);

% GreedyPRO
A_states        = GreedyPro_A_PV(D_states, Att_num, Def_num, Att_vel_max,...
                    Def_final_fraction, seed, Att_accel_steps, Att_kill_range, sim_time_steps);
A_states_scaled = SqueezeScaleStates(A_states, data);
label           = predict(net, A_states_scaled);
y2              = label(2);

% Auction
A_states        = Auction_A_PV(D_states, Att_num, Def_num, Att_vel_max,...
                    Def_final_fraction, seed, Att_accel_steps, Att_kill_range, sim_time_steps);
A_states_scaled = SqueezeScaleStates(A_states, data);
label           = predict(net, A_states_scaled);
y3              = label(3);

% AuctionPRO
A_states        = AuctionPro_A_PV(D_states, Att_num, Def_num, Att_vel_max,...
                    Def_final_fraction, seed, Att_accel_steps, Att_kill_range, sim_time_steps);
A_states_scaled = SqueezeScaleStates(A_states, data);
label           = predict(net, A_states_scaled);
y4              = label(4);

% FOUR TRUE PREDICTIONS based on 4 different attacker motions (sims)
y_true          = [y1, y2, y3, y4];
y_true          = double(y_true);
end


%% PADDING: Replaces NaNs with Last Alive Position Value       ***********
function A = fill_nans(A)
% Matrix "A" is of dimensions [Time, Agent]
% If an agent is killed, it's POSITION changes to "NaN" (Not a Number)
% Replaces the NaNs in each column (agent) with last non-NaN value.
% ie. values of position before the agent was killed
    for ii = 1:size(A,2)
        I = A(1,ii);
        for jj = 2:size(A,1)
            if isnan(A(jj,ii))
                A(jj,ii) = I;
            else
                I  = A(jj,ii);
            end
        end
    end
end


%% PADDING: Replaces Position NaNs assuming Constant Velocity       ******
function A = fill_nans_assume_constant_velocity(A)
% Matrix "A" is of dimensions [Time, Agent]
% If an agent is killed, its position changes to "NaN" (Not a Number)
% Replaces the position NaNs in each column (agent) with extrapolated
% values based on constant velocity calculated from the last two non-NaN 
% position values. This should help keep percceived acceleration within
% limits (ie. if velocity suddenly goes to zero, accelerations spikes)
    for ii = 1:size(A, 2)
        % Find the last two non-NaN indices
        non_nan_indices = find(~isnan(A(:, ii)));
        % Get the indices and values of the last two non-NaN values
        idx_last = non_nan_indices(end);
        idx_second_last = non_nan_indices(end-1);
        position_last = A(idx_last, ii);
        position_second_last = A(idx_second_last, ii);
        % Calculate the constant velocity
        velocity = position_last - position_second_last;
        % Fill the NaN values with extrapolated values based on velocity
        for jj = (idx_last+1):size(A, 1)
            A(jj, ii) = position_last + velocity * (jj - idx_last);
        end
    end
end


%% CALCULATE VEL & ACCEL GIVEN POSITION TIMESERIES & TIME       **********
function [Vel_x,Vel_y,Acc_x,Acc_y] = Position2Vel_Accel(Pos_x,Pos_y,tf)
    time_steps_normed = linspace(0,1,tf+1);
    dt = tf*(time_steps_normed(2) - time_steps_normed(1));

    Vel_x = (Pos_x(2:end,:)-Pos_x(1:end-1,:))/dt;
    Vel_y = (Pos_y(2:end,:)-Pos_y(1:end-1,:))/dt;

    Acc_x = (Vel_x(2:end,:)-Vel_x(1:end-1,:))/dt;
    Acc_y = (Vel_y(2:end,:)-Vel_y(1:end-1,:))/dt;
end


%% CONVERT OPTIMIZATION VARIABLES TO POS, VEL & ACCEL       **************
function [Px,Py,Vx,Vy,Ax,Ay] = Opt2PVA(x,data)
% Converts optimization variable to position, velocity, acceleration
% unpack "data" variables required for calculations below
if data.opt_obj == "t"      % variable tf
    tf  = x(end);
else
    tf  = data.tf;
end
x1_len  = data.x1_len;
x2_len  = data.x2_len;
Def_num = data.Def_num;
% convert opt variable to x&y ccoefficients
x1      = x(1:x1_len);
x2      = x(x1_len+1:x1_len+x2_len);
Cx      = reshape(x1,[],Def_num);
Cy      = reshape(x2,[],Def_num);
Cx      = [data.Cx0;Cx];
Cy      = [data.Cy0;Cy];
% convert coefficients to PVA
if data.traj_approx == "b"   % Bernstein
    BN  = data.BN;
    Dm  = data.Dm;
    Px  = BN*Cx;             % defender position [time, agent]
    Py  = BN*Cy;
    Vx  = BN*Dm*Cx/tf;       % defender velocity [time, agent]
    Vy  = BN*Dm*Cy/tf;
    Ax  = BN*Dm*Dm*Cx/tf^2;  % defender acceleration [time, agent]
    Ay  = BN*Dm*Dm*Cy/tf^2;
elseif data.traj_approx == "s"   % Spline
    Px  = Cx;
    Py  = Cy;
    [Vx,Vy,Ax,Ay] = Position2Vel_Accel(Px,Py,tf); % note: #A<#V<#P by 1
end
end


%% FIND BEST INITIAL MOTION       ****************************************

function Def_motion = findBestInitialMotion(data)
    motions = ["star", "str", "perL", "perR", "semi", "zero"]
    num_motions = size(motions,2);
    motion_total = zeros(1,num_motions);
    sim_IC          = @Greedy_AD_PV; % any will work
    do_plot_temp    = 0; % not used
    results_folder  = 0; % not used
    % unpack "data" variables required for calculations below
    Att_num             = data.Att_num;
    Def_num             = data.Def_num;
    Att_vel_max         = data.Att_vel_max;
    Def_vel_max         = data.Def_vel_max;
    Def_vel_min         = data.Def_vel_min;
    Att_accel_steps     = data.Att_accel_steps;
    Def_accel_flag      = data.Def_accel_flag;
    Def_accel_steps     = data.Def_accel_steps;
    Att_kill_range      = data.Att_kill_range;
    Def_final_fraction  = data.Def_final_fraction;
    seed                = data.seed;
    sim_time_steps      = data.sim_time_steps;

    for i = 1:num_motions
        Def_motion = motions(i);
        % fprintf("Motion: %s \n", Def_motion);
        % Generate IC: states [Batch=1,Time,Feature]: Pax,Pay,Vax,Vay...Pdx,Pdy,Vdx,Vdy
        [states] = sim_IC(Att_num, Def_num, Att_vel_max, Def_vel_max,...
        Def_vel_min , Att_accel_steps, Def_accel_flag, Def_accel_steps, Att_kill_range,...
        Def_motion, Def_final_fraction, do_plot_temp, results_folder, seed, sim_time_steps);
    
        % Parse "states" into features
        states           = squeeze(states); % remove batch dimension
        states_per_agent = length(states(1,:))/(Att_num+Def_num);
        
        def_idx         = states_per_agent*Att_num + 1; % first defender feature index
        
        DefendersPos_x  = states(:,def_idx:def_idx + Def_num-1);
        DefendersPos_y  = states(:,def_idx+1*Def_num:def_idx+2*Def_num-1);
        DefendersVel_x  = states(:,def_idx+2*Def_num:def_idx+3*Def_num-1);
        DefendersVel_y  = states(:,def_idx+3*Def_num:def_idx+4*Def_num-1);
    
        % If defenders are killed (used "FULL" time), replace "NaN" positions
        % (must have non-NaN initial conditions to start optimization)
        % Use last alive constant velocity
        DefendersPos_x  = fill_nans_assume_constant_velocity(DefendersPos_x);
        DefendersPos_y  = fill_nans_assume_constant_velocity(DefendersPos_y);
    
        D_ic= cat(3,DefendersPos_x',...         % initial decoy trajectory
                    DefendersPos_y',...
                    DefendersVel_x',...
                    DefendersVel_y');
    
        motion_total(i)=sum(truePredictAllClasses(D_ic, data));
    end
    motion_total
    [~,best_motion_idx] = max(motion_total);
    best_motion         = motions(best_motion_idx)
    Def_motion          = best_motion;
end