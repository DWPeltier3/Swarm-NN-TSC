function odm(comp_sys, Def_num, Def_accel_flag, Def_spread,...
            Best_IT_flag, Def_motion, seed, modelname,...
            opt_obj, traj_approx, fmin_algo,...
            PolyCon, scale_accel, scale_pos,...
            load_previous)

%% OPTIMAL DECOY MOTION
%
% Runs simulation involving attackers killing decoys.
% Then uses attacker & decoy initial trajectories as initial conditions for
% an optimization problem which optimizes decoy trajectories to improve
% attacker classification via neural network.

% clc; clear all; close all;
disp(datetime('now'))



%% PARAMETERS       ******************************************************

%% COMPUTATION SYSTEM
% comp_sys            = "h";      % "m" = mac, "h" = hamming: sets folder paths

%% ENGAGEMENT SIM PARAMETERS
tactic_name         = "ap";      % attacker tactic name (simulation prefix): g, gp, a, ap (greedy, greedy+, auction, auction+)
Att_num             = 10;       % number of attackers (agents that kill)
% Def_num             = 10;       % number of defenders (agents that can NOT kill)
Att_vel_max         = 1;        % attacker max velocity constraint
Def_vel_max         = 0.4;      % defender max velocity constraint
Def_vel_min         = 0.0001;   % defender min velocity; non-zero prevents dividing by zero when calculating acceleration
Att_accel_steps     = 10;       % attacker acceleration steps
% Def_accel_flag      = 0;        % T/F (1/0): include defender acceleration during initial trajectory generation
Def_accel_steps     = 10;       % defender acceleration steps
% Def_spread          =5;         % defender initial dispersion distance about starting point (default=5)
Att_kill_range      = 1;        % attacker weapons range
% Best_IT_flag        = 1;        % whether to use best initial trajectory or definded Def_motion
% Def_motion          = "star";   % defender motion profile: star, str, perL, perR, semi, zero
do_plot_flag        = 0;        % T/F (1/0): plot and save initial trajectories during datagen
save_movie_flag     = 0;        % T/F (1/0): save initial trajectories moving plot during datagen
Def_final_fraction  = 0;        % final proportion of defenders remaining
% seed                = 1201;     % random seed (for reproducability): use seed>1200 (so not part of train/test sets)

%% NN PARAMETERS
% NOTE: CN are CNNEXcomb10v10 (trained on combined motion dataset)
% modelname       = "CNmc20"; % CNmc20_10v1C10_semi, CNmc20, CNmc20_r40k, CNmcFull, LSTMMmc20, LSTMMmcFull
if modelname == "CNmc20"
    meanvarname     = "mean_var_10v10_combined"; % meanvar must be associated with dataset used to train NN
    
elseif modelname == "CNmc20_10v1C10_semi"
    meanvarname     = "mean_var_10v1C10_semi";

elseif modelname == "CNmc20_10v1C10_combDM"
    meanvarname     = "mean_var_10v1C10_combDM";

elseif modelname == "CNmc20_10v1C10_perL"
    meanvarname     = "mean_var_10v1C10_perL";

elseif modelname == "CNmc20_r40k"
    meanvarname     = "mean_var_10v10_combined_r40k";
end

if comp_sys         == "m"  % Macbook
    modelfolder     = "/Users/DWPeltier3/Library/CloudStorage/OneDrive-NavalPostgraduateSchool/1Documents/0Research_Code/swarm/models/";
    meanvarfolder   = "/Users/DWPeltier3/Library/CloudStorage/OneDrive-NavalPostgraduateSchool/1Documents/0Research_Code/swarm/data/mean_var/";
    results     = "/Users/DWPeltier3/Library/CloudStorage/OneDrive-NavalPostgraduateSchool/1Documents/0Research_Code/swarm/Matlab/results";
elseif comp_sys     == "h"  % Hamming
    modelfolder     = "/home/donald.peltier/swarm/model/matlab/";
    meanvarfolder   = "/home/donald.peltier/swarm/data/mean_var/";
    results     = "/home/donald.peltier/swarm/model/matlab/results";
end

modelpath       = modelfolder + modelname + ".mat";     % NN model
meanvarpath     = meanvarfolder + meanvarname + ".mat"; % mean & variance for scaling NN inputs (from training dataset)

%% OPTIMIZATION PARAMETERS
% opt_obj     = "p";      % optimization objective: y = max true_pred_all, t =  min tf, p = max probmin, o = max ONE tactic prob
% traj_approx = "b";      % trajectory approximation: "s" = spline, "b" = Bernstein
% fmin_algo   = 'sqp';    % algorithm used by fmincon: 'interior-point', 'sqp'
N           = 40;       % Bernsein Polynomial order: 15, 40
Def_vFlag   = 1;        % T/F: 1: limits Vmax>Vmag>Vmin;  0: limits Vmax>Vmag
ProbMin     = 0.3;      % min probability for NN TRUE outputs (NN_true_pred[all classes]>=ProbMin)
PxCon       = 40;       % constraint on defender x-position
PyCon       = 40;       % constraint on defender y-position
% PolyCon = [0 0; 0 10; 10 10; 10 0];   % safe inside polygon constraint; vertices (x,y)
% scale_accel = 1;      % acceleration constraint scale factor
% scale_pos   = 1;        % position constraint scale factor



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
data.comp_sys           = comp_sys;
data.Att_num            = Att_num;
data.Def_num            = Def_num;
data.Att_vel_max        = Att_vel_max;
data.Def_vel_max        = Def_vel_max;
data.Def_vel_min        = Def_vel_min;
data.Att_accel_steps    = Att_accel_steps;
data.Def_accel_flag     = Def_accel_flag;
data.Def_accel_steps    = Def_accel_steps;
data.Def_spread         = Def_spread;
data.Att_kill_range     = Att_kill_range;
data.Best_IT_flag       = Best_IT_flag;
data.Def_motion         = Def_motion;
data.do_plot_flag       = do_plot_flag;
data.save_movie_flag    = save_movie_flag;
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
data.PxCon              = PxCon;
data.PyCon              = PyCon;
data.PolyCon            = PolyCon;
data.scale_accel        = scale_accel;
data.scale_pos          = scale_pos;



%% USE BEST INITIAL MOTION (OPTIONAL)       *****************************
if Best_IT_flag
    Def_motion = findBestInitialMotion(data);
    data.Def_motion = Def_motion;
end



%% RESULTS & PLOTTING       **********************************************

% Get current date and time
current_time = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm');
% Convert datetime to string
time_str = char(current_time);

results_name = sprintf('%d_%s_%dv%d_Best%d_%s_Algo_%s_Traj_%s_Obj_%s_%s',...
                  seed, modelname, Att_num, Def_num, Best_IT_flag, Def_motion,...
                  fmin_algo, traj_approx, opt_obj, time_str);

% results_name = sprintf('%d_%s_%dv%d_Best%d_%s_Accel%dv%d_%d_Traj_%s_Obj_%s_PolyC_%d_As_%d_Ps_%d_%s',...
%                   seed, modelname, Att_num, Def_num, Best_IT_flag, Def_motion,...
%                   Att_accel_steps, Def_accel_steps, Def_accel_flag,...
%                   traj_approx, opt_obj, ~isempty(PolyCon), scale_accel, scale_pos, time_str);

results_folder  = fullfile(results, results_name); %'results' set near line 45 based on comp_sys
if ~exist(results_folder, 'dir')                % Check if the results folder exists
    mkdir(results_folder);                      % If the folder does not exist, create it
end
data.results_folder = results_folder;           % required by simIC for plotting

diary(fullfile(results_folder, 'output.txt'));  % start output diary; records all command window output to results folder
plot_width      = 600;                          % ensures able to see all plots easily
plot_height     = 350;                          % ensures able to see all plots easily



%% GENERATE INITIAL CONDITIONS       *************************************

%% Define the simulation function handles & true label index based on tactic_name
[sim_IC, sim, true_label_idx] = tacticSimAndLabel(tactic_name);

%% Generate IC: states [Batch=1,Time,Feature]: Pax,Pay,Vax,Vay...Pdx,Pdy,Vdx,Vdy
states = sim_IC(data);

%% Parse "states" into features
states           = squeeze(states); % remove batch dimension
states_per_agent = length(states(1,:))/(Att_num+Def_num);

AttackersPos_x  = states(:,1:Att_num);                              % dimensions [time, attacker number]
AttackersPos_y  = states(:,1*Att_num+1:2*Att_num);
AttackersVel_x  = states(:,2*Att_num+1:3*Att_num);
AttackersVel_y  = states(:,3*Att_num+1:4*Att_num);

def_idx         = states_per_agent*Att_num + 1; % first defender feature index

DefendersPos_x  = states(:,def_idx:def_idx + Def_num-1);            % dimensions [time, defender number]
DefendersPos_y  = states(:,def_idx+1*Def_num:def_idx+2*Def_num-1);
DefendersVel_x  = states(:,def_idx+2*Def_num:def_idx+3*Def_num-1);
DefendersVel_y  = states(:,def_idx+3*Def_num:def_idx+4*Def_num-1);

%% If defenders are killed (used "FULL" time), replace "NaN" positions
    % (must have non-NaN initial conditions to start optimization)
    % Use last alive constant velocity to avoid sudden acceleration
DefendersPos_x  = fill_nans_assume_constant_velocity(DefendersPos_x);
DefendersPos_y  = fill_nans_assume_constant_velocity(DefendersPos_y);



%%=========================================================================
%%====================== PLOT IC & CONSTRAINTS ============================
%%=========================================================================
%% Visualize decoy motion IC & Constraints
figure(5); hold on; sgtitle('Initial Defender Motions & Path Constraint');
set(gcf, 'Color', 'w');  % Set the figure background to white
% pcon = plot(PolyCon(:,1), PolyCon(:,2), 'g-', 'LineWidth', 2);  % Green outline of the polygon safe-zone
pcon = fill(PolyCon(:,1), PolyCon(:,2), [0.75, 1, 0.75], 'EdgeColor', 'g', 'LineWidth', 2, 'FaceAlpha', 0.2);  % Light green filled area with green edge
goD     = plot(DefendersPos_x, DefendersPos_y, '--','LineWidth', 1);    % initial decoy motion
if modelname == "CNmcFull"
    arrow_length = 1.5; % Desired arrow length (0.5 for 20, 1.5 for Full)
else
    arrow_length = 0.5;
end
arrow_head   = 1;   % doesn't do anything
%% Define custom colors (10 distinct colors)
colors = [
    0.1216, 0.4667, 0.7059;  % Blue
    1.0000, 0.4980, 0.0549;  % Orange
    0.1725, 0.6275, 0.1725;  % Green
    0.8392, 0.1529, 0.1569;  % Red
    0.5804, 0.4039, 0.7412;  % Purple
    0.5490, 0.3373, 0.2941;  % Brown
    0.8500, 0.3250, 0.0980;  % Dark Orange
    0.4980, 0.4980, 0.4980;  % Gray
    0.5020, 0,      0;       % Maroon
    0.0902, 0.7451, 0.8118;  % Cyan
];
% plot initial motions with arrowheads
for i = 1:size(DefendersPos_x, 2)
    colorD = colors(mod(i-1, 10) + 1, :);  % Mod function cycles the index between 1 and 10
    % Plot the initial decoy motion (goD)
    goD(i) = plot(DefendersPos_x(:,i), DefendersPos_y(:,i), '--', 'Color', colorD, 'LineWidth', 1);
    % Add arrowheads for the initial motion
    U = DefendersPos_x(end, i) - DefendersPos_x(end-1, i);
    V = DefendersPos_y(end, i) - DefendersPos_y(end-1, i);
    magnitude = sqrt(U^2 + V^2);
    if magnitude ~= 0
        U_normalized = (U / magnitude) * arrow_length;
        V_normalized = (V / magnitude) * arrow_length;
    else
        U_normalized = 0;
        V_normalized = 0;
    end
    quiver(DefendersPos_x(end, i), DefendersPos_y(end, i), ...
           U_normalized, V_normalized, ...
           'Color', colorD, 'MaxHeadSize', arrow_head, 'LineWidth', 1, ...
           'AutoScale', 'off', 'Alignment', 'head');
end
hvu = scatter(0, 0, 25^2, 'y', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5); % HVU is yellow
text(0, 0, 'HVU', 'VerticalAlignment','middle', 'HorizontalAlignment','center','FontSize', 10)  % Display HVU label
axis equal;       % Force the plot to have equal scaling for both axes
% axis off;
hold off;
set(gcf, 'Position', [700, 500, plot_width, plot_height]); % [left, bottom, width, height]
saveas(gcf, fullfile(results_folder, '0_constraints_and_initial_defender_motions.png'));



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

elseif traj_approx == "s"  % Spline
    Cx      = DefendersPos_x;   % dimensions [time, defender number]
    Cy      = DefendersPos_y;
end



%% STORE VARIABLES FOR FUNCTION PASSING       ****************************
data.sim                = sim;
data.true_label_idx     = true_label_idx;
data.tf                 = tf;
data.Cx0                = Cx(1,:); % keep Initial Position constant and vary other coefficients
data.Cy0                = Cy(1,:); % shouldn't = BN\DefPos? (actually, C=DefPos)



%% TEST NN INFERENCE ON IC DATA       ************************************
A_states = cat(2,AttackersPos_x,AttackersPos_y,AttackersVel_x,AttackersVel_y);
A_states_scaled = (A_states - mean_values) ./ std_values;
D_ic= cat(3,DefendersPos_x',...                                     % initial decoy trajectory
            DefendersPos_y',...                                     % original dimensions [time, defender]
            DefendersVel_x',...                                     % new dimensions [defender, time, PxPyVxVy]
            DefendersVel_y');
NN_Prediction_1_Class = predict(net, A_states_scaled)               % initial NN prediction (4 classes sum to 1)
y_true_i = truePredictAllClasses(D_ic, data)                        % initial NN true label predictions (4 classes individually)
sum_y_true_i = sum(y_true_i)                                        % sum of y_true_i

% Update ProbMin based on Initial Predictions (ProbMin = min prediction from all classes)
 if data.opt_obj     == "p"  % variable ProbMin
    ProbMin         =  min(y_true_i)
    data.ProbMin    =  ProbMin;
end
% output for diary recording
modelname
PolyCon
data




%% RUN OPTIMIZATION       ************************************************

%% Prepare optimization variable initial conditions
x1      = reshape(Cx(2:end,:),[],1);        % exclude initial position b/c constant
x2      = reshape(Cy(2:end,:),[],1);        % reshape to column vector
x1_len  = length(x1); data.x1_len=x1_len;
x2_len  = length(x2); data.x2_len=x2_len;

%% Set optimization variable Upper and Lower bounds (ub &  lb)
if data.opt_obj == "t"                              % variable tf
    x0  = [x1;x2;tf];
    lb  = [-inf*ones(x1_len + x2_len,1); 0];
    ub  = [ inf*ones(x1_len + x2_len,1); tf+1e-3];

elseif data.opt_obj == "p"                          % variable ProbMin
    x0  = [x1;x2;ProbMin];
    % lb  = [-inf*ones(x1_len + x2_len,1); ProbMin-1e-3]; % can result in positive cost function = bad
    lb  = [-inf*ones(x1_len + x2_len,1); 0];
    ub  = [ inf*ones(x1_len + x2_len,1); 1];

elseif data.opt_obj == "o" || data.opt_obj == "y"   % fixed tf & ProbMin
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

options = optimoptions(@fmincon,...
                       'Algorithm', fmin_algo, ...
                       'FiniteDifferenceType', 'central', ...
                       'ConstraintTolerance', 1e-8, ...         % Tighter constraint tolerance
                       'Display', 'iter', ...
                       'FiniteDifferenceStepSize', 1e-7, ...    % larger step size
                       'MaxFunctionEvaluations', 1e6, ...       % more evaluations
                       'MaxIterations', 5e2, ...                % more itereations
                       'OptimalityTolerance', 1e-6, ...
                       'StepTolerance', 1e-6, ...               % relaxed step tolerance
                       'FunctionTolerance', 1e-6, ...
                       'EnableFeasibilityMode', true, ...      % Disable feasibility mode
                       'SubproblemAlgorithm', 'cg', ...
                       'InitBarrierParam', 1e-2)                % Stronger barrier at start

%% Additional options commented out for potential future use
% 'Algorithm','interior-point',...          % 'interior-point', 'sqp' 
% 'FiniteDifferenceType', 'central',...     % 'forward', 'central'
% 'ConstraintTolerance',1e-3...             % 1e-8, 1e-3, 1e-6
% 'Display','iter',...                      % 'iter', 'none', 'off', 'final'
% 'FiniteDifferenceStepSize', 1e-9,...      % 1e-7, 1e-9, 1e-12
% 'MaxFunctionEvaluations',1e6,...          % 1e6
% 'MaxIterations',1e3,...                   % 5e2, 1e2, 1e1
% 'OptimalityTolerance',1e-12,...           % 1e-6, 1e-8, 1e-12
% 'StepTolerance',1e-10,...                 % 1e-6, 1e-4, 1e-10
% 'FunctionTolerance',1e-12,...             % 1e-6, 1e-8, 1e-12
% 'EnableFeasibilityMode',true,...          % true, false
% 'SubproblemAlgorithm' , 'cg',...          % use if feasibility mode used
% 'InitBarrierParam',1e-1);                 % 1e-2, 1e-1

%% Check if EXACT optimization has already been completed (if yes load results)
% will only work if "date/time" is removed from results folders
results_matfile = fullfile(results_folder, [results_name, '.mat']);

if isfile(results_matfile)
    %% LOAD PREVIOUS RESULTS (redundant run; no need to run optimization)
    fprintf("File %s already exists. Skipping fmincon & loading variables for plotting.\n", results_name);
    load(results_matfile,'x','data','Px','Py','Vx','Vy','Ax','Ay');

else
    if isfile(load_previous)
        %% LOAD PREVIOUS RESULTS FOR CONTINUATION
        fprintf("Loading previous output 'x' as new IC 'x0' for continuation:\n %s\n", load_previous);
        x0 = load(load_previous, 'x');      % returns a "struct" when assigned to a variable
        x0 = x0.x;                          % reassign for "double" class
    end

    %% RUN & SAVE
    fprintf("Running Optimization\n");
    tic
    [x_opt,Jout,exitflag,outputb] = fmincon(cost,x0,A,b,Aeq,beq,lb,ub,c,options);
    time2opt=toc
    data.time2opt = time2opt;
    fprintf("Exit Flag:  %d \n", exitflag)
    fprintf("Final Cost: J = %.4f \n", Jout)

    %% PARTIALLY SUCCESSFUL
    if exitflag ~= 1 && exitflag ~= 2 && exitflag ~= -2
        % Ensure "best feasible" solution is not empty
        if isfield(outputb, 'bestfeasible') && ~isempty(outputb.bestfeasible)
            disp('USED BEST FEASIBLE: TRY AGAIN WITH SubproblemAlgorithm = cg')
            x_opt = outputb.bestfeasible.x;
            disp(outputb.bestfeasible)
        else
            disp("BEST FEASIBLE NOT AVAILABLE")
            disp(outputb.bestfeasible)
        end

    %% NO CONVERGENCE
    elseif exitflag == -2
        fprintf('No feasible point found...PLOTS ARE JUST FOR SHOW')
    end
    
    %% Save optimized variables to 'data'
    x = x_opt;
    if data.opt_obj == "t"      % variable tf
        tf = x(end);
    elseif data.opt_obj == "p"  % variable ProbMin
        ProbMin = x(end);
    elseif data.opt_obj == "o"  % fixed tf & ProbMin
    end

    %% convert optimization variables to defender PVA and add to "data"
    [Px,Py,Vx,Vy,Ax,Ay] = Opt2PVA(x,data);

    %% Save "data": paramaters & optimized PVA
    save(results_matfile);
    fprintf("\nSaving results & workspace variables as filename %s .\n", results_matfile);
end




%% PLOTS        **********************************************************

displayConstraintViolations(x,data,lb,ub)

%% Visualize overall engagemnt IC
figure(1);  hold on; sgtitle('Engagement IC Overview');
goA     = plot(AttackersPos_x, AttackersPos_y,'r');
goD     = plot(DefendersPos_x, DefendersPos_y, 'b');
legend([goA(1),goD(1)],'Attacker','Defender','Location', 'best');   % Automatically place the legend in the best location
axis equal;                                                        % Force the plot to have equal scaling for both axes
hold off;
set(gcf, 'Position', [0, 500, plot_width, plot_height]);            % [left, bottom, width, height]
saveas(gcf, fullfile(results_folder, '1_engagement_IC_overview.png'));

%% Visualize optimal vs IC decoy motion
figure(2); hold on; sgtitle('Defender Motions: Optimal vs. Initial');
set(gcf, 'Color', 'w');  % Set the figure background to white
% pcon = plot(PolyCon(:,1), PolyCon(:,2), 'g-', 'LineWidth', 2);  % Green outline of the polygon safe-zone
pcon = fill(PolyCon(:,1), PolyCon(:,2), [0.75, 1, 0.75], 'EdgeColor', 'g', 'LineWidth', 2, 'FaceAlpha', 0.2);  % Light green filled area with green edge
if modelname == "CNmcFull"
    arrow_length = 1.5; % Desired arrow length (0.5 for 20, 1.5 for Full)
else
    arrow_length = 0.5;
end
arrow_head   = 1;   % doesn't do anything
% plot both initial and optimal motions with arrowheads
for i = 1:size(DefendersPos_x, 2)
    colorD = colors(mod(i-1, 10) + 1, :);  % Mod function cycles the index between 1 and 10
    % Plot the initial decoy motion (goD)
    goD(i) = plot(DefendersPos_x(:,i), DefendersPos_y(:,i), '--', 'Color', colorD, 'LineWidth', 1);
    % Add arrowheads for the initial motion
    U = DefendersPos_x(end, i) - DefendersPos_x(end-1, i);
    V = DefendersPos_y(end, i) - DefendersPos_y(end-1, i);
    magnitude = sqrt(U^2 + V^2);
    if magnitude ~= 0
        U_normalized = (U / magnitude) * arrow_length;
        V_normalized = (V / magnitude) * arrow_length;
    else
        U_normalized = 0;
        V_normalized = 0;
    end
    quiver(DefendersPos_x(end, i), DefendersPos_y(end, i), ...
           U_normalized, V_normalized, ...
           'Color', colorD, 'MaxHeadSize', arrow_head, 'LineWidth', 1, ...
           'AutoScale', 'off', 'Alignment', 'head');
    % Plot the optimal motion (goT)
    goT(i) = plot(Px(:,i), Py(:,i), 'Color', colorD, 'LineWidth', 2);  % Use the same color as goD for consistency
    % % Adding arrowheads for optimal motion doesn't work well because with curvy lines arrows don't overlay properly
    % U = Px(end, i) - Px(end-1, i);
    % V = Py(end, i) - Py(end-1, i);
    % magnitude = sqrt(U^2 + V^2);
    % if magnitude ~= 0
    %     U_normalized = (U / magnitude) * arrow_length;
    %     V_normalized = (V / magnitude) * arrow_length;
    % else
    %     U_normalized = 0;
    %     V_normalized = 0;
    % end
    % quiver(Px(end, i), Py(end, i), ...
    %        U_normalized, V_normalized, ...
    %        'Color', colorD, 'MaxHeadSize', arrow_head, 'LineWidth', 2, ...
    %        'AutoScale', 'off', 'Alignment', 'head');
end
hvu = scatter(0, 0, 25^2, 'y', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5); % HVU is yellow
text(0, 0, 'HVU', 'VerticalAlignment','middle', 'HorizontalAlignment','center','FontSize', 10);  % Display HVU label
axis equal;       % Force the plot to have equal scaling for both axes
% Create the legend using the custom lines
blackSolidLine = plot(nan, nan, 'k-', 'LineWidth', 2);   % Black solid line for Optimal Motion
blackDashLine = plot(nan, nan, 'k--', 'LineWidth', 1);   % Black dashed line for Initial Motion
legend([blackSolidLine, blackDashLine, pcon], 'Optimal Motion', 'Initial Motion', 'Allowable OpArea', 'Location', 'best');
axis off;
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
yline(Def_vel_max, '-k', 'Max Velocity', 'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'center', 'FontWeight', 'bold');
yline(Def_vel_min, '--r', 'Min Velocity', 'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'center', 'FontWeight', 'bold');
xlabel('Time [normalized 0 to 1]');
ylabel('Velocity');
ylim([0 1.5*Def_vel_max])
hold off;
set(gcf, 'Position', [0, 0, plot_width, plot_height]); % [left, bottom, width, height]
saveas(gcf, fullfile(results_folder, '3_optimal_defender_velocity.png'));

%% Visualize optimal decoy Acceleration Magnitude
An          = vecnorm(cat(3,Ax,Ay),2,3);                % norm of optimal acceleration
Def_accel_max = Def_vel_max/Def_accel_steps;
plot_lim    = size(An,1);                               % required b/c spline size(A)<Aic
figure(4);  hold on; sgtitle('Optimal Defender Acceleration');
plot(time_steps_normed(1:plot_lim),An);
yline(Def_accel_max, '-k', 'Max Acceleration', 'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'left', 'FontWeight', 'bold');
yline(0.04, '--r', '0.04 Acceleration', 'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'center', 'FontWeight', 'bold');
xlabel('Time [normalized 0 to 1]');
ylabel('Acceleration');
ylim([0 2*Def_accel_max])
hold off;
set(gcf, 'Position', [700, 0, plot_width, plot_height]); % [left, bottom, width, height]
saveas(gcf, fullfile(results_folder, '4_optimal_defender_acceleration.png'));

%% Compare optimal vs initial classifications
fprintf("\n\nCompare optimal vs initial classifications AND evaluate overall improvement.\n");
if data.traj_approx == "s"
    Vx = [Vx;Vx(end,:)]; % will never use last Vx, but required for "CAT" below
    Vy = [Vy;Vy(end,:)];
end
D_opt       = cat(3,Px',Py',Vx',Vy');               % optimized decoy trajectory
% D_ic        = cat(3,DefendersPos_x',...             % initial decoy trajectory
%                     DefendersPos_y',...
%                     DefendersVel_x',...
%                     DefendersVel_y');
y_true_o    = truePredictAllClasses(D_opt, data);   % optimized NN true predictions
% y_true_i    = truePredictAllClasses(D_ic, data);    % initial NN true predictions
delta       = y_true_o - y_true_i;                  % individual improvements
sum_delta   = sum(delta);                             % overall improvement
sum_y_true_o = sum(y_true_o);                    % sum of y_true_o
% sum_y_true_i = sum(y_true_i);                    % sum of y_true_i

% Print results nicely, one per output line
fprintf('y_true_o = '); fprintf('%.4f ', y_true_o); fprintf('\n');
fprintf('y_true_i = '); fprintf('%.4f ', y_true_i); fprintf('\n');
fprintf('delta = ');    fprintf('%.4f ', delta);    fprintf('\n');
fprintf('sum_delta = %.4f\n', sum_delta);
fprintf('sum_y_true_o = %.4f\n', sum_y_true_o);
fprintf('sum_y_true_i = %.4f\n', sum_y_true_i);

diary off;





%*************************************************************************
%*************************    FUNCTIONS   ********************************
%*************************************************************************



%% COST FUNCTION       ***************************************************
function J = costFunc(x,data)
% returns cost based on "optimization objective":
% t = min tf
% p = max ProbMin
% y = max sum(y_true)
% o = other

if data.opt_obj == "t"
    J = x(end);
elseif data.opt_obj == "p"
    J = -x(end);
elseif data.opt_obj == "y"
    [Px,Py,Vx,Vy,~,~] = Opt2PVA(x,data);        % convert optimization variables into defender positions and velocities
    if data.traj_approx == "s"
        Vx = [Vx;Vx(end,:)];                    % will never use last Vx, but required for "CAT" below
        Vy = [Vy;Vy(end,:)];
    end
    D_states = cat(3,Px',Py',Vx',Vy');          % [agent, time, feature Px=1 Py=2 Vx=3 Vy=4]
    J = -sum(truePredictAllClasses(D_states, data));
elseif data.opt_obj == "o"
    % unpack "data" variables required for calculations below
    sim                 = data.sim;
    true_label_idx      = data.true_label_idx;
    net                 = data.net;
    % convert optimization variables into defender positions and velocities
    [Px,Py,Vx,Vy,~,~] = Opt2PVA(x,data);
    if data.traj_approx == "s"
        Vx = [Vx;Vx(end,:)]; % will never use last Vx, but required for "CAT" below
        Vy = [Vy;Vy(end,:)];
    end
    % run sim using optimized defender positions to generate NN inputs = Attacker states
    D_states = cat(3,Px',Py',Vx',Vy');  % [agent, time, feature Px=1 Py=2 Vx=3 Vy=4]
    A_states = sim(D_states, data);
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
% PxCon           = data.PxCon;
% PyCon           = data.PyCon;
PolyCon         = data.PolyCon;                         % polygon constraint vertices
scale_accel     = data.scale_accel;
scale_pos       = data.scale_pos;
if data.opt_obj == "p"  % variable ProbMin
    ProbMin     = x(end);
else                    % constant ProbMin
    ProbMin     = data.ProbMin;
end

% convert optimization variables into defender positions, velocities, and accelerations
[Px,Py,Vx,Vy,Ax,Ay] = Opt2PVA(x,data);

% velocity and acceleration magnitudes squared
V_magnitude_squared = Vx.^2 + Vy.^2;
A_magnitude_squared = Ax.^2 + Ay.^2;
V_magnitude_squared = V_magnitude_squared(:);           % convert to column vector
A_magnitude_squared = A_magnitude_squared(:);           % convert to column vector

%% Normalized constraints
A_normalized = A_magnitude_squared / Def_accel_max^2;   % Normalized acceleration
V_normalized = V_magnitude_squared / Def_vel_max^2;     % Normalized velocity
V_min_normalized = Def_vel_min^2 / Def_vel_max^2;       % Normalized minimum velocity
% Px_normalized = Px(:) / PxCon;                          % Normalized X-Position: convert to column vector
% Py_normalized = Py(:) / PyCon;                          % Normalized Y-Position: convert to column vector

% Acceleration Constraint
C_accel = scale_accel*(A_normalized - 1);               % Acceleration constraint

% Velocity Constraint
if data.Def_vFlag == false
    C_vel = V_normalized - 1;                           % ONLY Max velocity constraint
else
    C_vel = [V_normalized - 1;...                       % Max velocity constraint
             V_min_normalized - V_normalized];          % Min velocity constraint
end

%% Position Constraints

% rectangular boundaries
% C_rect = [Px_normalized - 1;...                          % X-Position constraint
%           Py_normalized - 1];                            % Y-Position constraint

% polygonal safe zone
C_polygon = polygon_constraint(Px(:), Py(:), PolyCon);  % Polygon constraint (normalized by max dim of polygon)

% elliptical no-fly zones
% x_c = data.ellipse_center(1);
% y_c = data.ellipse_center(2);
% a = data.ellipse_a;
% b = data.ellipse_b;
% C_ellipse = ellipse_constraint(Px(:), Py(:), x_c, y_c, a, b);

% Combine all position constraints
% C_pos = scale_pos*[C_rect; C_polygon; C_ellipse]; 
C_pos = scale_pos*C_polygon; 

% NN Prediciton Probability Constraint: ALL FOUR SIM_A_PV --> NN True Label >= ProbMin
if data.opt_obj == "y"
    C_prob = [];                                            % no prediction constraint
else
    if data.traj_approx == "s"
        Vx = [Vx;Vx(end,:)];                                % will never use last Vx, but required for "CAT" below
        Vy = [Vy;Vy(end,:)];
    end
    D_states    = cat(3,Px',Py',Vx',Vy');                   % [agent, time, feature Px=1 Py=2 Vx=3 Vy=4]
    y_true      = truePredictAllClasses(D_states, data);    % 4 true prediction probabilities (double type)
    C_prob      = ProbMin - y_true';                        % column vector of constraints
end

% Combine all constraints
C   = [C_pos; C_vel; C_accel; C_prob];   % inequality
Ceq = [];                       % equality
end



%% DISPLAY CONSTRAINT VIOLATIONS      ************************************
function displayConstraintViolations(x,data,lb,ub)

% unpack "data" variables required for calculations below
Def_vel_min     = data.Def_vel_min;
Def_vel_max     = data.Def_vel_max;
Def_accel_max   = Def_vel_max/data.Def_accel_steps;
% PxCon           = data.PxCon;
% PyCon           = data.PyCon;
PolyCon         = data.PolyCon;                         % polygon constraint vertices
scale_accel     = data.scale_accel;
scale_pos       = data.scale_pos;
if data.opt_obj == "p"  % variable ProbMin
    ProbMin     = x(end);
else                    % constant ProbMin
    ProbMin     = data.ProbMin;
end

% convert optimization variables into defender positions, velocities, and accelerations
[Px,Py,Vx,Vy,Ax,Ay] = Opt2PVA(x,data);

% velocity and acceleration magnitudes squared
V_magnitude_squared = Vx.^2 + Vy.^2;
A_magnitude_squared = Ax.^2 + Ay.^2;
V_magnitude_squared = V_magnitude_squared(:);           % convert to column vector
A_magnitude_squared = A_magnitude_squared(:);           % convert to column vector

%% Normalized constraints
A_normalized = A_magnitude_squared / Def_accel_max^2;   % Normalized acceleration
V_normalized = V_magnitude_squared / Def_vel_max^2;     % Normalized velocity
V_min_normalized = Def_vel_min^2 / Def_vel_max^2;       % Normalized minimum velocity
% Px_normalized = Px(:) / PxCon;                          % Normalized X-Position: convert to column vector
% Py_normalized = Py(:) / PyCon;                          % Normalized Y-Position: convert to column vector

% Acceleration Constraint
C_accel = scale_accel*(A_normalized - 1);               % Acceleration constraint
labels_accel = repmat("Acceleration Violation", numel(A_magnitude_squared), 1);

% Velocity Constraint
if data.Def_vFlag == false
    C_vel = V_normalized - 1;                           % ONLY Max velocity constraint
    labels_vel = repmat("Max Velocity Violation", numel(V_magnitude_squared), 1);
else
    C_vel = [V_normalized - 1;...                       % Max velocity constraint
             V_min_normalized - V_normalized];          % Min velocity constraint
    labels_vel = [repmat("Max Velocity Violation", numel(V_magnitude_squared), 1);...
                  repmat("Min Velocity Violation", numel(V_magnitude_squared), 1)];
end


%% Position Constraints

% rectangular boundaries
% C_rect = [Px_normalized - 1;...                          % X-Position constraint
%           Py_normalized - 1];                            % Y-Position constraint

% polygonal safe zone
C_polygon = polygon_constraint(Px(:), Py(:), PolyCon);    % Polygon constraint (normalized by max dim of polygon)

% Combine all position constraints
% C_pos = scale_pos*[C_rect; C_polygon; C_ellipse];
C_pos = scale_pos*C_polygon; 
labels_pos = repmat("Position Violation", numel(C_pos), 1);

% NN Prediciton Constraint: ALL FOUR SIM_A_PV --> NN True Label >= ProbMin
if data.opt_obj == "y"
    C_prob = [];                                            % no prediction constraint
    labels_prob = "Probability Violation";                  % required for labels concat below
else
    if data.traj_approx == "s"
        Vx = [Vx;Vx(end,:)];                                % will never use last Vx, but required for "CAT" below
        Vy = [Vy;Vy(end,:)];
    end
    D_states    = cat(3,Px',Py',Vx',Vy');                   % [agent, time, feature Px=1 Py=2 Vx=3 Vy=4]
    y_true      = truePredictAllClasses(D_states, data);    % 4 true prediction probabilities (double type)
    C_prob      = ProbMin - y_true';                        % column vector of constraints
    labels_prob = repmat("Probability Violation", numel(C_prob), 1);
end

% Combine all constraints
C   = [C_pos; C_vel; C_accel; C_prob];   % inequality
Ceq = [];                       % equality

%% Display violations
labels = [labels_pos; labels_vel; labels_accel; labels_prob]; % Combine labels for all constraint violations
violations_idx = find(C > 0);  % Find indices of violated constraints
if isempty(violations_idx)
    fprintf('\nNO constraint violations:');
else
    fprintf('\nCONSTRAINT VIOLATIONS\n');
    disp(table(violations_idx, C(violations_idx), labels(violations_idx), ...
         'VariableNames', {'Index', 'Constraint Value', 'Violation Type'}))
end

violated_lb = x < lb;
if any(violated_lb)
    fprintf('\nLOWER BOUND VIOLATIONS');
    table(find(violated_lb), x(violated_lb), lb(violated_lb), ...
         'VariableNames', {'Index', 'Optimized Value', 'Lower Bound'})
else
    fprintf('\nNO lower bound violations');
end

violated_ub = x > ub;
if any(violated_ub)
    fprintf('\nUPPER BOUND VIOLATIONS');
    table(find(violated_ub), x(violated_ub), ub(violated_ub), ...
         'VariableNames', {'Index', 'Optimized Value', 'Upper Bound'})
else
    fprintf('\nNO upper bound violations');
end

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
net                 = data.net;

% Greedy
A_states        = Greedy_A_PV(D_states, data);
A_states_scaled = SqueezeScaleStates(A_states, data);
label           = predict(net, A_states_scaled);
y1              = label(1);

% GreedyPRO
A_states        = GreedyPro_A_PV(D_states, data);
A_states_scaled = SqueezeScaleStates(A_states, data);
label           = predict(net, A_states_scaled);
y2              = label(2);

% Auction
A_states        = Auction_A_PV(D_states, data);
A_states_scaled = SqueezeScaleStates(A_states, data);
label           = predict(net, A_states_scaled);
y3              = label(3);

% AuctionPRO
A_states        = AuctionPro_A_PV(D_states, data);
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
    % motions = ["star", "str", "perL", "perR", "semi", "zero"]
    motions = ["star", "str", "perL", "perR", "semi"]
    num_motions = size(motions,2);
    motion_total = zeros(1,num_motions);
    sim_IC          = @Greedy_AD_PV; % any will work
    % temporarily turn off "do_plot" and "save_movie", then reset at end of function
    original_save_movie_flag    = data.save_movie_flag;
    data.save_movie_flag        = 0;        % not needed
    original_do_plot_flag       = data.do_plot_flag;
    data.do_plot_flag           = 0;        % not needed
    data.results_folder         = [];       % not needed
    % unpack variables for use below
    Att_num                 = data.Att_num;
    Def_num                 = data.Def_num;

    for i = 1:num_motions
        data.Def_motion = motions(i);
        % Generate IC: states [Batch=1,Time,Feature]: Pax,Pay,Vax,Vay...Pdx,Pdy,Vdx,Vdy
        states = sim_IC(data);
        % Parse "states" into features
        states           = squeeze(states); % remove batch dimension
        states_per_agent = length(states(1,:))/(Att_num+Def_num);
        def_idx         = states_per_agent*Att_num + 1; % first defender feature index
        DefendersPos_x  = states(:,def_idx:def_idx + Def_num-1);
        DefendersPos_y  = states(:,def_idx+1*Def_num:def_idx+2*Def_num-1);
        DefendersVel_x  = states(:,def_idx+2*Def_num:def_idx+3*Def_num-1);
        DefendersVel_y  = states(:,def_idx+3*Def_num:def_idx+4*Def_num-1);
        % If defenders are killed (used "FULL" time), replace "NaN" positions
        % Use last alive constant velocity
        DefendersPos_x  = fill_nans_assume_constant_velocity(DefendersPos_x);
        DefendersPos_y  = fill_nans_assume_constant_velocity(DefendersPos_y);
        D_ic= cat(3,DefendersPos_x',...         % initial decoy trajectory
                    DefendersPos_y',...
                    DefendersVel_x',...
                    DefendersVel_y');
        motion_total(i)=sum(truePredictAllClasses(D_ic, data));
    end
    data.do_plot_flag       = original_do_plot_flag; % reset to original setting
    data.save_movie_flag    = original_save_movie_flag; % reset to original setting

    motion_total
    [~,best_motion_idx] = max(motion_total);
    best_motion         = motions(best_motion_idx)
    Def_motion          = best_motion;
end


%% POLYGON SAFE ZONE CONSTRAINT       *************************************
function C_polygon = polygon_constraint(Px, Py, polygon_vertices)
    % Px, Py are the points to check
    % polygon_vertices is an Nx2 array of [x, y] pairs of the polygon's vertices
    vx = polygon_vertices(:,1);
    vy = polygon_vertices(:,2);
    % Calculate signed distance of each point to the polygon
    distance_to_polygon = p_poly_dist(Px, Py, vx, vy);
    % Normalize by the maximum distance between any two polygon vertices
    vertex_distances = pdist2(polygon_vertices, polygon_vertices);
    max_distance = max(vertex_distances(:));
    C_polygon = distance_to_polygon / max_distance; % normalized distance to closest polygon edge
end
function dist = p_poly_dist(px, py, vx, vy)
    % p_poly_dist_vectorized computes the signed distance from multiple points (px, py)
    % to the closest polygon edge for polygon with vertices (vx, vy) in a vectorized manner.
    %
    % Inputs:
    %   px, py: Column vectors of point coordinates.
    %   vx, vy: Column vectors of polygon vertices (x, y).
    %
    % Output:
    %   dist: Signed distance from each point to the closest polygon edge.
    
    % Number of polygon vertices
    N = length(vx);
    
    % Create the start and end points of each edge
    % vx1, vy1 are the starting vertices of each edge
    % vx2, vy2 are the ending vertices of each edge (circularly wrapped)
    vx1 = vx;          % Starting points of each edge
    vy1 = vy;
    vx2 = circshift(vx, -1);  % Shift by -1 to get the ending points of each edge
    vy2 = circshift(vy, -1);

    % Compute the distance from each point (px, py) to each edge in a vectorized way
    dist_to_edges = arrayfun(@(i) point_to_line(px, py, vx1(i), vy1(i), vx2(i), vy2(i)), 1:N, 'UniformOutput', false);
    
    % Convert cell array to matrix and find minimum distance for each point
    dist_to_edges = cell2mat(dist_to_edges);
    dist = min(dist_to_edges, [], 2);  % Get the minimum distance for each point
    
    % Use inpolygon to check if the point is inside or outside
    inside_polygon = inpolygon(px, py, vx, vy);
    dist(inside_polygon) = -dist(inside_polygon);  % Assign negative distances if inside
end
function dist = point_to_line(px, py, x1, y1, x2, y2)
    % point_to_line computes the perpendicular distance from multiple points (px, py) 
    % to the line segment defined by (x1, y1) and (x2, y2).
    % Vector from (x1, y1) to each point (px, py)
    v1x = px - x1;
    v1y = py - y1;
    % Vector from (x1, y1) to (x2, y2)
    v2x = x2 - x1;
    v2y = y2 - y1;
    % Project v1 onto v2
    t = (v1x * v2x + v1y * v2y) / (v2x^2 + v2y^2);
    % Clamp t to the range [0, 1] for each point to get the closest point on the line segment
    t = max(0, min(1, t));
    % Closest points on the line segment
    x_closest = x1 + t * v2x;
    y_closest = y1 + t * v2y;
    % Compute the distance between the points and the closest point on the line
    dist = sqrt((px - x_closest).^2 + (py - y_closest).^2);
end


%% ELLIPSE OFF LIMITS ZONE CONSTRAINT       ******************************
function C_ellipse = ellipse_constraint(Px, Py, x_c, y_c, a, b)
    % Px, Py are the points to check
    % (x_c, y_c) is the center of the ellipse, a and b are the semi-axes lengths
    
    % Calculate the normalized position inside the ellipse
    ellipse_eq = ((Px - x_c).^2) / a^2 + ((Py - y_c).^2) / b^2;
    
    % Constraint: outside ellipse (ellipse_eq > 1) is valid, inside (ellipse_eq <= 1) is invalid
    % To make this a "no-fly zone", we require ellipse_eq <= 1 to be a violation, so:
    C_ellipse = 1 - ellipse_eq;  % must be <= 0 for valid points (outside ellipse)
end


end % ODM function