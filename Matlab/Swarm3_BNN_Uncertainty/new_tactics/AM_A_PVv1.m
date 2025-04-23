function [states] = AM_A_PV(info)
    
    %% UNPACK VARIABLES FOR FUNCTION USE       ***************************
    comp_sys            = info.comp_sys;
    Att_num             = info.Att_num;
    Def_num             = info.Def_num;
    Att_vel_max         = info.Att_vel_max;
    Def_vel_max         = info.Def_vel_max;
    Def_vel_min         = info.Def_vel_min;
    Att_accel_steps     = info.Att_accel_steps;
    Def_accel_flag      = info.Def_accel_flag;
    Def_accel_steps     = info.Def_accel_steps;
    % Att_kill_range      = info.Att_kill_range;
    Def_motion          = info.Def_motion;
    Att_motion          = info.Att_motion;
    do_plot_flag        = info.do_plot_flag;
    save_movie_flag     = info.save_movie_flag;
    Def_final_fraction  = info.Def_final_fraction;
    seed                = info.seed;
    results_folder      = info.results_folder;
    sim_time_steps      = info.sim_time_steps;

    %% Function Init
    rng(seed);      % specifies seed for random number generator
    t=1;            % initialize time step counter
    arrow_scale=10; % movie velocity arror scaling factor
    dv_scale = 0;   % defender vel vector auto-scale (0 = no autoscaling)
    av_scale = 0;   % attacker vel vector auto-scale (0 = no autoscaling)
    if do_plot_flag
        figure(10);
        if save_movie_flag
            % Initialize VideoWriter object to save engagement movie
            % Hamming requires .avi; formats: 'MPEG-4', 'Motion JPEG AVI'
            if comp_sys == "h"
                movie_type = 'Motion JPEG AVI';
            elseif comp_sys == "m"
                movie_type = 'MPEG-4';
            end
            movie = VideoWriter(fullfile(results_folder,'sim_IC_movie'),movie_type);
            movie.FrameRate = 10; % Set frame rate
            open(movie);
        end
    end

    %% Base Init
    def_spread=5; %controls the starting spread of the swarms
    att_spread=5;

    %% Attacker Init (killers)
    Att_pos=40+att_spread*rand(Att_num,2);
    % Att_vel=zeros(Att_num,2); % initial velocity
    Att_accel=zeros(Att_num,2); %  initial acceleration
    Att_accel_max=Att_vel_max/Att_accel_steps; % max acceleration increase per time step

    %% Defender Init (decoys)
    Def_pos=def_spread*rand(Def_num,2); % initial position (x,y)=[0,5]
    theta = pi/2*rand(Def_num,1);  % NOTE: order of using "rand" MATTERS!!
    Def_v_min=Def_vel_min;
    Def_v_max=Def_vel_max;
    Def_v_spread = Def_v_max-Def_v_min;             % velocity spread
    v = Def_v_min+Def_v_spread.*rand(Def_num,1);    % (Nx1)column vector of velocities (constant)
    % Add defender motion options
    if Def_motion == "star"
    elseif Def_motion == "str"
        theta = 45 * pi/180;    % Set angle in radians
    elseif Def_motion == "perL"
        theta = 135 * pi/180;    % Set angle in radians
    elseif Def_motion == "perR"
        theta = -45 * pi/180;    % Set angle in radians
    elseif Def_motion == "semi"
        theta = linspace(-45,135,Def_num)' * pi/180;    % Set angle in radians
    elseif Def_motion == "zero"     % remains still; with acceleration = star starting at v_min up to v_max
        v(:)  = 0.0001;
    end
    Def_vel(:,1)=v.*cos(theta); % (Nx2) x&y velocities (constant)
    Def_vel(:,2)=v.*sin(theta);
    Def_accel_max=Def_vel_max/Def_accel_steps; % max acceleration increase per time step
    
    %% Attacker RANDOM HEADING & VELOCITY (killers)
    % added here to keep "rand" usage above in original order (ORDER MATTERS!)
    Att_theta = pi+pi/2*rand(Att_num,1);        % 90deg fan starting at 180deg (to 270deg)
    Att_v = Att_vel_max.*rand(Att_num,1);       % (Nx1)column vector of velocities (constant)
    % Add attacker constant motion options
    if Att_motion == "star"
    elseif Att_motion == "str"
        Att_theta = 225 * pi/180;   % convert angle deg2rad
    elseif Att_motion == "perL"
        Att_theta = -45 * pi/180;   % convert angle deg2rad
    elseif Att_motion == "perR"
        Att_theta = 135 * pi/180;   % convert angle deg2rad
    elseif Att_motion == "semi"
        Att_theta = linspace(135,315,Att_num)' * pi/180;    % convert angle deg2rad
    elseif Att_motion == "zero"     % remains still; with acceleration = star starting at v_min up to v_max
        Att_v(:)  = 0.0001;
    elseif Att_motion == "left"
        Att_theta = 180 * pi/180;   % convert angle deg2rad
    elseif Att_motion == "down"
        Att_theta = -90 * pi/180;   % convert angle deg2rad
    end
    Att_vel(:,1) = Att_v.*cos(Att_theta);       % (Nx2) x&y velocities (constant)
    Att_vel(:,2) = Att_v.*sin(Att_theta);

    %% Targeting Init
    Def_alive=ones(Def_num,1); % alive (=1) col vector
  
    %% Prepare data to be saved for  NN training
    % fixed error due to unequal Def and Att
    Astates=[Att_pos Att_vel];
    % Dstates=[Def_pos Def_vel];
    % Flatten state vector into pages: features along 3rd dimension; column=timestep; row=sample (seed;run)
    Astates=reshape(Astates,1,1,[]);                % # pages = # agents * # features
    % Dstates=reshape(Dstates,1,1,[]);                % # pages = # agents * # features
    % states=cat(3,Astates,Dstates);
    states=cat(3,Astates);

    %% RUN SIMULATION
    while sum(Def_alive)>Def_final_fraction*Def_num
        
        %% Plot
        if do_plot_flag
            % Decoys are blue
            plot(Def_pos(:,1),Def_pos(:,2),'b.','MarkerSize',16)
            hold on;
            % Attackers are red
            plot(Att_pos(:,1),Att_pos(:,2),'r.','MarkerSize',16)
            % Add velocity vectors
            quiver(Def_pos(:,1), Def_pos(:,2), Def_vel(:,1)*arrow_scale, Def_vel(:,2)*arrow_scale, dv_scale, 'b')
            quiver(Att_pos(:,1), Att_pos(:,2), Att_vel(:,1)*arrow_scale, Att_vel(:,2)*arrow_scale, av_scale, 'r')
            % Add text
            text(0, 40, ['t=', num2str(t)], 'VerticalAlignment','top', 'HorizontalAlignment','left','FontSize', 20)  % Display time step
            % text(0, 35, ['A1_{vel}=', num2str(norm(Att_vel(1,:)))], 'VerticalAlignment','top', 'HorizontalAlignment','left','FontSize', 20)  % Display attacker 1 velocity magnitude
            % text(0, 30, ['D1_{vel}=', num2str(norm(Def_vel(1,:)))], 'VerticalAlignment','top', 'HorizontalAlignment','left','FontSize', 20)  % Display attacker 1 velocity magnitude
            % Set limits to see different motions
            xlim([-5 50])
            ylim([-5 50])
            set(gca,'XTickLabel',[], 'YTickLabel', [])
            if save_movie_flag % Capture the plot as a frame
                frame = getframe(gcf);
                writeVideo(movie, frame);
            end
            pause(.1)
            hold off;
        end

        
        t = t + 1; % Increment time step counter
        % Stop the simulation after sim time steps complete
        if t > sim_time_steps
            break;
        end
        

        %% Update position, velocity, accleration
        % Att_accel = Att_accel-Att_vel/Att_accel_steps;  %  A = max cmd - already attained
        Att_vel =   Att_vel+Att_accel;                  %  V = Vprev + A
        Att_pos =   Att_pos+Att_vel;                    %  P = Pprev + V

        %% Defender Accleration (optional)
        if Def_accel_flag
            vecnorm_vals = vecnorm(Def_vel, 2, 2);                      % Calculate the norms of the velocity vectors
            zero_norm_idx = vecnorm_vals == 0;                          % Find indices where the norm is zero
            Def_accel = zeros(size(Def_vel));                           % Initialize Def_accel: equals zero where the norm is zero
            non_zero_norm_idx = ~zero_norm_idx;                         % Compute acceleration only for non-zero norms
            Def_accel(non_zero_norm_idx,:) = Def_accel_max * Def_vel(non_zero_norm_idx,:) ./ vecnorm_vals(non_zero_norm_idx);
            Def_vel = Def_vel + Def_accel;                              % Vx = vprev + accel
            over_idx = find(vecnorm(Def_vel, 2, 2) > Def_vel_max);      % if V > Vmax, set V = Vmax
            Def_vel(over_idx,:) = Def_vel_max * (Def_vel(over_idx,:) ./ vecnorm(Def_vel(over_idx,:), 2, 2));
        end

        Def_pos = Def_pos+Def_vel;                         %  P = Pprev + V
        
        %% Update 'states' matrix history for output
        newAstate=[Att_pos Att_vel]; %PV
        % newDstate=[Def_pos Def_vel]; %PV
        newAstate=reshape(newAstate,1,1,[]);
        % newDstate=reshape(newDstate,1,1,[]);
        
        % newstate=cat(3,newAstate,newDstate);
        newstate=cat(3,newAstate);
        states=cat(2,states,newstate); %add new column (time step) with pages (updated states)

    end
    if save_movie_flag
        close(movie); % Close the VideoWriter object
    end
end