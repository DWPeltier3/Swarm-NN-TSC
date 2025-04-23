function [states] = SK_A_A_PV(info)
    
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
    Att_kill_range      = info.Att_kill_range;
    Def_motion          = info.Def_motion;
    do_plot_flag        = info.do_plot_flag;
    save_movie_flag     = info.save_movie_flag;
    Def_final_fraction  = info.Def_final_fraction;
    seed                = info.seed;
    results_folder      = info.results_folder;
    sim_time_steps      = info.sim_time_steps;
    min_dist            = info.min_dist;
    repel_gain          = info.repel_gain;

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
            movie_name = sprintf("SK_3A_movie_seed_%d", seed);
            movie = VideoWriter(fullfile(results_folder,movie_name),movie_type);
            movie.FrameRate = 10; % Set frame rate
            open(movie);
        end
    end

    %% Base Init
    def_spread=5; %controls the starting spread of the swarms
    att_spread=5;

    %% Attacker Init (killers)
    Att_pos=40+att_spread*rand(Att_num,2);
    Att_vel=zeros(Att_num,2); % initial velocity
    Att_accel=zeros(Att_num,2); %  initial acceleration
    Att_accel_max=Att_vel_max/Att_accel_steps; % max acceleration increase per time step
    
    %% Defender Init (decoys)
    Def_pos=def_spread*rand(Def_num,2); % initial position (x,y)=[0,5]
    theta = pi/2*rand(Def_num,1);  % NOTE: order of using "rand" MATTERS!!
    Def_vel_spread = Def_vel_max-Def_vel_min; % velocity spread
    v = Def_vel_min+Def_vel_spread.*rand(Def_num,1); % (Nx1)column vector of velocities (constant)
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
    elseif Def_motion == "zero"
        v(:)=0.0001;
    end
    Def_vel(:,1)=v.*cos(theta); % (Nx2) x&y velocities (constant)
    Def_vel(:,2)=v.*sin(theta);
    Def_accel_max=Def_vel_max/Def_accel_steps; % max acceleration increase per time step
    
    %% Targeting Init
    Def_alive=ones(Def_num,1); % attacker alive (=1) col vector

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
            if save_movie_flag
                % Capture the plot as a frame
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

        %% Compute Distances (Rows: Attacker; Columns: Defender)
        Dist = zeros(Att_num, Def_num);
        for att = 1:Att_num
            for defIdx = 1:Def_num
                Dist(att, defIdx) = norm(Att_pos(att, :) - Def_pos(defIdx, :));
            end
        end
        
        %{
        %% Targeting Algorithm
        iter=1;
        target_num=nan(Att_num,1); %col vector = # def; defender->attacker assignments
        Dist_check=Dist; %matrix (row=defender, col=attacker)
        while iter<=Att_num %for each attacker
            minMatrix = min(Dist_check(:)); %find closest defender
            [row,col] = find(Dist_check==minMatrix); %match attacker to closest defender
            Dist_check(row,:)=NaN; %prevent multiple attackers matched to same defender
            Dist_check(:,col)=NaN;
            if(min(Dist(row,:))) <Att_kill_range %can attacker kill closest defender
                Def_pos(col,:)=NaN;
                Dist(:,col)=NaN;
                Def_vel(col,:)=0;
                Def_alive(col,1)=0;
            end
            if Def_alive(col,1)==1 %if can't kill closest defender, move towards it and ASSIGN to attacker
                xdiff=Def_pos(col,1)-Att_pos(row,1);
                ydiff=Def_pos(col,2)-Att_pos(row,2);
                vec=[xdiff ydiff];
                Att_accel(row,1)=Att_accel_max*vec(1)/norm(vec);
                Att_accel(row,2)=Att_accel_max*vec(2)/norm(vec);
            end
            target_num(row,1)=col; %assign defender to killer
            iter=iter+1;
        end

        % pair multiple defenders to one attacker once defenders<attackers
        iter=1; %defender
        while iter<=Att_num %for each attacker NOT ASSIGNED an defender
            if isnan(target_num(iter,1)) % attacker NOT ASSIGNED an defender
                [~,iter2] = min(Dist(iter,:)); %find closest defender (index) to attacker
                target_num(iter,1)=iter2; % ASSIGN closest alive defender to attacker
                if(min(Dist(iter,:))) < Att_kill_range % if attacker can kill closest defender, kill it
                    Def_pos(iter2,:)=NaN; %kill closest defender
                    Def_vel(iter2,:)=0;
                    Def_alive(iter2,1)=0;
                end
                if Def_alive(iter2,1)==1 %if can't kill closest defender, move towards it
                    xdiff=Def_pos(iter2,1)-Att_pos(iter,1); %x&y dist to closest defender
                    ydiff=Def_pos(iter2,2)-Att_pos(iter,2);
                    vec=[xdiff ydiff];
                    Att_accel(iter,1)=Att_accel_max*vec(1)/norm(vec); % attacker xvelocity towards defender (unit vector)
                    Att_accel(iter,2)=Att_accel_max*vec(2)/norm(vec); % attacker yvelocity towards defender (unit vector)
                end
            end
            iter=iter+1;
        end
        %}

        %% Auction Targeting Assignment
        % Assign defenders to attackers using auction-style matching.
        target_num = nan(Att_num, 1);  % defender assignment for each attacker
        Dist_check = Dist;             % copy for processing assignments
        
        for iter = 1:Att_num
            % Find the minimum distance in the remaining matrix
            [min_val, linearInd] = min(Dist_check(:));
            if isnan(min_val)
                break;  % no more valid assignments
            end
            [attIdx, defIdx] = ind2sub(size(Dist_check), linearInd);
            % Remove this defender from further consideration:
            Dist_check(attIdx, :) = NaN;
            Dist_check(:, defIdx) = NaN;
            % If the defender is within kill range, mark it as killed.
            if min(Dist(attIdx, :)) < Att_kill_range
                Def_pos(defIdx, :) = NaN;
                Def_vel(defIdx, :) = 0;
                Def_alive(defIdx) = 0;
            end
            % If the defender is still alive, assign it to the attacker.
            if Def_alive(defIdx) == 1
                xdiff = Def_pos(defIdx, 1) - Att_pos(attIdx, 1);
                ydiff = Def_pos(defIdx, 2) - Att_pos(attIdx, 2);
                target_vec = [xdiff, ydiff];
                if norm(target_vec) > 0
                    target_vec = target_vec / norm(target_vec);
                end
                Att_accel(attIdx, :) = Att_accel_max * target_vec;
            end
            target_num(attIdx) = defIdx;
        end

        % For attackers not assigned a defender, assign the closest alive defender.
        for att = 1:Att_num
            if isnan(target_num(att))
                [~, defIdx] = min(Dist(att, :));
                target_num(att) = defIdx;
                if min(Dist(att, :)) < Att_kill_range
                    Def_pos(defIdx, :) = NaN;
                    Def_vel(defIdx, :) = 0;
                    Def_alive(defIdx) = 0;
                end
                if Def_alive(defIdx) == 1
                    xdiff = Def_pos(defIdx, 1) - Att_pos(att, 1);
                    ydiff = Def_pos(defIdx, 2) - Att_pos(att, 2);
                    target_vec = [xdiff, ydiff];
                    if norm(target_vec) > 0
                        target_vec = target_vec / norm(target_vec);
                    end
                    Att_accel(att, :) = Att_accel_max * target_vec;
                end
            end
        end

        %% Collision Avoidance (Repulsion)
        % For each attacker, compute a repulsion vector from other nearby attackers.
        for att = 1:Att_num
            target_vec = Att_accel(att, :) / Att_accel_max; % unit vector from targeting
            repel_vec = [0, 0];
            for other_att = 1:Att_num
                if att ~= other_att
                    diff = Att_pos(att, :) - Att_pos(other_att, :);
                    d = norm(diff);
                    if (d < min_dist) && (d > 0)

                        % % Repel directly away from other attacker
                        % repel_vec = repel_vec + (min_dist - d) * (diff/d); % Weight repulsion by (min_dist - d)

                        % Repel away from other attacker, perpendicular to target_vector
                        v = diff / d;  % repel element unit vector
                        proj = dot(v, target_vec) * target_vec; % Compute projection of repel element onto target_vec
                        v_perp = v - proj; % Perpendicular component of repel element
                        repel_vec = repel_vec + (min_dist - d) * v_perp;

                    end
                end
            end
            % Combine the original targeting acceleration (normalized) with the repulsion.
            % current_target_accel = Att_accel(att, :) / Att_accel_max; % unit vector from targeting
            combined_vec = target_vec + repel_gain * repel_vec;
            if norm(combined_vec) > 0
                combined_unit = combined_vec / norm(combined_vec);
            else
                combined_unit = [0, 0];
            end
            % Set the final acceleration with fixed magnitude.
            Att_accel(att, :) = Att_accel_max * combined_unit;
        end

        %% Update position, velocity, accleration
        Att_accel = Att_accel-Att_vel/Att_accel_steps;  %  A = max cmd - already attained
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

