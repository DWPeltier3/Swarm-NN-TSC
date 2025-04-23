function [states] = SK_AP_A_PV(info)

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
            movie_name = sprintf("SK_4AP_movie_seed_%d", seed);
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

    % %% HVU Init (visualization only)
    % HVU_pos     = [0 0];                            % initial position (x,y)
    % HVU_theta   = pi/4;                             % initial heading [rad]
    % HVU_vel_mag = Def_vel_min;                    % initial speed
    % HVU_vel(:,1)= HVU_vel_mag.*cos(HVU_theta);      % (Nx2) x&y velocities (constant)
    % HVU_vel(:,2)= HVU_vel_mag.*sin(HVU_theta);

    %% Targeting Init
    Def_alive=ones(Def_num,1);
    Att_Velocity_vect=zeros(Att_num,2); % used for ProNav; should it be N_defender????
    Dist=zeros(Att_num,Def_num);
    Disto=zeros(1,Def_num);
    target_num=nan(Att_num,1); %col vector of NaN x Ndef
    totalkilled=0;
    
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
            % % HVU is yellow
            % scatter(HVU_pos(:,1), HVU_pos(:,2), 25^2, 'y', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
            % Add velocity vectors
            quiver(Def_pos(:,1), Def_pos(:,2), Def_vel(:,1)*arrow_scale, Def_vel(:,2)*arrow_scale, dv_scale, 'b')
            quiver(Att_pos(:,1), Att_pos(:,2), Att_vel(:,1)*arrow_scale, Att_vel(:,2)*arrow_scale, av_scale, 'r')
            % quiver(HVU_pos(1), HVU_pos(2), HVU_vel(1)*arrow_scale, HVU_vel(2)*arrow_scale, av_scale, 'k')
            % Add text
            text(0, 40, ['t=', num2str(t)], 'VerticalAlignment','top', 'HorizontalAlignment','left','FontSize', 20)  % Display time step
            % text(HVU_pos(:,1), HVU_pos(:,2), 'HVU', 'VerticalAlignment','middle', 'HorizontalAlignment','center','FontSize', 10)  % Display HVU label
            % text(0, 35, ['A1_{vel}=', num2str(norm(Att_vel(1,:)))], 'VerticalAlignment','top', 'HorizontalAlignment','left','FontSize', 20)  % Display attacker 1 velocity magnitude
            % text(0, 30, ['D1_{vel}=', num2str(norm(Def_vel(1,:)))], 'VerticalAlignment','top', 'HorizontalAlignment','left','FontSize', 20)  % Display attacker 1 velocity magnitude
            % Set limits to see different motions
            % xlim([-30 50])
            % ylim([-30 50])
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

        %% Compute Distances between each attacker and defender
        for att = 1:Att_num
            for defIdx = 1:Def_num
                Dist(att, defIdx) = norm(Att_pos(att,:) - Def_pos(defIdx,:));
            end
        end

        %% Targets distance from origin (prioritize target furthest from origin)
        for defIdx = 1:Def_num
            Disto(defIdx) = norm(Def_pos(defIdx,:));
        end
        
        %% For each attacker: destroy target if within kill range
        for att = 1:Att_num
            if ~isnan(target_num(att))
                if Dist(att, target_num(att)) <= Att_kill_range
                    defAssigned = target_num(att);
                    Def_pos(defAssigned, :) = NaN;
                    Dist(:, defAssigned) = NaN;
                    Def_vel(defAssigned, :) = 0;
                    Def_alive(defAssigned) = 0;
                    totalkilled = totalkilled + 1;
                end
            end
        end
        
        %{
        %% Targeting Algorithm
        iteri=1;
        Dist_check=Dist;
        Disto_check=Disto; %row vector
        target_num=nan(Att_num,1); %col vector of NaN x Ndef
        while iteri<=Att_num && iteri+totalkilled<=Def_num
            maxMatrix=max(Disto_check(:)); %scalar (max attacker distance from origin)
            [~,iter4] = find(Disto_check==maxMatrix); %index of attacker furthest from origin
            minMatrix = min(Dist_check(:,iter4)); %find defender closest to this attacker
            %determine closest attacker/defender combos; exclude attacker once already paired
            [iter,iter2] = find(Dist_check==minMatrix); %find defender and attacker that meets criteria above
            Dist_check(iter,:)=NaN;
            Dist_check(:,iter2)=NaN;
            Disto_check(iter4)=NaN;
            target_num(iter,1)=iter2; %assign attacker (iter2) to defender (iter)
            if Def_alive(iter2,1)==1
                %calculate intercept and point assigned defender acceleration vector to this intercept
                xdiff=Def_pos(iter2,1)-Att_pos(iter,1);
                ydiff=Def_pos(iter2,2)-Att_pos(iter,2);
                c2=Def_vel(iter2,1)^2+Def_vel(iter2,2)^2-Att_vel_max^2;
                c3=2*xdiff*Def_vel(iter2,1)+2*ydiff*Def_vel(iter2,2);
                c4=xdiff^2+ydiff^2;
                % Check the discriminant of the quadratic equation
                discriminant = c3^2 - 4 * c2 * c4;
                if discriminant < 0
                    % Handle the case when discriminant is negative
                    % Adjust heading towards the defender without solving quadratic equation
                    vec = [xdiff, ydiff];
                    vec_norm = norm(vec);
                    if vec_norm ~= 0
                        vec = vec / vec_norm; % Normalize vector
                    end
                    Att_accel(iter, 1) = Att_accel_max * vec(1);
                    Att_accel(iter, 2) = Att_accel_max * vec(2);
                else
                    ts=roots([c2 c3 c4]);
                    ts=max(ts);
                    Att_Velocity_vect(iter,1)=((xdiff+Def_vel(iter2,1)*ts))/ts;
                    Att_Velocity_vect(iter,2)=((ydiff+Def_vel(iter2,2)*ts))/ts;
                    vec=[Att_Velocity_vect(iter,1) Att_Velocity_vect(iter,2)];
                    Att_accel(iter,1)=Att_accel_max*Att_Velocity_vect(iter,1)/norm(vec);
                    Att_accel(iter,2)=Att_accel_max*Att_Velocity_vect(iter,2)/norm(vec);
                end
            end
            iteri=iteri+1;
        end
        
        %pair multiple defenders to one attacker once attackers alive<def
        iter=1;
        while iter<=Att_num
            if isnan(target_num(iter,1))
                [~,iter2] = min(Dist(iter,:));
                target_num(iter,1)=iter2;
                if(min(Dist(iter,:))) < Att_kill_range % % kill closest defender if able
                    Def_pos(iter2,:)=NaN;
                    Dist(:,iter2)=NaN;
                    Def_vel(iter2,:)=0;
                    Def_alive(iter2,1)=0;
                    totalkilled=totalkilled+1;
                end
                if Def_alive(iter2,1)==1 %if can't kill closest defender, move towards it
                    xdiff=Def_pos(iter2,1)-Att_pos(iter,1);
                    ydiff=Def_pos(iter2,2)-Att_pos(iter,2);
                    c2=Def_vel(iter2,1)^2+Def_vel(iter2,2)^2-Att_vel_max^2;
                    c3=2*xdiff*Def_vel(iter2,1)+2*ydiff*Def_vel(iter2,2);
                    c4=xdiff^2+ydiff^2;
                    % Check the discriminant of the quadratic equation
                    discriminant = c3^2 - 4 * c2 * c4;
                    if discriminant < 0
                        % Handle the case when discriminant is negative
                        % Adjust heading towards the defender without solving quadratic equation
                        vec = [xdiff, ydiff];
                        vec_norm = norm(vec);
                        if vec_norm ~= 0
                            vec = vec / vec_norm; % Normalize vector
                        end
                        Att_accel(iter, 1) = Att_accel_max * vec(1);
                        Att_accel(iter, 2) = Att_accel_max * vec(2);
                    else
                        ts=roots([c2 c3 c4]);
                        ts=max(ts);
                        Att_Velocity_vect(iter,1)=((xdiff+Def_vel(iter2,1)*ts))/ts;
                        Att_Velocity_vect(iter,2)=((ydiff+Def_vel(iter2,2)*ts))/ts;
                        vec=[Att_Velocity_vect(iter,1) Att_Velocity_vect(iter,2)];
                        Att_accel(iter,1)=Att_accel_max*Att_Velocity_vect(iter,1)/norm(vec);
                        Att_accel(iter,2)=Att_accel_max*Att_Velocity_vect(iter,2)/norm(vec);
                    end
                end
            end
            iter=iter+1;
        end
        %}

        %% Auction Targeting with ProNav Approximation
        % Reset assignments
        target_num = nan(Att_num, 1);
        Dist_check = Dist;
        Disto_check = Disto;
        for i = 1:Att_num
            if i + totalkilled > Def_num
                break;
            end
            % Find the defender (indexed via the maximum distance from origin)
            [~, attCandidate] = max(Disto_check);
            if isempty(attCandidate)
                break;
            end
            % For that attacker, find its closest defender:
            minVal = min(Dist_check(:, attCandidate));
            [attIdx, defIdx] = find(Dist_check == minVal, 1, 'first');
            % Remove this attacker and defender from further matching:
            Dist_check(attIdx, :) = NaN;
            Dist_check(:, defIdx) = NaN;
            Disto_check(attCandidate) = NaN;
            target_num(attIdx) = defIdx;
            if Def_alive(defIdx) == 1
                % Compute intercept using ProNav approximation:
                xdiff = Def_pos(defIdx,1) - Att_pos(attIdx,1);
                ydiff = Def_pos(defIdx,2) - Att_pos(attIdx,2);
                c2 = Def_vel(defIdx,1)^2 + Def_vel(defIdx,2)^2 - Att_vel_max^2;
                c3 = 2 * xdiff * Def_vel(defIdx,1) + 2 * ydiff * Def_vel(defIdx,2);
                c4 = xdiff^2 + ydiff^2;
                discriminant = c3^2 - 4 * c2 * c4;
                if discriminant < 0
                    vec = [xdiff, ydiff];
                    vec_norm = norm(vec);
                    if vec_norm ~= 0
                        vec = vec / vec_norm;
                    end
                    Att_accel(attIdx, :) = Att_accel_max * vec;
                else
                    ts = roots([c2, c3, c4]);
                    ts = max(ts);
                    Att_Velocity_vect(attIdx,1) = (xdiff + Def_vel(defIdx,1)*ts) / ts;
                    Att_Velocity_vect(attIdx,2) = (ydiff + Def_vel(defIdx,2)*ts) / ts;
                    vec = [Att_Velocity_vect(attIdx,1), Att_Velocity_vect(attIdx,2)];
                    Att_accel(attIdx,1) = Att_accel_max * Att_Velocity_vect(attIdx,1) / norm(vec);
                    Att_accel(attIdx,2) = Att_accel_max * Att_Velocity_vect(attIdx,2) / norm(vec);
                end
            end
        end

        %% Pair remaining attackers to closest defenders if unassigned
        for att = 1:Att_num
            if isnan(target_num(att))
                [~, defIdx] = min(Dist(att, :));
                target_num(att) = defIdx;
                if Dist(att, defIdx) < Att_kill_range
                    Def_pos(defIdx, :) = NaN;
                    Dist(:, defIdx) = NaN;
                    Def_vel(defIdx, :) = 0;
                    Def_alive(defIdx) = 0;
                    totalkilled = totalkilled + 1;
                end
                if Def_alive(defIdx) == 1
                    xdiff = Def_pos(defIdx,1) - Att_pos(att,1);
                    ydiff = Def_pos(defIdx,2) - Att_pos(att,2);
                    c2 = Def_vel(defIdx,1)^2 + Def_vel(defIdx,2)^2 - Att_vel_max^2;
                    c3 = 2 * xdiff * Def_vel(defIdx,1) + 2 * ydiff * Def_vel(defIdx,2);
                    c4 = xdiff^2 + ydiff^2;
                    discriminant = c3^2 - 4 * c2 * c4;
                    if discriminant < 0
                        vec = [xdiff, ydiff];
                        vec_norm = norm(vec);
                        if vec_norm ~= 0
                            vec = vec / vec_norm;
                        end
                        Att_accel(att, :) = Att_accel_max * vec;
                    else
                        ts = roots([c2, c3, c4]);
                        ts = max(ts);
                        Att_Velocity_vect(att,1) = (xdiff + Def_vel(defIdx,1)*ts) / ts;
                        Att_Velocity_vect(att,2) = (ydiff + Def_vel(defIdx,2)*ts) / ts;
                        vec = [Att_Velocity_vect(att,1), Att_Velocity_vect(att,2)];
                        Att_accel(att,1) = Att_accel_max * Att_Velocity_vect(att,1) / norm(vec);
                        Att_accel(att,2) = Att_accel_max * Att_Velocity_vect(att,2) / norm(vec);
                    end
                end
            end
        end

        %% --- Collision Avoidance (Repulsion) ---
        for att = 1:Att_num
            target_vec = Att_accel(att, :) / Att_accel_max; % unit vector from targeting
            repel_vec = [0, 0];
            for other_att = 1:Att_num
                if att ~= other_att
                    diff = Att_pos(att,:) - Att_pos(other_att,:);
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
            % Combine the originally computed (targeting) acceleration with the repulsion
            % current_target = Att_accel(att,:) / Att_accel_max; % unit vector from targeting
            combined_vec = target_vec + repel_gain * repel_vec;
            if norm(combined_vec) > 0
                combined_unit = combined_vec / norm(combined_vec);
            else
                combined_unit = [0, 0];
            end
            % Set the final acceleration with fixed magnitude
            Att_accel(att,:) = Att_accel_max * combined_unit;
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
        
        % HVU_pos = HVU_pos+HVU_vel;

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

