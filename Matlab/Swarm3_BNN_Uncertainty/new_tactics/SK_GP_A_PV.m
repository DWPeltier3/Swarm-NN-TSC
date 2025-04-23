function [states] = SK_GP_A_PV(info)
    
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
            movie_name = sprintf("SK_2GP_movie_seed_%d", seed);
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
    Att_Velocity_vect=zeros(Att_num,2); % used for ProNav
    
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

        %% Compute distances between each attacker and each defender
        Dist = zeros(Att_num, Def_num); % rows: attackers, cols: defenders
        for att = 1:Att_num
            for def = 1:Def_num
                Dist(att,def) = norm(Att_pos(att,:) - Def_pos(def,:));
            end
        end
        
        %% Compute acceleration for each attacker (targeting + collision avoidance)
        for att = 1:Att_num % for each attacker
            [~,closest_def] = min(Dist(att,:)); % find closest defender
            
            % kill closest defender if able
            if(min(Dist(att,:))) <Att_kill_range
                Def_pos(closest_def)=NaN;
                Def_pos(closest_def,2)=NaN;
                Dist(:,closest_def)=NaN;
                Def_vel(closest_def,1)=0;
                Def_vel(closest_def,2)=0;
                Def_alive(closest_def,1)=0;
            end
            
            %if can't kill closest defender, move towards it
            if Def_alive(closest_def,1)==1
                xdiff=Def_pos(closest_def,1)-Att_pos(att,1);
                ydiff=Def_pos(closest_def,2)-Att_pos(att,2);
                c2=Def_vel(closest_def,1)^2+Def_vel(closest_def,2)^2-Att_vel_max^2;
                c3=2*xdiff*Def_vel(closest_def,1)+2*ydiff*Def_vel(closest_def,2);
                c4=xdiff^2+ydiff^2;

                % Check the discriminant of the quadratic equation
                discriminant = c3^2 - 4 * c2 * c4;
                if discriminant < 0
                    % Handle the case when discriminant is negative
                    % Adjust heading towards the defender without solving quadratic equation
                    vec = [xdiff, ydiff];
                    if norm(vec) > 0
                        target_vec = vec / vec_norm; % Normalize vector
                    else
                        target_vec = [0, 0];
                    end
                else
                    ts=roots([c2 c3 c4]);
                    ts=max(ts);
                    Att_Velocity_vect(att,1)=((xdiff+Def_vel(closest_def,1)*ts))/ts;
                    Att_Velocity_vect(att,2)=((ydiff+Def_vel(closest_def,2)*ts))/ts;
                    vec=[Att_Velocity_vect(att,1) Att_Velocity_vect(att,2)];
                    if norm(vec) > 0
                        target_vec = vec / norm(vec);
                    else
                        target_vec = [0, 0];
                    end
                end
                % --- Collision Avoidance (Repulsion) ---
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

                % --- Combine Targeting and Repulsion ---
                combined_vec = target_vec + repel_gain * repel_vec;
                if norm(combined_vec) > 0
                    combined_unit = combined_vec / norm(combined_vec);
                else
                    combined_unit = [0, 0];
                end

                % Set final acceleration (fixed magnitude in the combined direction)
                Att_accel(att,:) = Att_accel_max * combined_unit;
            end
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