function [states] = AuctionPro_A_PV(D_states, info)
    
    %% UNPACK VARIABLES FOR FUNCTION USE       ***************************
    N_attacker          = info.Att_num;
    N_defender          = info.Def_num;
    Att_vel_max         = info.Att_vel_max;
    accel_steps         = info.Att_accel_steps;
    kill_range          = info.Att_kill_range;
    Def_final_fraction  = info.Def_final_fraction;
    seed                = info.seed;
    sim_time_steps      = info.sim_time_steps;
    
    %% Function Init
    rng(seed);      %specifies seed for random number generator
    t=1;            % initialize time step counter
    
    %% Base Init
    att_spread=5;

    %% Attacker Init (killers)
    Att_pos=40+att_spread*rand(N_attacker,2);
    Att_vel=zeros(N_attacker,2); % initial velocity
    Att_accel=zeros(N_attacker,2); %  initial acceleration
    Att_accel_max=Att_vel_max/accel_steps; % max acceleration increase per time step

    %% Targeting Init
    Def_alive=ones(N_defender,1);
    Att_Velocity_vect=zeros(N_attacker,2); % used for ProNav; should it be N_defender????
    Dist=zeros(N_attacker,N_defender);
    Disto=zeros(1,N_defender);
    target_num=nan(N_attacker,1); %col vector
    totalkilled=0;
    
    %% Prepare data to be saved for  NN training
    states=[Att_pos Att_vel]; %initial state matrix: rows=Att; col=states (PV):PxPyVxVy
    % Flatten state vector into pages: features along 3rd dimension; column=timestep; row=sample (seed;run)
    states=reshape(states,1,1,[]); % # pages = # agents * # features

    %% RUN SIMULATION
    % while sum(Def_alive)>Def_final_fraction*N_defender
    while t <= sim_time_steps
        
        %% Parse "D_states" for defender position & velocity
            % use previous time step to sync properly with attacker time step
            % ie. when attacker on t=1, defender must be on t=1
            % Update defender positions and velocities ONLY IF ALIVE
        alive_defenders = Def_alive == 1; % Logical indexing to find defenders that are still alive
        Def_pos(alive_defenders, :) = [D_states(alive_defenders, t, 1), D_states(alive_defenders, t, 2)]; % row=agent; col=PxPy
        Def_vel(alive_defenders, :) = [D_states(alive_defenders, t, 3), D_states(alive_defenders, t, 4)]; % row=agent; col=VxVy
        
        % %% Plot for debugging
        % arrow_scale=10; % movie velocity arror scaling factor
        % dv_scale = 0;   % defender vel vector auto-scale (0 = no autoscaling)
        % av_scale = 0;   % attacker vel vector auto-scale (0 = no autoscaling)
        % % Decoys are blue
        % plot(Def_pos(:,1),Def_pos(:,2),'b.','MarkerSize',16)
        % hold on;
        % % Attackers are red
        % plot(Att_pos(:,1),Att_pos(:,2),'r.','MarkerSize',16)
        % % Add velocity vectors
        % quiver(Def_pos(:,1), Def_pos(:,2), Def_vel(:,1)*arrow_scale, Def_vel(:,2)*arrow_scale, dv_scale, 'b')
        % quiver(Att_pos(:,1), Att_pos(:,2), Att_vel(:,1)*arrow_scale, Att_vel(:,2)*arrow_scale, av_scale, 'r')
        % % Add text
        % text(0, 40, ['t=', num2str(t)], 'VerticalAlignment','top', 'HorizontalAlignment','left','FontSize', 20)  % Display time step
        % text(0, 35, ['A1_{vel}=', num2str(norm(Att_vel(1,:)))], 'VerticalAlignment','top', 'HorizontalAlignment','left','FontSize', 20)  % Display attacker 1 velocity magnitude
        % text(0, 30, ['D1_{vel}=', num2str(norm(Def_vel(1,:)))], 'VerticalAlignment','top', 'HorizontalAlignment','left','FontSize', 20)  % Display defender 1 velocity magnitude
        % % Set limits to see different motions
        % xlim([-30 50])
        % ylim([-30 50])
        % set(gca,'XTickLabel',[], 'YTickLabel', [])
        % pause(.1)
        % hold off;

        % Stop the simulation after sim time steps complete
        t = t + 1; % Increment time step counter
        if t > sim_time_steps
            break;
        end

        % Distances between each defender and attacker
        iter=1;
        while iter<=N_attacker
            iter2=1;
            while iter2<=N_defender
                Dist(iter,iter2)=norm([Att_pos(iter,1) Att_pos(iter,2)]-[Def_pos(iter2,1) Def_pos(iter2,2)]);
                iter2=iter2+1;
            end
           iter=iter+1; 
        end

        % Attacker distance from origin (prioritize attackers furthest from origin)
        iter2=1;
        while iter2<=N_defender
            Disto(1,iter2)=norm([0 0]-[Def_pos(iter2,1) Def_pos(iter2,2)]);
            iter2=iter2+1;
        end
        
        % For each attacker destroy target if within kill range
        iter=1;
        while iter <=N_attacker % for each attacker
            if ~isnan(target_num(iter,1)) %if NOT NaN (attacker has an assigned target)
                if(Dist(iter,target_num(iter,1))) <=kill_range %destroy targeted defender if within minimum range
                    iter2=target_num(iter,1);
                    Def_pos(iter2,:)=NaN;
                    Dist(:,iter2)=NaN;
                    Def_vel(iter2,:)=0;
                    Def_alive(iter2,1)=0;
                    totalkilled=totalkilled+1;
                end
            end
            iter=iter+1;
        end

        %% Targeting Algorithm
        iteri=1;
        Dist_check=Dist;
        Disto_check=Disto; %row vector
        target_num=nan(N_attacker,1); %col vector of NaN x Ndef
        while iteri<=N_attacker && iteri+totalkilled<=N_defender
            maxMatrix=max(Disto_check(:)); %scalar (distance of defender furthest from origin)
            [~,iter4] = find(Disto_check==maxMatrix); %index of defender furthest from origin
            minMatrix = min(Dist_check(:,iter4)); %scalar (distance of attacker closest to priority defender)
            %determine closest attacker/defender combos; exclude defender once already paired
            [iter,iter2] = find(Dist_check==minMatrix); %find indicies of attacker & defender that meet criteria above
            Dist_check(iter,:)=NaN;
            Dist_check(:,iter2)=NaN;
            Disto_check(iter4)=NaN;
            target_num(iter,1)=iter2; %assign defender (iter2) to attacker (iter)
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
        while iter<=N_attacker
            % if target_num(iteri,1)==0 || isnan(target_num(iteri,1))
            if isnan(target_num(iter,1))
                [~,iter2] = min(Dist(iter,:)); %find closest defender (index) to attacker
                target_num(iter,1)=iter2; % ASSIGN closest defender to attacker
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
            end
            iter=iter+1;
        end
        
        %% Update position, velocity, accleration
        Att_accel(:,1)=Att_accel(:,1)-Att_vel(:,1)/accel_steps;     %  Ax = max cmd - already attained
        Att_accel(:,2)=Att_accel(:,2)-Att_vel(:,2)/accel_steps;     %  Ay
        Att_vel(:,1)=Att_vel(:,1)+Att_accel(:,1);                   %  Vx = vprev + accel
        Att_vel(:,2)=Att_vel(:,2)+Att_accel(:,2);                   %  Vy
        Att_pos(:,1)=Att_pos(:,1)+Att_vel(:,1);                     %  Px = xprev + xvel
        Att_pos(:,2)=Att_pos(:,2)+Att_vel(:,2);                     %  Py

        %% Update 'states' matrix history for output
        newstate=[Att_pos Att_vel];
        newstate=reshape(newstate,1,1,[]);
        states=cat(2,states,newstate); %add new column (time step) with pages (updated states)

    end
end

