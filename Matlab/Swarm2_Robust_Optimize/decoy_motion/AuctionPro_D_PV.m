function [states] = AuctionPro_D_PV(N_attacker,N_defender,Defender_v_max,do_plot,final_fraction,seed,accel,kill_range,rand_start,Def_motion)

    %% Function Init
    close all; %close all figures
    rng(seed); %specifies seed for random number generator
    
    %% Base Init
    defender_base_pos = [0 0]; %defenders start at origin
    attacker_start_distance = 40; %attackers start away from origin
    def_spread=5; %controls the starting spread of the swarms
    att_spread=5;
    plot_axes_limit = attacker_start_distance*1.25; % Make sure we can see attackers when they start

    %% Defender Init
    if rand_start
        Defender_pos=def_spread*(rand(N_defender,2)-0.5)+defender_base_pos;
    else
        Defender_pos=40+def_spread*rand(N_defender,2);
    end

    Def_v=zeros(N_defender,2); % defender initial velocity

    Def_Acceleration=zeros(N_defender,2); % defender initial acceleration
    steps_to_accel=accel;
    ramp_time=1/steps_to_accel;
    Def_a=Defender_v_max*ramp_time; % defender max acceleration increase each time step

    %% Attacker Init
    if rand_start
        attacker_start_bearing = 2*pi*rand; %attacker swarm: random start bearing between 0 and 2*pi
        attacker_start_pos = attacker_start_distance*[cos(attacker_start_bearing), sin(attacker_start_bearing)]; %Attacker swarm center: PxPy along bearing
        Attacker_pos=att_spread*(rand(N_attacker,2)-0.5)+attacker_start_pos; %attacker agents start position (spread about swarm center)
        theta = pi/2*(rand(N_attacker,1)-0.5)+pi+attacker_start_bearing; %+/- 45 deg opposite start bearing
    else
        Attacker_pos=att_spread*rand(N_attacker,2);
        theta = pi/2*rand(N_attacker,1);
    end

    % v=0.2; %constant velocity [min/max=0.05-0.4]
    Att_v_min=0.05;
    Att_v_max=0.4;
    vm = Att_v_max-Att_v_min; %attacker velocity spread
    v = Att_v_min+vm.*rand(N_attacker,1); %%(Nx1)column vector of attacker velocities (constant)
    Att_vel = zeros(N_attacker, 2);  % Initialize velocity matrix
    % Add defender motion options
    if Def_motion == "star"
    elseif Def_motion == "str"
        theta = 45 * pi/180;    % Set angle in radians
    elseif Def_motion == "perL"
        theta = 135 * pi/180;    % Set angle in radians
    elseif Def_motion == "perR"
        theta = -45 * pi/180;    % Set angle in radians
    elseif Def_motion == "semi"
        theta = linspace(-45,135,N_attacker)' * pi/180;    % Set angle in radians
    elseif Def_motion == "zero"
        v(:)=0.0001;
    end
    Att_vel(:,1) = v .* cos(theta);  % x component of velocity for all defenders
    Att_vel(:,2) = v .* sin(theta);  % y component of velocity for all defenders
    
    %% Targeting Init
    Att_alive=ones(N_attacker,1);
    Def_Velocity_vect=zeros(N_defender,2); % used for ProNav; should it be N_defender????
    Dist=zeros(N_defender,N_attacker);
    Disto=zeros(1,N_attacker);
    target_num=nan(N_defender,1); %col vector of NaN x Ndef
    totalkilled=0;
    
    %% Prepare data to be saved for  NN training
    states=[Defender_pos Def_v]; % initial state matrix: row=defender ONLY; col=states (PV):PxPyVxVy
    % Flatten state vector into pages: features along 3rd dimension; column=timestep; row=sample (seed;run)
    states=reshape(states,1,1,[]); % # pages = # agents * # features

    %% RUN SIMULATION
    while sum(Att_alive)>final_fraction*N_attacker
        
        % Distances between each defender and attacker
        iter=1;
        while iter<=N_defender
            iter2=1;
            while iter2<=N_attacker
                Dist(iter,iter2)=norm([Defender_pos(iter,1) Defender_pos(iter,2)]-[Attacker_pos(iter2,1) Attacker_pos(iter2,2)]);
                iter2=iter2+1;
            end
           iter=iter+1; 
        end

        % Attacker distance from origin (prioritize attackers furthest from origin)
        iter2=1;
        while iter2<=N_attacker
            Disto(1,iter2)=norm([0 0]-[Attacker_pos(iter2,1) Attacker_pos(iter2,2)]);
            iter2=iter2+1;
        end
        
        % For each defender destroy targeted attacker if within kill range
        iter=1;
        while iter <=N_defender % for each defender
            if ~isnan(target_num(iter,1)) %if NOT NaN (defender has an assigned target)
                if(Dist(iter,target_num(iter,1))) <=kill_range %destroy targeted attacker if within minimum range
                    iter2=target_num(iter,1);
                    Attacker_pos(iter2,:)=NaN;
                    Dist(:,iter2)=NaN;
                    Att_vel(iter2,:)=0;
                    Att_alive(iter2,1)=0;
                    totalkilled=totalkilled+1;
                end
            end
            iter=iter+1;
        end

        %% Targeting Algorithm
        iteri=1;
        Dist_check=Dist;
        Disto_check=Disto; %row vector
        target_num=nan(N_defender,1); %col vector of NaN x Ndef
        while iteri<=N_defender && iteri+totalkilled<=N_attacker
            maxMatrix=max(Disto_check(:)); %scalar (max attacker distance from origin)
            [~,iter4] = find(Disto_check==maxMatrix); %index of attacker furthest from origin
            minMatrix = min(Dist_check(:,iter4)); %find defender closest to this attacker
            %determine closest attacker/defender combos; exclude attacker once already paired
            [iter,iter2] = find(Dist_check==minMatrix); %find defender and attacker that meets criteria above
            Dist_check(iter,:)=NaN;
            Dist_check(:,iter2)=NaN;
            Disto_check(iter4)=NaN;
            target_num(iter,1)=iter2; %assign attacker (iter2) to defender (iter)
            if Att_alive(iter2,1)==1
                %calculate intercept and point assigned defender acceleration vector to this intercept
                xdiff=Attacker_pos(iter2,1)-Defender_pos(iter,1);
                ydiff=Attacker_pos(iter2,2)-Defender_pos(iter,2);
                c2=Att_vel(iter2,1)^2+Att_vel(iter2,2)^2-Defender_v_max^2;
                c3=2*xdiff*Att_vel(iter2,1)+2*ydiff*Att_vel(iter2,2);
                c4=xdiff^2+ydiff^2;
                ts=roots([c2 c3 c4]);
                ts=max(ts);
                Def_Velocity_vect(iter,1)=((xdiff+Att_vel(iter2,1)*ts))/ts;
                Def_Velocity_vect(iter,2)=((ydiff+Att_vel(iter2,2)*ts))/ts;
                vec=[Def_Velocity_vect(iter,1) Def_Velocity_vect(iter,2)];
                Def_Acceleration(iter,1)=Def_a*Def_Velocity_vect(iter,1)/norm(vec);
                Def_Acceleration(iter,2)=Def_a*Def_Velocity_vect(iter,2)/norm(vec);
            end
            iteri=iteri+1;
        end
        
        %pair multiple defenders to one attacker once attackers alive<def
        iteri=1;
        while iteri<=N_defender
            if target_num(iteri,1)==0 || isnan(target_num(iteri,1))
                [~,iter2] = min(Dist(iteri,:));
                target_num(iteri,1)=iter2;
                if Att_alive(iter2,1)==1
                    xdiff=Attacker_pos(iter2,1)-Defender_pos(iteri,1);
                    ydiff=Attacker_pos(iter2,2)-Defender_pos(iteri,2);
                    c2=Att_vel(iter2,1)^2+Att_vel(iter2,2)^2-Defender_v_max^2;
                    c3=2*xdiff*Att_vel(iter2,1)+2*ydiff*Att_vel(iter2,2);
                    c4=xdiff^2+ydiff^2;
                    ts=roots([c2 c3 c4]);
                    ts=max(ts);
                    Def_Velocity_vect(iteri,1)=((xdiff+Att_vel(iter2,1)*ts))/ts;
                    Def_Velocity_vect(iteri,2)=((ydiff+Att_vel(iter2,2)*ts))/ts;
                    vec=[Def_Velocity_vect(iteri,1) Def_Velocity_vect(iteri,2)];
                    Def_Acceleration(iteri,1)=Def_a*Def_Velocity_vect(iteri,1)/norm(vec);
                    Def_Acceleration(iteri,2)=Def_a*Def_Velocity_vect(iteri,2)/norm(vec);
                end
            end
            iteri=iteri+1;
        end
        
        %% Update position, velocity, accleration
        Def_Acceleration(:,1)=Def_Acceleration(:,1)-Def_v(:,1)*ramp_time;   % defender Ax = max cmd - already attained
        Def_Acceleration(:,2)=Def_Acceleration(:,2)-Def_v(:,2)*ramp_time;   % defender Ay
        Def_v(:,1)=Def_v(:,1)+Def_Acceleration(:,1);                        % defender Vx = vprev + accel
        Def_v(:,2)=Def_v(:,2)+Def_Acceleration(:,2);                        % defender Vy
        Defender_pos(:,1)=Defender_pos(:,1)+Def_v(:,1);                     % defender Px = xprev+xvel
        Defender_pos(:,2)=Defender_pos(:,2)+Def_v(:,2);                     % defender Py

        Attacker_pos(:,1)=Attacker_pos(:,1)+Att_vel(:,1);                   % attacker Px = xprev+xvel
        Attacker_pos(:,2)=Attacker_pos(:,2)+Att_vel(:,2);                   % attacker Py = yprev+yvel

        %% Update 'states' matrix history for output
        newstate=[Defender_pos Def_v]; %PV of defenders only
        newstate=reshape(newstate,1,1,[]);
        states=cat(2,states,newstate); %add new column (time step) with pages (updated states)

        %% Plot
        if do_plot
            plot(Defender_pos(:,1),Defender_pos(:,2),'r.','MarkerSize',16)
            hold on;
            plot(Attacker_pos(:,1),Attacker_pos(:,2),'b.','MarkerSize',16)
            if rand_start
                xlim(plot_axes_limit*[-1 1])
                ylim(plot_axes_limit*[-1 1])
            else
                xlim([-30 50])
                ylim([-30 50])
            end
            set(gca,'XTickLabel',[], 'YTickLabel', [])
            pause(.1)
            hold off;
        end
        
    end
end

