function [states] = Auction_D_PV(defender, attacker, defender_v,do_plot,kill_pro,seed,accel,kill_range,rand_start, Def_motion)
    
    %% Function Init
    close all; %close all figures
    rng(seed); %specifies seed for random number generator

    %% Base Init
    defender_base_pos = [0 0]; %defenders start at origin
    attacker_start_distance = 40; %attackers start away from origin
    def_spread=5; %controls the starting spread of the swarms
    att_spread=5;
    plot_axes_limit = attacker_start_distance*1.25; % Make sure we can see attackers when they start

    %% Defender Init (killers)
    N_att=defender; % # defenders
    if rand_start
        Att_pos=def_spread*(rand(N_att,2)-0.5)+defender_base_pos; % defender initial position
    else
        Att_pos=40+def_spread*rand(N_att,2);
    end
    
    a_vel=zeros(N_att,2); % defender initial velocity

    avel=zeros(N_att,2); % defender initial acceleration
    steps_to_accel=accel;
    ramp_time=1/steps_to_accel;
    Att_a=defender_v*ramp_time; %defender acceleration each time step
    
    %% Attacker Init
    N_def=attacker; %# attackers
    if rand_start
        attacker_start_bearing = 2*pi*rand; %attacker swarm: random start bearing between 0 and 2*pi
        attacker_start_pos = attacker_start_distance*[cos(attacker_start_bearing), sin(attacker_start_bearing)]; %Attacker swarm center: PxPy along bearing
        Def_pos=att_spread*(rand(N_def,2)-0.5)+attacker_start_pos; %attacker agents start position (spread about swarm center)
        theta = pi/2*(rand(N_def,1)-0.5)+pi+attacker_start_bearing; %+/- 45 deg opposite start bearing
    else
        Def_pos=att_spread*rand(N_def,2); % attacker initial position (x,y)=[0,5]
        theta = pi/2*rand(N_def,1); %default motion is "stardust"
    end
    
    % v=0.2; %constant velocity [min/max=0.05-0.4]
    Att_v_min=0.05;
    Att_v_max=0.4;
    vm = Att_v_max-Att_v_min; %attacker velocity spread
    v = Att_v_min+vm.*rand(N_def,1); %%(Nx1)column vector of attacker velocities (constant)
    
    vel = zeros(N_def, 2);  % Initialize velocity matrix
    % Add defender motion options
    if Def_motion == "star"
    elseif Def_motion == "str"
        theta = 45 * pi/180;    % Set angle in radians
    elseif Def_motion == "perL"
        theta = 135 * pi/180;    % Set angle in radians
    elseif Def_motion == "perR"
        theta = -45 * pi/180;    % Set angle in radians
    elseif Def_motion == "semi"
        theta = linspace(-45,135,N_def)' * pi/180;    % Set angle in radians
    elseif Def_motion == "zero"
        v(:)=0.0001;
    end
    vel(:,1) = v .* cos(theta);  % x component of velocity for all defenders
    vel(:,2) = v .* sin(theta);  % y component of velocity for all defenders
    
    %% Targeting Init
    Def_alive=ones(N_def,1); % attacker alive (=1) col vector
    Dist=zeros(N_att,N_def); % distance matrix (row=defender, col=attacker)

    %% Prepare data to be saved for  NN training
    states=[Att_pos a_vel]; % initial state matrix: row=defender ONLY; col=states (PVA):PxPyVxVy
    % Flatten state vector into pages: features along 3rd dimension; column=timestep; row=sample (seed;run)
    states=reshape(states,1,1,[]); % # pages = # agents * # features

    %% RUN SIMULATION
    while sum(Def_alive)>kill_pro*N_def %while #att alive > #def*constant

        % Distances between each defender and attacker
        iter=1; %init counter
        while iter<=N_att %calculate distance between every attacker and defender
            iter2=1;
            while iter2<=N_def
                Dist(iter,iter2)=norm([Att_pos(iter,1) Att_pos(iter,2)]-[Def_pos(iter2,1) Def_pos(iter2,2)]);
                iter2=iter2+1;
            end
           iter=iter+1;
        end

        %% Targeting Algorithm
        iter=1;
        target_num=zeros(N_att,1); %col vector = # def; defender->attacker assignments
        Dist_check=Dist; %matrix (row=defender, col=attacker)
        while iter<=N_att %for each defender
            minMatrix = min(Dist_check(:)); %find closest attacker
            [row,col] = find(Dist_check==minMatrix); %match defender to closest attacker
            Dist_check(row,:)=NaN; %prevent multiple defenders matched to same attacker
            Dist_check(:,col)=NaN;
            if(min(Dist(row,:))) <kill_range %can defender kill closest attacker
                Def_pos(col,1)=NaN;
                Def_pos(col,2)=NaN;
                vel(col,1)=0;
                vel(col,2)=0;
                Def_alive(col,1)=0;
            end
            if Def_alive(col,1)==1 %if can't kill closest attacker, move towards it and ASSIGN to defender
                xdiff=Def_pos(col,1)-Att_pos(row,1);
                ydiff=Def_pos(col,2)-Att_pos(row,2);
                vec=[xdiff ydiff];
                avel(row,1)=Att_a*vec(1)/norm(vec);
                avel(row,2)=Att_a*vec(2)/norm(vec);
            end
            target_num(row,1)=col; %assign attacker to defender
            iter=iter+1;
        end

        % pair multiple defenders to one attacker once attackers<def
        iter=1; %defender
        while iter<=N_att %for each defender NOT ASSIGNED an attacker
            if target_num(iter,1)==0 % defender NOT ASSIGNED an attacker
                [~,I] = min(Dist(iter,:)); %find closest attacker (index) to defender
                if(min(Dist(iter,:))) <kill_range % if defender can kill closest attacker, kill it
                    Def_pos(I,1)=NaN; %kill closest attacker
                    Def_pos(I,2)=NaN;
                    vel(I,1)=0;
                    vel(I,2)=0;
                    Def_alive(I,1)=0;
                end
                if Def_alive(I,1)==1 %if can't kill closest attacker, move towards it and ASSIGN to defender
                    xdiff=Def_pos(I,1)-Att_pos(iter,1); %x&y dist to closest attacker
                    ydiff=Def_pos(I,2)-Att_pos(iter,2);
                    vec=[xdiff ydiff];
                    avel(iter,1)=Att_a*vec(1)/norm(vec); %defender xvelocity towards attacker (unit vector)
                    avel(iter,2)=Att_a*vec(2)/norm(vec); %defender yvelocity towards attacker (unit vector)
                end
                target_num(iter,1)=I; % ASSIGN closest alive attacker to defender
            end
            iter=iter+1;
        end

        %% Update position, velocity, accleration
        avel(:,1)=avel(:,1)-a_vel(:,1)*ramp_time;   % defender Ax = max cmd - already attained
        avel(:,2)=avel(:,2)-a_vel(:,2)*ramp_time;   % defender Ay
        a_vel(:,1)=a_vel(:,1)+avel(:,1);            % defender Vx = vprev + accel
        a_vel(:,2)=a_vel(:,2)+avel(:,2);            % defender Vy
        Att_pos(:,1)=Att_pos(:,1)+a_vel(:,1);       % defender Px = xprev+xvel
        Att_pos(:,2)=Att_pos(:,2)+a_vel(:,2);       % defender Py

        Def_pos(:,1)=Def_pos(:,1)+vel(:,1);         % attacker Px = xprev+xvel
        Def_pos(:,2)=Def_pos(:,2)+vel(:,2);         % attacker Py = yprev+yvel
        
        
        %% Update 'states' matrix history for output
        newstate=[Att_pos a_vel]; %defender ONLY
        newstate=reshape(newstate,1,1,[]);
        states=cat(2,states,newstate); %add new column (time step) with pages (updated states)
        
        %% Plot
        if do_plot
            % switched colors for ONR brief
            plot(Def_pos(:,1),Def_pos(:,2),'b.','MarkerSize',16)
            hold on;
            plot(Att_pos(:,1),Att_pos(:,2),'r.','MarkerSize',16)
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

