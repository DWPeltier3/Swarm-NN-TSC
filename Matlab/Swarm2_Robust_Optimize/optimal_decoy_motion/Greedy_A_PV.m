function [states] = Greedy_A_PV(D_states, info)

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
    rng(seed);  % specifies seed for random number generator
    t=1;        % initialize time step counter

    %% Base Init
    att_spread=5;

    %% Attacker Init (killers)
    Att_pos=40+att_spread*rand(N_attacker,2);
    Att_vel=zeros(N_attacker,2); % initial velocity
    Att_accel=zeros(N_attacker,2); %  initial acceleration
    Att_accel_max=Att_vel_max/accel_steps; % max acceleration increase per time step
    
    %% Targeting Init
    Def_alive=ones(N_defender,1); % alive (=1) col vector
  
    %% Prepare data to be saved for  NN training
    states=[Att_pos Att_vel]; %initial state matrix: rows=Att ; col=states (PV):PxPyVxVy
    % Flatten state vector into pages: features (PV) along 3rd dimension (pages); column=timestep; row=sample (seed;run)
    states=reshape(states,1,1,[]); % # pages = # agents * # features
    
    %% RUN SIMULATION
    % while sum(Def_alive)>Def_final_fraction*N_defender
    while t <= sim_time_steps
        
        %% Parse "D_states" into defender position & velocity
            % use previous time step to sync defenders properly with attacker time step
            % ie. when attacker on t=1, defender must be on t=1
            % Update defender positions and velocities ONLY IF ALIVE
        alive_defenders = Def_alive == 1; % Logical indexing to find defenders that are still alive
        Def_pos(alive_defenders, :) = [D_states(alive_defenders, t, 1), D_states(alive_defenders, t, 2)]; % row=agent; col=PxPy
        Def_vel(alive_defenders, :) = [D_states(alive_defenders, t, 3), D_states(alive_defenders, t, 4)]; % row=agent; col=VxVy

        % Stop the simulation after sim time steps complete
        t = t + 1; % Increment time step counter
        if t > sim_time_steps
            break;
        end

        % Distances between each defender and attacker
        Dist=zeros(N_attacker,N_defender); %distance matrix (row=attacker, col=defender)
        iter=1;
        while iter<=N_attacker %calculate distance between every attacker and defender
            iter2=1;
            while iter2<=N_defender
                Dist(iter,iter2)=norm([Att_pos(iter,1) Att_pos(iter,2)]-[Def_pos(iter2,1) Def_pos(iter2,2)]);
                iter2=iter2+1;
            end
           iter=iter+1; 
        end

        iter=1;
        while iter<=N_attacker % for each attacker
            [~,I] = min(Dist(iter,:)); % find closest defender
            if(min(Dist(iter,:))) <kill_range % kill closest defender if able
                Def_pos(I,1)=NaN;
                Def_pos(I,2)=NaN;
                Dist(:,I)=NaN;
                Def_vel(I,1)=0;
                Def_vel(I,2)=0;
                Def_alive(I,1)=0;
            end
            if Def_alive(I,1)==1 %if can't kill closest defender, move towards it
                xdiff=Def_pos(I,1)-Att_pos(iter,1);
                ydiff=Def_pos(I,2)-Att_pos(iter,2);
                vec=[xdiff ydiff];
                Att_accel(iter,1)=Att_accel_max*vec(1)/norm(vec); %update Vx
                Att_accel(iter,2)=Att_accel_max*vec(2)/norm(vec); %update Vy
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