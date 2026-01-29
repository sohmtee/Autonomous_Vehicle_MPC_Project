clc; clear; close all;

%% Load and Process Trajectory Data
csvFile = 'C:\Users\soguchie\OneDrive - purdue.edu\ECE 699\Summer 2025\RoadRunner Projects\RoadRunner Project 1\Exports\SingaporeIntersection_design_varying_velocities2_for_hybrid_models\Sedan3_3a.csv';
traj = readmatrix(csvFile);

time_raw = traj(:,1);
X_raw = traj(:,2);
Y_raw = traj(:,3);
yaw_deg = traj(:,5);
vx_raw = traj(:,8);

X_path = X_raw;
Y_path = Y_raw;
psi_path = deg2rad(yaw_deg);

% Create uniform time grid
dt = 0.05;
t_uniform = (0:dt:time_raw(end))';

% Interpolate all signals
vx_ref = interp1(time_raw, vx_raw, t_uniform, 'linear');
X_ref = interp1(time_raw, X_path, t_uniform, 'pchip');
Y_ref = interp1(time_raw, Y_path, t_uniform, 'pchip');
psi_ref = interp1(time_raw, psi_path, t_uniform, 'pchip');
psi_ref = unwrap(psi_ref);

vx_ref = smooth(vx_ref, 25); % was 15

fprintf('Starting simulation at t=%.2fs (v=%.2f m/s)\n', t_uniform(1), vx_ref(1));

%% Load UIO Gains for Dynamic Model
fprintf('Loading observer gain schedule for dynamic model...\n');
load('observer_gain_schedule.mat')

fprintf('Loaded observer gains: vx_grid has %d points\n', length(vx_grid));
fprintf('  vx range: [%.1f, %.1f] m/s\n', min(vx_grid), max(vx_grid));
fprintf('  L_schedule size: [%d, %d, %d]\n', size(L_schedule));

%% Vehicle Parameters
vehicle_params = struct();
vehicle_params.Cf = 30000;
vehicle_params.Cr = 30000;
vehicle_params.l1 = 1.2;
vehicle_params.l2 = 1.22;
vehicle_params.L = vehicle_params.l1 + vehicle_params.l2;
vehicle_params.m = 1280;
vehicle_params.Iz = 2500;

nn_kin = 2; % no of states [y, psi]
m1_kin = 2; % no of inputs [v, delta]
pp_kin = 2; % no of outputs [y, psi]

nn_dyn = 4; % no of states [vy, psi, r, Y]
m1_dyn = 1; % no of inputs [delta]
pp_dyn = 2; % no of outputs [psi, Y]

%% Controller Parameters
% Switching thresholds (with hysteresis)
v_switch_high = 2;    % Switch to dynamic at this speed
v_switch_low = 1;     % Switch back to kinematic at this speed

% MPC Parameters
Np = 50;

% Cost weights
Q_kinematic = diag([2000, 800]);      % [y, psi]
Q_dynamic_psi = 1000;
Q_dynamic_y = 3000;
R_kinematic = diag([0, 2000]);      % [dv, ddelta]
R_dynamic = 3000;                     % delta_u

% Constraints
delta_rate_max = deg2rad(50);
delta_step_max = delta_rate_max * dt;

% Optimization options
opts = optimoptions('fmincon','Algorithm','sqp','Display','off', ...
    'MaxIterations',100,'OptimalityTolerance',1e-6);

%% sparse error recovery setup
tau = 15;

% attack simulation params
sensor_attack_prob = 0.01;
actuator_attack_prob = 0.01;

fprintf('--- system configuration --\n');
fprintf('Switching: Kinematic (vx <= %.1f) <-> Dynamic (vx >= %.1f)\n', ...
        v_switch_low, v_switch_high);
fprintf('Observation window: tau = %d samples\n', tau);
fprintf('Sensor attack probability: %.1f%%\n', sensor_attack_prob*100);
fprintf('Actuator attack probability: %.1f%%\n', actuator_attack_prob*100);

%% Initialize simulation
Nsim = length(t_uniform) - Np;

% Unified state: [y, psi, vy, r]
x_true = [Y_ref(1); psi_ref(1); 0; 0];
x_estimated = x_true;

% UIO observer state (only used in dynamic mode)
z_uio = zeros(4,1);  % Will be initialized at first switch to dynamic

% Controller state
current_model = 'kinematic';
u_prev = [vx_ref(98); 0];   % [v, delta]

% Storage for sparse recovery
y_init_meas = [Y_ref(1); psi_ref(1)]; % Initial measurement [Y, psi] % 1/26
% vecyc_kin = zeros(tau*pp_kin, 1);
vecyc_kin = repmat(y_init_meas, tau, 1); % 1/26
vecuc_kin = zeros(m1_kin*(tau-1), 1);  % Inputs can remain zero (assuming start from rest/steady)   
vecyc_dyn = zeros(tau*pp_dyn, 1);
vecuc_dyn = zeros(m1_dyn*(tau-1), 1);

% FIX: Track steps since last model switch
% steps_since_switch = tau + 1;  % Start as "converged"
steps_since_switch = 0;

% Logging arrays
x_hist = zeros(4, Nsim);
x_est_hist = zeros(4, Nsim);
u_cmd_hist = zeros(2, Nsim);
u_act_hist = zeros(2, Nsim);
model_hist = cell(Nsim, 1);
sensor_error_hist = zeros(Nsim, 2);
actuator_error_hist = zeros(Nsim, 2);
estimation_error_hist = zeros(Nsim, 4);
delta_u_hist = zeros(Nsim, 1);

fprintf('\n--- starting hybrid simulation with sparse error recovery --\n');

%% Main simulation loop
for k = 1:Nsim
    if mod(k, 100) == 0
        fprintf('Step %d/%d (%.1f%%), Model: %s, Speed: %.2f m/s\n', ...
                k, Nsim, 100*k/Nsim, current_model, vx_ref(k));
    end

    % % Debug output
    % if mod(k, 2) == 0 || any(abs(x_true(1:2)) > 500)  % Check only y and psi
    %     fprintf('DEBUG k=%d: x_true=[Y=%.1f, psi=%.2f deg, vy=%.2f, r=%.2f]\n', ...
    %             k, x_true(1), rad2deg(x_true(2)), x_true(3), x_true(4));
    %     fprintf('       u_prev=[%.2f, %.2f deg], vx_ref(k)=%.2f\n', ...
    %             u_prev(1), rad2deg(u_prev(2)), vx_ref(k));
    % 
    %     if any(abs(x_true(1:2)) > 500)
    %         warning('State showing large values at k=%d! Possible divergence.', k);
    %     end
    % end

    vx_current = vx_ref(k);

    %% Model switching logic (with hysteresis)
    previous_model = current_model;

    if strcmp(current_model, 'kinematic')
        if vx_current >= v_switch_high
            current_model = 'dynamic';
            fprintf('Switched to DYNAMIC at t=%.2f s (v=%.2f m/s)\n', t_uniform(k), vx_current);

            % Convert unified state to dynamic order [vy, psi, r, Y]
            x_dyn_current = [x_estimated(3); x_estimated(2); x_estimated(4); x_estimated(1)];
            
            % Get gains
            M_switch = interpGain(vx_current, vx_grid, M_schedule);
            Cd = [0 1 0 0; 0 0 0 1];
            
            % Initialize z_uio Smartly (Uncommented your logic)
            y_init = Cd * x_dyn_current;
            z_uio = x_dyn_current - M_switch * y_init; 

            fprintf('  Initialized UIO: z_uio = [%.2f, %.2f, %.2f, %.2f]\n', ...
                z_uio(1), z_uio(2), z_uio(3), z_uio(4));
            
            % === FIX: Handover Kinematic Buffer to Dynamic Buffer ===
            % Kinematic y is [Y; psi]. Dynamic y is [psi; Y]. We must swap rows.
            y_hist = reshape(vecyc_kin, 2, tau);       % Unpack history
            y_hist_flipped = flipud(y_hist);           % Swap Y and psi
            vecyc_dyn = y_hist_flipped(:);             % Repack for Dynamic mode
            
            % Kinematic u is [v; delta]. Dynamic u is [delta].
            u_hist = reshape(vecuc_kin, 2, tau-1);     % Unpack history
            u_hist_delta = u_hist(2, :);               % Extract only steering
            vecuc_dyn = u_hist_delta(:);               % Repack
            
            % Keep estimator ON
            steps_since_switch = tau + 1; 
            fprintf('  Buffer translated. Estimator remains ACTIVE.\n');
        end

    else  % dynamic model
        if vx_current <= v_switch_low
            current_model = 'kinematic';
            fprintf('Switched to KINEMATIC at t=%.2f s (v=%.2f m/s)\n', t_uniform(k), vx_current);
            
            % === FIX: Handover Dynamic Buffer to Kinematic Buffer ===
            % Dynamic y is [psi; Y]. Kinematic y is [Y; psi]. Swap rows.
            y_hist = reshape(vecyc_dyn, 2, tau);
            y_hist_flipped = flipud(y_hist);
            vecyc_kin = y_hist_flipped(:);
            
            % Dynamic u is [delta]. Kinematic u is [v; delta].
            % We need to reconstruct 'v' history from reference or previous estimates
            delta_hist = reshape(vecuc_dyn, 1, tau-1);
            
            % Reconstruct velocity history from reference (approximate)
            idx_start = max(1, k - (tau - 1));
            idx_end = k - 1;
            v_hist_approx = vx_ref(idx_start:idx_end)';
            
            % Handle edge case at very start of sim (unlikely for switch, but safe)
            if length(v_hist_approx) < (tau-1)
                pad = tau - 1 - length(v_hist_approx);
                v_hist_approx = [repmat(v_hist_approx(1), 1, pad), v_hist_approx];
            end
            
            u_hist_new = [v_hist_approx; delta_hist];
            vecuc_kin = u_hist_new(:);
            
            % Keep estimator ON
            steps_since_switch = tau + 1; 
            fprintf('  Buffer translated. Estimator remains ACTIVE.\n');
        end
    end
    
    % Increment transition counter
    steps_since_switch = steps_since_switch + 1;

    %% Sparse error recovery and state estimation
    if strcmp(current_model, 'kinematic')
        %% Kinematic model with sparse recovery
        
        % FIX 3: Better velocity clamping
        v_current = max(vx_ref(k), 0.5);  % Higher minimum for stability
        delta_current = u_prev(2);
        psi_current = x_estimated(2);
        
        [Ad_kin, Bd_kin] = linearizeKinematic(v_current, psi_current, ...
                                                delta_current, vehicle_params.L, dt);

        % Build error matrices
        Cd_kin = eye(2);
        [V_kin, Ob_kin] = buildErrorMatrices(Ad_kin, Bd_kin, Cd_kin, tau, m1_kin, pp_kin);
        Q2_kin = null(Ob_kin');

        % Simulate sensor attacks
        % y_sensor = Cd_kin * x_true(1:2);
        % Rho = diag([rand > sensor_attack_prob, rand > sensor_attack_prob]);
        % y_corrupted = Rho * y_sensor;


        % added 1/28
        y_sensor = Cd_kin * x_true(1:2);
        % prevent attacks during critical transition
        if steps_since_switch > tau
            Rho = diag([rand > sensor_attack_prob, rand > sensor_attack_prob]);
            y_corrupted = Rho * y_sensor;
        else
            % during buffer refill: no attacks allowed
            y_corrupted = y_sensor;
        end

        % CRITICAL FIX: Only do sparse recovery if buffer is valid
        if steps_since_switch > tau
            % Sparse error recovery
            vecyc_kin = [y_corrupted; vecyc_kin(1:end-pp_kin)];
            vecYc_kin = vecyc_kin(1:tau*pp_kin);

            F_kin = V_kin * diag(vecuc_kin);
            Y_vec_kin = vecYc_kin - V_kin * vecuc_kin;
            Omega_kin = [eye(tau*pp_kin) F_kin];
            Zt_kin = Q2_kin' * Y_vec_kin;
            Wt_kin = Q2_kin' * Omega_kin;

            % L1 optimization
            A_lp = [Wt_kin -Wt_kin];
            f = ones(1, 2*(tau*pp_kin+(tau-1)*m1_kin));
            lb_lp = zeros(1, 2*(tau*pp_kin+(tau-1)*m1_kin));
            options_lp = optimoptions('linprog','Display','none');
            
            Et = linprog(f, [], [], A_lp, Zt_kin, lb_lp, [], options_lp);

            if ~isempty(Et)
                Et = Et(1:(tau*pp_kin+(tau-1)*m1_kin),:) - Et((tau*pp_kin+(tau-1)*m1_kin+1):end,:);
                est_sensor_error = Et(1:pp_kin);
            else
                est_sensor_error = zeros(pp_kin,1);
            end
        else
            % During transition: just update buffer, don't estimate errors
            vecyc_kin = [y_corrupted; vecyc_kin(1:end-pp_kin)];
            est_sensor_error = zeros(pp_kin,1);
            if mod(steps_since_switch, 5) == 0
                fprintf('  Kinematic buffer filling: %d/%d steps\n', steps_since_switch, tau);
            end
        end

        % Corrected measurement
        y_corrected_kin = y_corrupted - est_sensor_error;
        x_estimated = [y_corrected_kin; x_estimated(3:4)];

        sensor_error_hist(k, 1:2) = (y_corrupted - y_sensor)';
    
    else
        %% Dynamic model with UIO and sparse recovery

        [A_c, B_c] = computeVehicleModel(vx_current, vehicle_params);
        C_c = [0 1 0 0; 0 0 0 1]; % measure [psi, Y]
        D_c = zeros(2,1);
        
        sysd = c2d(ss(A_c, B_c, C_c, D_c), dt);
        Ad_dyn = sysd.A;
        Bd_dyn = sysd.B;
        Cd_dyn = sysd.C;

        % Interpolate UIO gains
        L_current = interpGain(vx_current, vx_grid, L_schedule);
        M_current = interpGain(vx_current, vx_grid, M_schedule);

        P_t = eye(4) - M_current*Cd_dyn;

        % Build error matrices
        [V_dyn, Ob_dyn] = buildErrorMatrices(Ad_dyn, Bd_dyn, Cd_dyn, tau, m1_dyn, pp_dyn);
        Q2_dyn = null(Ob_dyn');

        % Simulate sensor attacks
        x_dyn_true = [x_true(3); x_true(2); x_true(4); x_true(1)]; % [vy, psi, r, Y]
        y_sensor = Cd_dyn * x_dyn_true;
        Rho = diag([rand > sensor_attack_prob, rand > sensor_attack_prob]);
        y_corrupted = Rho * y_sensor;

        % CRITICAL FIX: Only do sparse recovery if buffer is valid
        if steps_since_switch > tau
            % Sparse error recovery
            vecyc_dyn = [y_corrupted; vecyc_dyn(1:end-pp_dyn)];
            vecYc_dyn = vecyc_dyn(1:tau*pp_dyn);

            F_dyn = V_dyn * diag(vecuc_dyn);
            Y_vec_dyn = vecYc_dyn - V_dyn * vecuc_dyn;
            Omega_dyn = [eye(tau*pp_dyn) F_dyn];
            Zt_dyn = Q2_dyn' * Y_vec_dyn;
            Wt_dyn = Q2_dyn' * Omega_dyn;

            % L1 optimization
            A_lp = [Wt_dyn -Wt_dyn];
            f = ones(1, 2*(tau*pp_dyn+(tau-1)*m1_dyn));
            lb_lp = zeros(1, 2*(tau*pp_dyn+(tau-1)*m1_dyn));
            options_lp = optimoptions('linprog','Display','none');
            
            Et = linprog(f, [], [], A_lp, Zt_dyn, lb_lp, [], options_lp);

            if ~isempty(Et)
                Et = Et(1:(tau*pp_dyn+(tau-1)*m1_dyn),:) - ...
                     Et((tau*pp_dyn+(tau-1)*m1_dyn+1):(2*(tau*pp_dyn+(tau-1)*m1_dyn)),:);
                est_sensor_error = Et(1:pp_dyn);
            else
                est_sensor_error = zeros(pp_dyn,1);
            end
        else
            % During transition: just update buffer, don't estimate errors
            vecyc_dyn = [y_corrupted; vecyc_dyn(1:end-pp_dyn)];
            est_sensor_error = zeros(pp_dyn,1);
            if mod(steps_since_switch, 5) == 0
                fprintf('  Dynamic buffer filling: %d/%d steps\n', steps_since_switch, tau);
            end
        end

        % Corrected measurement
        y_corrected_dyn = y_corrupted - est_sensor_error;

        % UIO state estimation
        x_dyn_estimated = z_uio + M_current * y_corrected_dyn;

        % Convert to unified state [y, psi, vy, r]
        x_estimated = [x_dyn_estimated(4);     % Y
                       x_dyn_estimated(2);     % psi
                       x_dyn_estimated(1);     % vy
                       x_dyn_estimated(3)];    % r

        sensor_error_hist(k, 1:2) = (y_corrupted - y_sensor)';
    end

    %% MPC control
    if strcmp(current_model, 'kinematic')
        [u_command, x_pred] = solveKinematicMPC(x_estimated, u_prev, k, ...
                                                X_ref, Y_ref, psi_ref, vx_ref, dt, Np, ...
                                                vehicle_params.L, Q_kinematic, ...
                                                R_kinematic, delta_step_max, opts);
    else
        [u_command, x_pred] = solveDynamicMPC(x_estimated, u_prev, k, ...
                                            Y_ref, psi_ref, vx_ref, dt, Np, ...
                                            vehicle_params, Q_dynamic_psi, Q_dynamic_y, R_dynamic, ...
                                            delta_step_max, opts);
    end

    %% Simulate actuator attacks
    Lambda = diag([1, rand > actuator_attack_prob]);
    u_actuated = Lambda * u_command;

    %% Update true plant
    if strcmp(current_model, 'kinematic')
        % Kinematic dynamics
        x_kin_true = x_true(1:2);   % [y, psi]
        x_kin_next = simulateKinematicStep(x_kin_true, u_actuated, dt, vehicle_params.L);
        x_true = [x_kin_next; x_true(3:4)];
    else
        % Dynamic dynamics
        x_dyn_true = [x_true(3); x_true(2); x_true(4); x_true(1)];
        [A_c, B_c] = computeVehicleModel(vx_current, vehicle_params);
        plant_dynamics = @(t, x_state) A_c * x_state + B_c * u_actuated(2);
        [~, x_solution] = ode23s(plant_dynamics, [0 dt], x_dyn_true);
        x_dyn_next = x_solution(end, :)';
    
        x_true = [x_dyn_next(4);    % Y
                  x_dyn_next(2);    % psi
                  x_dyn_next(1);    % vy
                  x_dyn_next(3)];   % r
    end

    %% Update UIO observer (only in dynamic model)
    if strcmp(current_model, 'dynamic')
    % Use corrected measurement for innovation
    y_expected = Cd_dyn * (z_uio + M_current * y_corrected_dyn);
    innovation = y_corrected_dyn - y_expected;
    
    % UIO update with corrected measurement
    z_uio = P_t * (Ad_dyn*z_uio + Ad_dyn*M_current*y_corrected_dyn + Bd_dyn*u_command(2)) + ...
            L_current * innovation;
    end

    delta_u_current = u_command(2) - u_prev(2);

    %% Store results
    x_hist(:, k) = x_true;
    x_est_hist(:, k) = x_estimated;
    u_cmd_hist(:, k) = u_command;
    u_act_hist(:, k) = u_actuated;
    model_hist{k} = current_model;
    delta_u_hist(k) = delta_u_current;
    actuator_error_hist(k, :) = (u_actuated - u_command)';
    estimation_error_hist(k, :) = (x_true - x_estimated)';

    %% Update for next iteration
    u_prev = u_command;

    if strcmp(current_model, 'kinematic')
        vecuc_kin = [u_command; vecuc_kin(1:end-2)];
    else
        vecuc_dyn = [u_command(2); vecuc_dyn(1:end-1)];
    end
end

fprintf('--- simulation complete ---\n');

%% plot results
plotHybridResults(t_uniform(1:Nsim), x_hist, x_est_hist, u_cmd_hist, u_act_hist, ...
                    model_hist, X_ref(1:Nsim), Y_ref(1:Nsim), psi_ref(1:Nsim), ...
                    vx_ref(1:Nsim), sensor_error_hist, actuator_error_hist, ...
                    estimation_error_hist, delta_u_hist);


%% == helper functions == %
function [Ad, Bd] = linearizeKinematic(v_r, psi_r, delta_r, L, dt)
    % v_r = max(v_r, 0.3);  % Minimum velocity
    delta_r = max(min(delta_r, deg2rad(85)), deg2rad(-85));
    Ac = [0,  v_r*cos(psi_r);
          0,  0];
%     Bc = [sin(psi_r), 0;
%           0, v_r/(L*cos(delta_r)^2)];
    % I CHANGED THIS ON 1/1/26
    Bc = [sin(psi_r), 0;
          tan(delta_r)/L, v_r/(L*cos(delta_r)^2)];
    sysd = c2d(ss(Ac,Bc,eye(2),zeros(2,2)), dt, 'zoh');
    Ad = sysd.A; 
    Bd = sysd.B;
end    

function [A_c, B_c] = computeVehicleModel(vx, params)
    Cf = params.Cf; Cr = params.Cr;
    l1 = params.l1; l2 = params.l2;
    m = params.m; Iz = params.Iz;
    
    if vx < 0.1, vx = 0.1; end
    
    A_c = [-(2*Cf + 2*Cr)/(m*vx), 0, -(2*Cf*l1 - 2*Cr*l2)/(m*vx) - vx, 0;
           0, 0, 1, 0;
           -(2*l1*Cf - 2*l2*Cr)/(Iz*vx), 0, -(2*l1^2*Cf + 2*l2^2*Cr)/(Iz*vx), 0;
           -1, -vx, 0, 0];
    
    B_c = [(2*Cf)/m; 0; (2*l1*Cf)/Iz; 0];
end

function L_interp = interpGain(vx_current, vx_grid, L_schedule)
    nn = size(L_schedule, 1);
    pp = size(L_schedule, 2);
    L_interp = zeros(nn, pp);
    vx_clamped = max(min(vx_current, max(vx_grid)), min(vx_grid));
    
    for i = 1:nn
        for j = 1:pp
            L_ij_schedule = squeeze(L_schedule(i, j, :));
            L_interp(i, j) = interp1(vx_grid, L_ij_schedule, vx_clamped, 'pchip');
        end
    end
end

function [V, Ob] = buildErrorMatrices(Ad, Bd, Cd, tau, m1, pp)
    temp1 = [];
    for i=1:tau-1
        temp1 = [temp1 Cd*Ad^(i-1)*Bd];
    end
    V = [temp1];
    temp3 = temp1;
    for j=1:tau-1
        temp3 = [zeros([pp m1]) temp3(:,1:(tau-2)*m1)];
        V = [V; temp3];
    end
    
    Ob = [];
    for i = 1:tau
        Ob = [Ob; Cd*Ad^(tau-i)];
    end
end

% function [u_opt, x_next] = solveKinematicMPC(x, u_prev, k, X_ref, Y_ref, psi_ref, ...
%                                            vx_ref, dt, Np, L, Q, R, delta_step_max, opts)
%     x_kin = x(1:2);
% 
%     % get reference horizon
%     idx = k:min(k+Np-1, length(X_ref));
%     if length(idx) < Np
%         pad_length = Np - length(idx);
%         Y_horizon = [Y_ref(idx); repmat(Y_ref(end), pad_length, 1)];
%         psi_horizon = [psi_ref(idx); repmat(psi_ref(end), pad_length, 1)];
%         v_horizon = [vx_ref(idx); repmat(vx_ref(end), pad_length, 1)];
%     else
%         Y_horizon = Y_ref(idx);
%         psi_horizon = psi_ref(idx);
%         v_horizon = vx_ref(idx);
%     end
% 
%     Xr = [Y_horizon'; psi_horizon'];
% 
%     % feedforward steering reference
%     psi_dot = gradient(psi_horizon, dt);
%     psi_dot = medfilt1(psi_dot, min(3, length(psi_dot)));
%     delta_ref_horizon = atan(L * psi_dot ./ v_horizon);
%     ur = [v_horizon'; delta_ref_horizon'];
% 
%     % added this
%     v_current = max(vx_ref(k), 0.5);
%     delta_current = u_prev(2);
%     [Ad, Bd] = linearizeKinematic(v_current, x_kin(2), delta_current, L, dt);
% 
%     % Ad_horizon = cell(Np, 1);
%     % Bd_horizon = cell(Np, 1);
%     % 
%     % for i = 1:Np
%     %     v_step = max(v_horizon(i), 0.5);    % clamp velocity
%     %     psi_step = psi_horizon(i);
%     %     delta_step = delta_ref_horizon(i);
%     %     [Ad_horizon{i}, Bd_horizon{i}] = linearizeKinematic(v_step, psi_step, delta_step, L, dt);
%     % end
% 
%     nu = 2;
%     du0 = zeros(nu*Np, 1);
%     lb = repmat([0; -delta_step_max], Np, 1);
%     ub = repmat([0;  delta_step_max], Np, 1);
% 
%     Q_bar = kron(eye(Np), Q);
%     R_bar = kron(eye(Np), R);
% 
%     % cost_fun = @(du_vec) kinematicMPCCost(du_vec, x_kin, u_prev, Xr, ur, ...
%     %                                      Ad_horizon, Bd_horizon, Q_bar, R_bar, Np, 2, nu);
% 
%     cost_fun = @(du_vec) kinematicMPCCost(du_vec, x_kin, u_prev, Xr, ur, ...
%                                          Ad, Bd, Q_bar, R_bar, Np, 2, nu);
% 
%     du_opt = fmincon(cost_fun, du0, [], [], [], [], lb, ub, [], opts);
%     u_opt = u_prev + du_opt(1:nu);
% 
%     % u_opt = [vx_ref(k); u_prev(2) + du_opt(2)];
% 
%     x_next_kin = simulateKinematicStep(x_kin, u_opt, dt, L);
%     x_next = [x_next_kin; x(3:4)];
% end

function [u_opt, x_next] = solveKinematicMPC(x, u_prev, k, X_ref, Y_ref, psi_ref, ...
                                           vx_ref, dt, Np, L, Q, R, delta_step_max, opts)
    x_kin = x(1:2);  % [y, psi]

    % Get reference horizon
    idx = k:min(k+Np-1, length(X_ref));
    if length(idx) < Np
        pad_length = Np - length(idx);
        Y_horizon = [Y_ref(idx); repmat(Y_ref(end), pad_length, 1)];
        psi_horizon = [psi_ref(idx); repmat(psi_ref(end), pad_length, 1)];
        v_horizon = [vx_ref(idx); repmat(vx_ref(end), pad_length, 1)];
    else
        Y_horizon = Y_ref(idx);
        psi_horizon = psi_ref(idx);
        v_horizon = vx_ref(idx);
    end

    Xr = [Y_horizon'; psi_horizon'];

    % Feedforward steering reference
    psi_dot = gradient(psi_horizon, dt);
    psi_dot = medfilt1(psi_dot, min(3, length(psi_dot)));
    delta_ref_horizon = atan(L * psi_dot ./ max(v_horizon, 0.5));
    ur = [v_horizon'; delta_ref_horizon'];

    % Linearize at current state
    v_current = max(vx_ref(k), 0.5);
    delta_current = u_prev(2);
    [Ad, Bd] = linearizeKinematic(v_current, x_kin(2), delta_current, L, dt);

    % ============================================================
    % CRITICAL FIX: Only optimize steering, velocity follows reference
    % ============================================================
    nu = 1;  % Only steering angle change
    du0 = zeros(nu*Np, 1);
    lb = repmat(-delta_step_max, Np, 1);
    ub = repmat(delta_step_max, Np, 1);

    Q_bar = kron(eye(Np), Q);
    R_bar = R(2,2) * eye(Np);  % Only steering penalty

    % Use steering-only cost function
    cost_fun = @(du_vec) kinematicMPCCostSteeringOnly(du_vec, x_kin, u_prev(2), Xr, ur, ...
                                         Ad, Bd, Q_bar, R_bar, Np, 2, v_horizon);

    du_opt = fmincon(cost_fun, du0, [], [], [], [], lb, ub, [], opts);
    
    % ============================================================
    % CRITICAL: Velocity follows reference exactly
    % ============================================================
    u_opt = [vx_ref(k); u_prev(2) + du_opt(1)];

    x_next_kin = simulateKinematicStep(x_kin, u_opt, dt, L);
    x_next = [x_next_kin; x(3:4)];
end

function [u_opt, x_next] = solveDynamicMPC(x, u_prev, k, Y_ref, psi_ref, vx_ref, ...
                                         dt, Np, vehicle_params, Q_psi, Q_y, R, ...
                                         delta_step_max, opts)
    % x is [y, psi, vy, r]
    % x_dyn should be [vy, psi, r, Y]

    x_dyn = [x(3); x(2); x(4); x(1)];
    
    idx = k:min(k+Np-1, length(vx_ref));
    if length(idx) < Np
        pad_length = Np - length(idx);
        vx_horizon = [vx_ref(idx); repmat(vx_ref(end), pad_length, 1)];
        psi_horizon = [psi_ref(idx); repmat(psi_ref(end), pad_length, 1)];
        Y_horizon = [Y_ref(idx); repmat(Y_ref(end), pad_length, 1)];
    else
        vx_horizon = vx_ref(idx);
        psi_horizon = psi_ref(idx);
        Y_horizon = Y_ref(idx);
    end
    
    ref_trajectory = [psi_horizon, Y_horizon];
    
    du0 = zeros(Np, 1);
    lb = -delta_step_max * ones(Np, 1);
    ub = delta_step_max * ones(Np, 1);
    
    Q_mat = diag([Q_psi, Q_y]);
    Q_bar = kron(eye(Np), Q_mat);
    R_bar = R * eye(Np);
    
    cost_fun = @(delta_U) dynamicMPCCost(delta_U, x_dyn, u_prev(2), Q_bar, R_bar, ...
                                        Np, dt, vehicle_params, vx_horizon, ref_trajectory);
    
    % delta_U_opt = fmincon(cost_fun, du0, [], [], [], [], lb, ub, [], opts);
    [delta_U_opt, ~] = fmincon(cost_fun, du0, [], [], [], [], lb, ub, [], opts);
    u_opt = [vx_ref(k); u_prev(2) + delta_U_opt(1)];
    
    [A_c, B_c] = computeVehicleModel(vx_ref(k), vehicle_params);
    plant_dynamics = @(t, x_state) A_c * x_state + B_c * u_opt(2);
    [~, x_solution] = ode23s(plant_dynamics, [0 dt], x_dyn);
    x_next_dyn = x_solution(end, :)';
    
    % x_next_dyn is [vy, psi, r, Y]
    % Return [y, psi, vy, r]
    x_next = [x_next_dyn(4);    % Y
              x_next_dyn(2);    % psi
              x_next_dyn(1);    % vy
              x_next_dyn(3)];   % r
end

% function J = kinematicMPCCost(du_vec, x0, u_prev, Xr, ur, Ad_horizon, Bd_horizon, Qbar, Rbar, Np, nx, nu)
%     du = reshape(du_vec, nu, Np);
%     xerr = x0 - Xr(:,1);
%     Jstate = 0;
%     uk = u_prev;
%     for i = 1:Np
%         uk = uk + du(:,i);
%         u_tilde = uk - ur(:,i);
%         xerr = Ad_horizon{i}*xerr + Bd_horizon{i}*u_tilde;
%         Qi = Qbar((i-1)*nx+1:i*nx, (i-1)*nx+1:i*nx);
%         Jstate = Jstate + (xerr.'*Qi*xerr);
%     end
%     J = Jstate + du_vec.' * Rbar * du_vec;
% end


function J = kinematicMPCCost(du_vec, x0, u_prev, Xr, ur, Ad, Bd, Qbar, Rbar, Np, nx, nu)
    du = reshape(du_vec, nu, Np);
    xerr = x0 - Xr(:,1);
    Jstate = 0;
    uk = u_prev;
    for i = 1:Np
        uk = uk + du(:,i);
        u_tilde = uk - ur(:,i);
        xerr = Ad*xerr + Bd*u_tilde;
        Qi = Qbar((i-1)*nx+1:i*nx, (i-1)*nx+1:i*nx);
        Jstate = Jstate + (xerr.'*Qi*xerr);
    end
    J = Jstate + du_vec.' * Rbar * du_vec;
end

function J = kinematicMPCCostSteeringOnly(du_vec, x0, delta_prev, Xr, ur, Ad, Bd, ...
                                          Qbar, Rbar, Np, nx, v_horizon)
    % Only optimize steering, velocity follows reference
    du_delta = reshape(du_vec, 1, Np);
    xerr = x0 - Xr(:,1);
    Jstate = 0;
    delta_k = delta_prev;

    for i = 1:Np
        % Velocity follows reference, steering is optimized
        delta_k = delta_k + du_delta(i);
        u_k = [v_horizon(i); delta_k];
        u_tilde = u_k - ur(:,i);

        xerr = Ad * xerr + Bd * u_tilde;

        Qi = Qbar((i-1)*nx+1:i*nx, (i-1)*nx+1:i*nx);
        Jstate = Jstate + (xerr.' * Qi * xerr);
    end

    J = Jstate + du_vec.' * Rbar * du_vec;
end


% function J = kinematicMPCCostSteering(du_vec, x0, delta_prev, Xr, ur, Ad, Bd, ...
%                                       Qbar, Rbar, Np, nx, v_horizon)
%     du_delta = reshape(du_vec, 1, Np);  % Only steering changes
%     xerr = x0 - Xr(:,1);
%     Jstate = 0;
%     delta_k = delta_prev;
% 
%     for i = 1:Np
%         % ✅ Velocity follows reference, steering is optimized
%         delta_k = delta_k + du_delta(i);
%         u_k = [v_horizon(i); delta_k];
%         u_tilde = u_k - ur(:,i);
% 
%         xerr = Ad * xerr + Bd * u_tilde;
% 
%         Qi = Qbar((i-1)*nx+1:i*nx, (i-1)*nx+1:i*nx);
%         Jstate = Jstate + (xerr.' * Qi * xerr);
%     end
% 
%     J = Jstate + du_vec.' * Rbar * du_vec;
% end

function J = dynamicMPCCost(delta_U_seq, x_current, u_last, Q, R, Np, dt, ...
                          vehicle_params, vx_horizon, ref_traj)
    cost = 0;
    x_pred = x_current;
    u_pred = u_last;
    C_c = [0 1 0 0; 0 0 0 1];
    
    for i = 1:Np
        [A_c_step, B_c_step] = computeVehicleModel(vx_horizon(i), vehicle_params);
        sysd_step = c2d(ss(A_c_step, B_c_step, C_c, zeros(2,1)), dt);
        
        u_pred = u_pred + delta_U_seq(i);
        x_pred = sysd_step.A * x_pred + sysd_step.B * u_pred;
        y_pred = sysd_step.C * x_pred;
        
        error = ref_traj(i, :)' - y_pred;
        Q_step = Q((i-1)*2+1:i*2, (i-1)*2+1:i*2);
        cost = cost + error' * Q_step * error;
    end
    
    J = cost + delta_U_seq' * R * delta_U_seq;
end

function x_next = simulateKinematicStep(x, u, dt, L)
    f = @(~,xx) [u(1)*sin(xx(2)); (u(1)/L)*tan(u(2))];
    [~,xx] = ode45(f, [0 dt], x);
    x_next = xx(end,:).';
end


function plotHybridResults(t, x_hist, x_est_hist, u_cmd_hist, u_act_hist, ...
                          model_hist, X_ref, Y_ref, psi_ref, vx_ref, ...
                          sensor_error_hist, actuator_error_hist, estimation_error_hist, delta_u_hist)
    
    % Determine model switching points
    kinematic_idx = strcmp(model_hist, 'kinematic');
    dynamic_idx = strcmp(model_hist, 'dynamic');
    
    % figure('Name', 'Hybrid MPC with Sparse Error Recovery', 'Position', [50, 50, 1400, 1000]);
    figure(1)
    % Velocity profile with model indication
    subplot(5,1,1);
    plot(t, vx_ref, 'k--', 'LineWidth', 2); hold on;
    scatter(t(kinematic_idx), vx_ref(kinematic_idx), 30, 'b', 'filled');
    scatter(t(dynamic_idx), vx_ref(dynamic_idx), 30, 'r', 'filled');
    grid on; 
    ylabel('Velocity (m/s)', 'FontSize', 11); 
    title('Velocity Profile with Model Switching', 'FontSize', 13, 'FontWeight', 'bold');
    legend('Reference', 'Kinematic Mode', 'Dynamic Mode', 'Location', 'best');
    
    % Lateral position tracking
    subplot(5,1,2);
    plot(t, Y_ref, 'r--', 'LineWidth', 2); hold on;
    plot(t, x_hist(1,:), 'b-', 'LineWidth', 1.5);
    plot(t, x_est_hist(1,:), 'g:', 'LineWidth', 1.5);
    grid on; 
    ylabel('Y Position (m)', 'FontSize', 11); 
    title('Lateral Position Tracking', 'FontSize', 13, 'FontWeight', 'bold');
    legend('Reference', 'Actual', 'Estimated', 'Location', 'best');
    
    % Yaw angle tracking
    subplot(5,1,3);
%     plot(t, rad2deg(psi_ref), 'r--', 'LineWidth', 2); hold on;
    plot(t, psi_ref, 'r--', 'LineWidth', 2); hold on;
    plot(t, x_hist(2,:), 'b-', 'LineWidth', 1.5);
    plot(t, x_est_hist(2,:), 'g:', 'LineWidth', 1.5);
    grid on; 
    ylabel('Yaw Angle (deg)', 'FontSize', 11); 
    title('Yaw Angle Tracking', 'FontSize', 13, 'FontWeight', 'bold');
    legend('Reference', 'Actual', 'Estimated', 'Location', 'best');
    
    % Control inputs
    subplot(5,1,4);
    plot(t, rad2deg(u_cmd_hist(2,:)), 'b-', 'LineWidth', 1.5); hold on;
    plot(t, rad2deg(u_act_hist(2,:)), 'r--', 'LineWidth', 1.5);
    grid on; 
    ylabel('Steering (deg)', 'FontSize', 11); 
    title('Control Input (Steering Angle)', 'FontSize', 13, 'FontWeight', 'bold');
    legend('Commanded', 'Actuated', 'Location', 'best');

    % % rate of change of steering angle in deg/step
    % subplot(5,1,5);
    % plot(t, rad2deg(delta_u_hist), 'm-', 'LineWidth', 1.5);
    % grid on;
    % ylabel('\Delta\delta_f (deg/step)');
    % % plot(t_uniform(1:sim_steps), rad2deg(delta_u_hist)/delta_t, 'm-', 'LineWidth', 1.5);
    % % grid on
    % % ylabel('Steering Rate (deg/s)');
    % xlabel('Time (s)');
    % title('Rate of Change of Steering Angle');

    % rate of change of steering angle in deg/s (preferred)
    subplot(5,1,5);
    dt_plot = mean(diff(t));
    steering_rate_deg_s = rad2deg(delta_u_hist) / dt_plot;
    plot(t, steering_rate_deg_s, 'm-', 'LineWidth', 1.5); hold on;
    grid on;
    ylabel('Steering Rate (deg/s)', 'FontSize', 11);
    xlabel('Time (s)', 'FontSize', 11);
    title('Rate of Change of Steering Angle', 'FontSize', 13, 'FontWeight', 'bold');
    ylim([-55, 55]);
   
    
    % Figure 2: Sensor attacks
    % figure('Name', 'Sensor Attacks', 'Position', [100, 100, 1200, 800]);
    figure(2)

    % Actuator attacks
    subplot(3,1,1);
    plot(t, actuator_error_hist(:,2), 'r-', 'LineWidth', 1.5);
    grid on; 
    ylabel('e_a (deg)', 'FontSize', 11); 
    xlabel('Time (s)', 'FontSize', 11);
    title('Actuator Attacks (Steering)', 'FontSize', 13, 'FontWeight', 'bold');
    
    subplot(3,1,2);
    plot(t, sensor_error_hist(:,1), 'r-', 'LineWidth', 1.5);
    grid on;
    ylabel('e_{s,y} (m)', 'FontSize', 11);
    title('Sensor Attacks (Position & Orientation)', 'FontSize', 13, 'FontWeight', 'bold');
    
    subplot(3,1,3);
%     plot(t, rad2deg(sensor_error_hist(:,2)), 'r-', 'LineWidth', 1.5);
    plot(t, sensor_error_hist(:,2), 'r-', 'LineWidth', 1.5);
    grid on;
    ylabel('e_{s,\psi} (deg)', 'FontSize', 11);
    xlabel('Time (s)', 'FontSize', 11);
    
    % Figure 3: State estimation errors
    % figure('Name', 'State Estimation Errors', 'Position', [150, 150, 1200, 900]);
    figure(3)
  
    subplot(2,1,1);
    plot(t, estimation_error_hist(:,1), 'b-', 'LineWidth', 1.5);
    grid on;
    ylabel('e_Y (m)', 'FontSize', 11);
    title('State Estimation Errors', 'FontSize', 13, 'FontWeight', 'bold');
    
    subplot(2,1,2);
%     plot(t, rad2deg(estimation_error_hist(:,2)), 'b-', 'LineWidth', 1.5);
    plot(t, estimation_error_hist(:,2), 'b-', 'LineWidth', 1.5);
    grid on;
    ylabel('e_{\psi} (deg)', 'FontSize', 11);
end