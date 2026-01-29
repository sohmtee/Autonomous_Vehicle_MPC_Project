% Hybrid MPC Controller - Switches between Kinematic and Dynamic Models

clc; clear; close all;

%% Load and Process Trajectory Data
csvFile = 'C:\Users\soguchie\OneDrive - purdue.edu\ECE 699\Summer 2025\RoadRunner Projects\RoadRunner Project 1\Exports\SingaporeIntersection_design_varying_velocities2_for_hybrid_models\Sedan3_3a.csv';
traj = readmatrix(csvFile);

time_raw = traj(:,1);
X_raw = traj(:,2);
Y_raw = traj(:,3);
yaw_deg = traj(:,5);
vx_raw = traj(:,8);

% Process trajectory
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

% Smooth velocity to avoid abrupt changes
vx_ref = smooth(vx_ref, 25);

%% Controller Parameters
% Switching thresholds (with hysteresis)
v_switch_high = 5;    % Switch to dynamic at this speed
v_switch_low = 3;     % Switch back to kinematic at this speed

% Vehicle parameters (for dynamic model)
vehicle_params = struct();
vehicle_params.Cf = 30000; 
vehicle_params.Cr = 30000; 
vehicle_params.l1 = 1.2; 
vehicle_params.l2 = 1.22; 
vehicle_params.L = vehicle_params.l1 + vehicle_params.l2;  % Total wheelbase
vehicle_params.m = 1280; 
vehicle_params.Iz = 2500;

% MPC Parameters
Np = 50;  % Prediction horizon

% Cost weights
Q_kinematic = diag([100, 1000, 500]);  % [x, y, psi] weights
Q_dynamic_psi = 1000; 
Q_dynamic_y = 800;
R_kinematic = diag([0, 15000]);          % [dv, ddelta] weights  
R_dynamic = 20000;                     % delta_u weight

% Constraints
delta_rate_max = deg2rad(50);         % 50 deg/s steering rate limit.
delta_step_max = delta_rate_max * dt; % per time step

% Optimization options
opts = optimoptions('fmincon','Algorithm','sqp','Display','off', ...
    'MaxIterations',100,'OptimalityTolerance',1e-6);

%% Initialize Simulation
Nsim = length(t_uniform) - Np;

% Unified state vector: [x, y, psi, vy, r] (5 states)
% - Kinematic model uses: [x, y, psi]
% - Dynamic model uses: [psi, Y, vy, r] (Y computed from trajectory)
x_unified = [X_ref(1); Y_ref(1); psi_ref(1); 0; 0];  % [x, y, psi, vy, r]

% Controller state
current_model = 'kinematic';  % Start with kinematic
u_prev = [vx_ref(100); 0];      % [v, delta]

% Storage arrays
x_hist = zeros(5, Nsim);
u_hist = zeros(2, Nsim);
model_hist = cell(Nsim, 1);

fprintf('Starting Hybrid MPC Simulation...\n');
fprintf('Switching: Kinematic (vx <= %.1f) <-> Dynamic (vx >= %.1f)\n', ...
        v_switch_low, v_switch_high);

%% Main Simulation Loop
for k = 1:Nsim
    tic;
    
    % Current velocity
    vx_current = vx_ref(k);
    
    %% Model Switching Logic (with hysteresis)
    if strcmp(current_model, 'kinematic')
        if vx_current >= v_switch_high
            current_model = 'dynamic';
            fprintf('Switched to DYNAMIC model at t=%.2f s (v=%.2f m/s)\n', ...
                    t_uniform(k), vx_current);
            % Convert state from kinematic to dynamic representation
            x_unified = convertKinematicToDynamic(x_unified, Y_ref(k));
        end
    else  % current_model == 'dynamic'
        if vx_current <= v_switch_low
            current_model = 'kinematic';
            fprintf('Switched to KINEMATIC model at t=%.2f s (v=%.2f m/s)\n', ...
                    t_uniform(k), vx_current);
            % Convert state from dynamic to kinematic representation  
            x_unified = convertDynamicToKinematic(x_unified, X_ref(k));
        end
    end
    
    %% Solve MPC based on current model
    if strcmp(current_model, 'kinematic')
        current_delta_max = deg2rad(50) * dt;
        [u_opt, x_next] = solveKinematicMPC(x_unified, u_prev, k, X_ref, Y_ref, ...
                                          psi_ref, vx_ref, dt, Np, ...
                                          vehicle_params.L, Q_kinematic, ...
                                          R_kinematic, current_delta_max, opts);
    else
        current_delta_max = deg2rad(50) * dt;
        [u_opt, x_next] = solveDynamicMPC(x_unified, u_prev, k, Y_ref, ...
                                        psi_ref, vx_ref, dt, Np, ...
                                        vehicle_params, Q_dynamic_psi, ...
                                        Q_dynamic_y, R_dynamic, current_delta_max, opts);
    end
    
    %% Store results
    x_hist(:, k) = x_next;
    u_hist(:, k) = u_opt;
    model_hist{k} = current_model;
    % computation_time(k) = toc;
    
    %% Update for next iteration
    x_unified = x_next;
    u_prev = u_opt;
    
    % Progress indicator
    if mod(k, 200) == 0
        fprintf('Step %d/%d, Current model: %s, Speed: %.2f m/s\n', ...
                k, Nsim, current_model, vx_current);
    end
end

fprintf('Hybrid MPC Simulation Complete!\n');
% fprintf('Average computation time: %.2f ms\n', mean(computation_time) * 1000);

%% Plot Results
plotHybridResults(t_uniform(1:Nsim), x_hist, u_hist, model_hist, ...
                 X_ref(1:Nsim), Y_ref(1:Nsim), psi_ref(1:Nsim), vx_ref(1:Nsim));

%% =========================== HELPER FUNCTIONS ===========================

function x_dynamic = convertKinematicToDynamic(x_kinematic, Y_current)
    % Convert [x, y, psi, vy, r] kinematic to dynamic representation
    % Dynamic state focuses on [psi, Y, vy, r] where Y comes from reference
    x_dynamic = [x_kinematic(1);    % x (keep for unified state)
                 Y_current;         % Y (use current reference)  
                 x_kinematic(3);    % psi
                 x_kinematic(4);    % vy (already present)
                 x_kinematic(5)];   % r (already present)
end

function x_kinematic = convertDynamicToKinematic(x_dynamic, X_current)
    % Convert dynamic back to kinematic representation
    x_kinematic = [X_current;       % x (use current reference)
                   x_dynamic(2);    % y/Y
                   x_dynamic(3);    % psi  
                   x_dynamic(4);    % vy (keep for unified state)
                   x_dynamic(5)];   % r (keep for unified state)
end

function [u_opt, x_next] = solveKinematicMPC(x, u_prev, k, X_ref, Y_ref, psi_ref, ...
                                           vx_ref, dt, Np, L, Q, R, delta_step_max, opts)
    % Kinematic MPC solver
    
    % Extract kinematic states
    x_kin = x(1:3);  % [x, y, psi]
    
    % Reference trajectory for horizon
    idx = k:min(k+Np-1, length(X_ref));
    if length(idx) < Np
        % Pad with last values
        pad_length = Np - length(idx);
        X_horizon = [X_ref(idx); repmat(X_ref(end), pad_length, 1)];
        Y_horizon = [Y_ref(idx); repmat(Y_ref(end), pad_length, 1)];
        psi_horizon = [psi_ref(idx); repmat(psi_ref(end), pad_length, 1)];
        v_horizon = [vx_ref(idx); repmat(vx_ref(end), pad_length, 1)];
    else
        X_horizon = X_ref(idx);
        Y_horizon = Y_ref(idx);
        psi_horizon = psi_ref(idx);
        v_horizon = vx_ref(idx);
    end
    
    Xr = [X_horizon'; Y_horizon'; psi_horizon'];

    % added this:
    psi_dot = gradient(psi_horizon, dt);
    psi_dot = medfilt1(psi_dot, min(3, length(psi_dot)));
    delta_ref_horizon = atan(L * psi_dot ./ v_horizon);


    % ur = [v_horizon'; atan(L * gradient(psi_horizon, dt)' / mean(v_horizon))];
    ur = [v_horizon'; delta_ref_horizon'];


    % added this:
    v_current = max(vx_ref(k), 0.05);  % can still be v_current = max(vx_ref(k), 0.00). works too!
    delta_current = u_prev(2);
    
    % Linearize at current reference
    % [Ad, Bd] = linearizeKinematic(mean(v_horizon), psi_ref(k), ur(2,1), L, dt);
    [Ad, Bd] = linearizeKinematic(v_current, x_kin(3), delta_current, L, dt);
    
    % MPC optimization
    nu = 2;
    du0 = zeros(nu*Np, 1);
    lb = repmat([0; -delta_step_max], Np, 1);
    ub = repmat([0;  delta_step_max], Np, 1);
    
    Q_bar = kron(eye(Np), Q);
    R_bar = kron(eye(Np), R);
    
    cost_fun = @(du_vec) kinematicMPCCost(du_vec, x_kin, u_prev, Xr, ur, ...
                                         Ad, Bd, Q_bar, R_bar, Np, 3, nu);
    
    du_opt = fmincon(cost_fun, du0, [], [], [], [], lb, ub, [], opts);
    u_opt = u_prev + du_opt(1:nu);
    
    % Simulate one step with kinematic model
    x_next_kin = simulateKinematicStep(x_kin, u_opt, dt, L);
    
    % Update unified state
    x_next = [x_next_kin; x(4:5)];  % Keep vy, r from previous
end

function [u_opt, x_next] = solveDynamicMPC(x, u_prev, k, Y_ref, psi_ref, vx_ref, ...
                                         dt, Np, vehicle_params, Q_psi, Q_y, R, ...
                                         delta_step_max, opts)
    % Dynamic MPC solver
    
    % Extract dynamic states: [vy, psi, r, Y]
    x_dyn = [x(4); x(3); x(5); x(2)];  % [vy, psi, r, Y]
    
    % Get velocity horizon
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
    
    % Setup optimization
    du0 = zeros(Np, 1);
    lb = -delta_step_max * ones(Np, 1);
    ub = delta_step_max * ones(Np, 1);
    
    Q_mat = diag([Q_psi, Q_y]);
    Q_bar = kron(eye(Np), Q_mat);
    R_bar = R * eye(Np);
    
    cost_fun = @(delta_U) dynamicMPCCost(delta_U, x_dyn, u_prev(2), Q_bar, R_bar, ...
                                        Np, dt, vehicle_params, vx_horizon, ref_trajectory);
    
    [delta_U_opt, ~] = fmincon(cost_fun, du0, [], [], [], [], lb, ub, [], opts);
    
    u_opt = [vx_ref(k); u_prev(2) + delta_U_opt(1)];  % [v, delta]
    
    % Simulate one step with dynamic model
    [A_c, B_c] = computeVehicleModel(vx_ref(k), vehicle_params);
    plant_dynamics = @(t, x_state) A_c * x_state + B_c * u_opt(2);
    [~, x_solution] = ode23s(plant_dynamics, [0 dt], x_dyn);
    x_next_dyn = x_solution(end, :)';
    
    % Update unified state: [x, y, psi, vy, r]
    % Integrate x position
    x_next = [x(1) + vx_ref(k) * cos(x_next_dyn(2)) * dt;  % x
              x_next_dyn(4);                                % Y
              x_next_dyn(2);                                % psi  
              x_next_dyn(1);                                % vy
              x_next_dyn(3)];                               % r
end

function [Ad, Bd] = linearizeKinematic(v_r, psi_r, delta_r, L, dt)
    % Linearize kinematic bicycle model
    Ac = [0, 0, -v_r*sin(psi_r);
          0, 0,  v_r*cos(psi_r);
          0, 0,  0];
    Bc = [cos(psi_r), 0;
          sin(psi_r), 0;
          tan(delta_r)/L, v_r/(L*cos(delta_r)^2)];  % formerly tan(delta_r) and works
    sysd = c2d(ss(Ac,Bc,eye(3),zeros(3,2)), dt, 'zoh');
    Ad = sysd.A; Bd = sysd.B;
end

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

function x_next = simulateKinematicStep(x, u, dt, L)
    f = @(~,xx) [u(1)*cos(xx(3)); u(1)*sin(xx(3)); (u(1)/L)*tan(u(2))];
    [~,xx] = ode45(f, [0 dt], x);
    x_next = xx(end,:).';
end

function [A_c, B_c] = computeVehicleModel(vx, params)
    % Dynamic vehicle model
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

function plotHybridResults(t, x_hist, u_hist, model_hist, X_ref, Y_ref, psi_ref, vx_ref)
    figure('Name', 'Hybrid MPC Results', 'Position', [100, 100, 1400, 1200]);
    
    % Determine model switching points
    kinematic_idx = strcmp(model_hist, 'kinematic');
    dynamic_idx = strcmp(model_hist, 'dynamic');

    % addeddd this:
    dt = mean(diff(t));
    % addeddd this
    delta = u_hist(2, 1:length(t));
    delta_rate = [0, diff(delta)] / dt;
    delta_rate_deg = rad2deg(delta_rate);
    
    % Velocity profile with model indication
    % figure('Name','Velocity Profile')
    subplot(5,1,1);
    plot(t, vx_ref(1:length(t)), 'k--', 'LineWidth', 2); hold on;
    scatter(t(kinematic_idx), vx_ref(kinematic_idx), 30, 'b', 'filled');
    scatter(t(dynamic_idx), vx_ref(dynamic_idx), 30, 'r', 'filled');
    %yline(8, 'g--', 'LineWidth', 1.5);
    %yline(7, 'm--', 'LineWidth', 1.5);
    grid on; 
    ylabel('Velocity (m/s)', 'FontSize', 12); 
    title('(a) Velocity Profile', 'FontWeight', 'bold', 'FontSize', 14);
    % legend('Reference', 'Kinematic', 'Dynamic', 'Switch High', 'Switch Low');
    legend('Reference', 'Kinematic', 'Dynamic');
    ylim([0, 25]); 
    
    % Lateral position tracking
    % figure('Name','Lateral Position')
    subplot(5,1,2);
    plot(t, Y_ref, 'r--', 'LineWidth', 2); hold on;
    plot(t, x_hist(2,1:length(t)), 'b-', 'LineWidth', 1.5);
    grid on; 
    ylabel('Y Position (m)', 'FontSize', 12); 
    title('(b) Lateral Position Tracking', 'FontWeight', 'bold', 'FontSize', 14);
    legend('Reference', 'Actual');
    
    % Yaw angle tracking  
    % figure('Name','Yaw Angle')
    subplot(5,1,3);
    plot(t, psi_ref(1:length(t)), 'r--', 'LineWidth', 2); hold on;
    plot(t, x_hist(3,1:length(t)), 'b-', 'LineWidth', 1.5);
    grid on; 
    ylabel('Yaw Angle (deg)', 'FontSize', 12); 
    title('(c) Yaw Angle Tracking', 'FontWeight', 'bold', 'FontSize', 14);
    legend('Reference', 'Actual');
    
    % Control input
    % figure('Name','Control Input')
    subplot(5,1,4);
    plot(t, rad2deg(u_hist(2,1:length(t))), 'k-', 'LineWidth', 1.5);
    grid on; 
    ylabel('Steering Angle (deg)', 'FontSize', 12); 
    xlabel('Time (s)', 'FontSize', 12);
    title('(d) Control Input (Steering Angle)', 'FontWeight', 'bold', 'FontSize', 14);

    % Steering Rate (Rate of Change of Steering Angle)
    subplot(5,1,5);
    plot(t, delta_rate_deg, 'b-', 'LineWidth', 1.5); hold on;
    % yline(30, 'r--', 'LineWidth', 1.5, 'Label', 'Limit (+30°/s)');
    % yline(-30, 'r--', 'LineWidth', 1.5, 'Label', 'Limit (-30°/s)');
    grid on; 
    ylabel('Steering Rate (deg/s)','FontSize',12); 
    xlabel('Time (s)', 'FontSize', 12);
    title('(e) Rate of Change of Steering Angle', 'FontWeight', 'bold', 'FontSize', 14);
    ylim([-55, 55]); 
end