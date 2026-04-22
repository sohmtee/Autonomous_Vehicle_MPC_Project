% True LTV-MPC — GTZ Kinematic / Dynamic Bicycle Model Switch
% Solved efficiently using Quadratic Programming (quadprog)
% EXACT DISCRETIZATION (Module 49)

clc; clear; close all;

%% Vehicle & controller parameters
Cf = 30000;   Cr = 30000;
l1 = 1.4225;  l2 = 1.4225;
l  = l1 + l2;
m  = 1280;    Iz = 2500;

delta_t = 0.05;
Np      = 30;

Q_psi_kin = 1000;  Q_y_kin = 400;  R_du_kin = 500;
Q_psi_dyn = 1000;  Q_y_dyn = 400;  R_du_dyn = 2000;

du_max = deg2rad(5);    % Rate constraint (per step)
u_max  = deg2rad(27.1); % Magnitude constraint

v_switch_lo = 3.0;
v_switch_hi = 4.0;

vx_min_dyn = 0.5;
vx_min_kin = 0;  

%% Load reference trajectory
traj = readmatrix('C:\Users\soguchie\OneDrive - purdue.edu\ECE 699\Summer 2025\RoadRunner Projects\RoadRunner Project 1\Exports\BostonIntersection_design\Sedan_1.csv');


time     = traj(:,1);
X_path   = traj(:,2);
Y_path   = traj(:,3);
psi_path = traj(:,5);
vx_raw   = traj(:,8);

% Yaw convention check
n_chk   = min(10, length(time)-1);
psi_num = atan2(mean(diff(Y_path(1:n_chk+1))), mean(diff(X_path(1:n_chk+1))));
if abs(angdiff(psi_num, psi_path(1))) > deg2rad(5)
    psi_path = psi_path + pi/2;
end

psi_unwrapped = unwrap(psi_path);
t_uniform     = (0 : delta_t : time(end))';

v_ref   = interp1(time, vx_raw, t_uniform, 'linear');
X_ref   = interp1(time, X_path, t_uniform, 'pchip');
Y_ref   = interp1(time, Y_path, t_uniform, 'pchip');
psi_ref = interp1(time, psi_unwrapped, t_uniform, 'pchip');

N_total = length(t_uniform);

try
    pp          = pchip(t_uniform, psi_ref);
    psi_dot_ref = ppval(fnder(pp, 1), t_uniform);
catch
    psi_dot_ref = smooth(gradient(psi_ref, delta_t), 5);
end

% Reference steering feedforward — CG kinematic model
beta_r_ref  = asin( min( abs(l2 .* psi_dot_ref ./ max(v_ref, vx_min_kin)), 1) .* sign(psi_dot_ref) );
delta_r_ref = atan( l * tan(beta_r_ref) / l2 );
delta_r_ref = max(min(delta_r_ref, u_max), -u_max);

%% Cost matrices & Output Setup
Q_kin_bar = kron(eye(Np), diag([Q_psi_kin, Q_y_kin]));
R_kin_bar = kron(eye(Np), R_du_kin);

Q_dyn_bar = kron(eye(Np), diag([Q_psi_dyn, Q_y_dyn]));
R_dyn_bar = kron(eye(Np), R_du_dyn);

% Output matrices (Maps [states] to [e_psi, e_y])
Cd_kin = [0 0 1; 
          0 1 0]; 
Cd_dyn = [0 1 0 0; 
          0 0 0 1];

%% Initialize simulation
sim_steps = N_total - Np;

e_kin = zeros(3, 1);
x_dyn = zeros(4, 1);

e_kin_prev = e_kin;
x_dyn_prev = x_dyn;

X_global   = X_ref(1);
Y_global   = Y_ref(1);
psi_global = psi_ref(1);

if v_ref(1) < v_switch_lo
    active_model = 'kinematic';
else
    active_model = 'dynamic';
end

u_prev = delta_r_ref(1);

% Logs
e_psi_log  = zeros(sim_steps, 1);
e_y_log    = zeros(sim_steps, 1);
Y_log      = zeros(sim_steps, 1);
X_log      = zeros(sim_steps, 1);
psi_log    = zeros(sim_steps, 1);
u_log      = zeros(sim_steps, 1);
du_log     = zeros(sim_steps, 1);
v_log      = zeros(sim_steps, 1);
model_log  = strings(sim_steps, 1);

options = optimoptions('quadprog', 'Display', 'off');

fprintf('Starting True LTV-MPC (%d steps)...\n', sim_steps);

%% Main simulation loop
for k = 1 : sim_steps

    vx_now = v_ref(k);

    % Model switching
    if strcmp(active_model, 'kinematic') && vx_now >= v_switch_hi
        active_model = 'dynamic';
        beta_prev = atan2(l2 * tan(u_prev), l);
        r_init    = max(vx_now, vx_min_dyn) / l2 * sin(beta_prev);
        e_y_path  = -e_kin(1)*sin(psi_ref(k)) + e_kin(2)*cos(psi_ref(k));
        x_dyn     = [0; e_kin(3); r_init; e_y_path];
        x_dyn_prev = x_dyn; 
    elseif strcmp(active_model, 'dynamic') && vx_now <= v_switch_lo
        active_model = 'kinematic';
        e_s_path = (X_global - X_ref(k))*cos(psi_ref(k)) + (Y_global - Y_ref(k))*sin(psi_ref(k));
        e_y_path = x_dyn(4);
        e_kin    = [e_s_path * cos(psi_ref(k)) - e_y_path * sin(psi_ref(k)); 
                    e_s_path * sin(psi_ref(k)) + e_y_path * cos(psi_ref(k)); 
                    x_dyn(2)];
        e_kin_prev = e_kin;
    end

    % Extract outputs for logs
    if strcmp(active_model, 'kinematic')
        y_out = Cd_kin * e_kin; 
        dx = e_kin - e_kin_prev;
        x_a = [dx; y_out];
        Q_bar = Q_kin_bar;
        R_bar = R_kin_bar;
        C_mat = Cd_kin;
        n_st = 3;
    else
        y_out = Cd_dyn * x_dyn; 
        dx = x_dyn - x_dyn_prev;
        x_a = [dx; y_out];
        Q_bar = Q_dyn_bar;
        R_bar = R_dyn_bar;
        C_mat = Cd_dyn;
        n_st = 4;
    end
    
    e_psi_log(k) = y_out(1);
    e_y_log(k)   = y_out(2);
    Y_log(k)     = Y_global;
    X_log(k)     = X_global;
    psi_log(k)   = psi_global;
    v_log(k)     = vx_now;
    model_log(k) = active_model;

    %% 1. Build Time-Varying Sequences for the Horizon
    Phi_a_seq   = cell(Np, 1);
    Gamma_a_seq = cell(Np, 1);
    
    p = size(C_mat, 1);
    C_a = [zeros(p, n_st), eye(p)];

    for i = 1:Np
        idx = k + i - 1; % Index of the future reference point
        
        v_r = max(v_ref(idx), vx_min_kin);
        psi_r = psi_ref(idx);
        delta_r = delta_r_ref(idx);
        
        if strcmp(active_model, 'kinematic')
            beta_r = atan2(l2 * tan(delta_r), l);
            dbeta_ddelta = l2 * l / (l^2 * cos(delta_r)^2 + l2^2 * sin(delta_r)^2);
            
            A_T = [0, 0, -v_r * sin(psi_r + beta_r);
                   0, 0,  v_r * cos(psi_r + beta_r);
                   0, 0,  0];
                   
            B_T = [cos(psi_r + beta_r),  -v_r * sin(psi_r + beta_r) * dbeta_ddelta;
                   sin(psi_r + beta_r),   v_r * cos(psi_r + beta_r) * dbeta_ddelta;
                   sin(beta_r) / l2,     (v_r / l2) * cos(beta_r) * dbeta_ddelta];
            
            % Exact Discretization (Module 49)
            M_kin = expm([A_T, B_T(:,2); zeros(1, 4)] * delta_t);
            Phi   = M_kin(1:3, 1:3);
            Gamma = M_kin(1:3, 4);
        else
            v_r_dyn = max(v_r, vx_min_dyn);
            A_c = dynMatrix_A(v_r_dyn, Cf, Cr, m, Iz, l1, l2);
            B_c = dynMatrix_B(Cf, m, l1, Iz);
            
            % Exact Discretization (Module 49)
            M_dyn = expm([A_c, B_c; zeros(1, 5)] * delta_t);
            Phi   = M_dyn(1:4, 1:4);
            Gamma = M_dyn(1:4, 5);
        end
        
        % Augment the matrices for this specific step in the horizon
        Phi_a_seq{i}   = [Phi, zeros(n_st, p); 
                          C_mat*Phi, eye(p)];
        Gamma_a_seq{i} = [Gamma; 
                          C_mat*Gamma];
    end
    
    %% 2. Build LTV Prediction Matrices (W, Z)
    n_a = n_st + p;
    W = zeros(p * Np, n_a);
    Z = zeros(p * Np, Np);
    
    Phi_prod = eye(n_a);
    for i = 1:Np
        % W matrix rows: C_a * Phi_{i-1} * ... * Phi_0
        Phi_prod = Phi_a_seq{i} * Phi_prod;
        W((i-1)*p+1 : i*p, :) = C_a * Phi_prod;
        
        % Z matrix blocks: maps control j to output i
        for j = 1:i
            if i == j
                temp_prod = eye(n_a);
            else
                temp_prod = eye(n_a);
                for m = i:-1:j+1
                    temp_prod = temp_prod * Phi_a_seq{m};
                end
            end
            Z((i-1)*p+1 : i*p, j) = C_a * temp_prod * Gamma_a_seq{j};
        end
    end
    
    %% 3. Define Constraints for Quadprog
    lb_du = -du_max * ones(Np, 1);
    ub_du =  du_max * ones(Np, 1);
    
    E_mat = ones(Np, 1);
    H_mat = tril(ones(Np));
    
    U_max_vec = u_max * ones(Np, 1);
    U_min_vec = -u_max * ones(Np, 1);
    
    A_ineq = [H_mat; -H_mat];
    b_ineq = [U_max_vec - E_mat * u_prev; 
             -U_min_vec + E_mat * u_prev];

    %% 4. Solve Quadratic Program
    H_qp = Z' * Q_bar * Z + R_bar;
    H_qp = (H_qp + H_qp') / 2; % Enforce strict symmetry for MATLAB solver
    f_qp = Z' * Q_bar * (W * x_a);
    
    [delta_U_opt, ~, exitflag] = quadprog(H_qp, f_qp, A_ineq, b_ineq, [], [], lb_du, ub_du, [], options);
    
    if exitflag ~= 1 && exitflag ~= 2
        warning('Optimization failed at step %d. Coasting.', k);
        du_now = 0; 
    else
        du_now = delta_U_opt(1);
    end
    
    u_current = u_prev + du_now;

    %% 5. Propagate Nonlinear/Actual Plant
    e_kin_prev = e_kin;
    x_dyn_prev = x_dyn;
    
    if strcmp(active_model, 'kinematic')
        beta_now = atan2(l2 * tan(u_current), l);
        ode_kin  = @(~, s) [vx_now * cos(s(3) + beta_now);
                            vx_now * sin(s(3) + beta_now);
                            (vx_now / l2) * sin(beta_now)];
        [~, s] = ode23s(ode_kin, [0, delta_t], [X_global; Y_global; psi_global]);

        X_global   = s(end,1);
        Y_global   = s(end,2);
        psi_global = s(end,3);

        e_kin(1) = X_global - X_ref(k+1);
        e_kin(2) = Y_global - Y_ref(k+1);
        e_kin(3) = angdiff(psi_ref(k+1), psi_global);

    else
        vx_pl = max(vx_now, vx_min_dyn);
        A_c_act = dynMatrix_A(vx_pl, Cf, Cr, m, Iz, l1, l2);
        B_c_act = dynMatrix_B(Cf, m, l1, Iz);
        W_c_act = [0; -psi_dot_ref(k); 0; 0];

        ode_dyn   = @(~, xs) A_c_act * xs + B_c_act * u_current + W_c_act;
        [~, s]    = ode23s(ode_dyn, [0, delta_t], x_dyn);
        x_dyn     = s(end,:)';

        psi_global = psi_ref(k+1) + x_dyn(2);
        Y_global   = Y_ref(k+1)   + x_dyn(4) * cos(psi_ref(k+1));
        X_global   = X_ref(k+1)   - x_dyn(4) * sin(psi_ref(k+1));
    end

    % Store & shift for next step
    u_log(k)  = u_current;
    du_log(k) = du_now;
    u_prev    = u_current;
end

fprintf('Simulation Complete.\n');

%% Plots
t_plot  = t_uniform(1:sim_steps);
kin_idx = model_log == "kinematic";
dyn_idx = model_log == "dynamic";

figure(1);
subplot(4,1,1);
plot(t_plot, v_log, 'k-', 'LineWidth', 1.5); hold on;
scatter(t_plot(kin_idx), v_log(kin_idx), 20, 'b', 'filled');
scatter(t_plot(dyn_idx), v_log(dyn_idx), 20, 'r', 'filled');
ax = gca; grid on; legend('Speed','Kinematic','Dynamic','Location','best');
ylabel('v (m/s)');

subplot(4,1,2);
plot(t_plot, Y_ref(1:sim_steps), 'r--', 'LineWidth', 2); hold on;
plot(t_plot, Y_log, 'b-', 'LineWidth', 2);
ax = gca; grid on; legend('Y_{ref}','Y_{actual}','Location','best'); 
ylabel('Y (m)');

subplot(4,1,3);
plot(t_plot, psi_ref(1:sim_steps), 'r--', 'LineWidth', 2); hold on;
plot(t_plot, psi_log, 'b-', 'LineWidth', 2);
ax = gca; grid on; legend('\psi_{ref}','\psi_{actual}','Location','best'); 
ylabel('\psi (rad)');

subplot(4,1,4);
plot(t_plot, rad2deg(u_log), 'k-', 'LineWidth', 2);
ax = gca;
yline( rad2deg(u_max),  'r--', 'LineWidth',2); yline(-rad2deg(u_max),  'r--', 'LineWidth',2);
grid on; ylabel('\delta_f (deg)'); xlabel('Time (s)'); ylim([-30 30]);

figure(2);
subplot(3,1,1);
plot(t_plot, e_psi_log, 'r-', 'LineWidth', 2);
ax = gca; grid on; ylabel('e_\psi (rad)');

subplot(3,1,2);
plot(t_plot, e_y_log, 'b-', 'LineWidth', 2);
ax = gca; grid on; ylabel('e_y (m)');

subplot(3,1,3);
plot(t_plot, rad2deg(du_log), 'k-', 'LineWidth', 2);
ax = gca;
yline( rad2deg(du_max),  'r--', 'LineWidth',2); yline(-rad2deg(du_max),  'r--', 'LineWidth',2);
grid on; ylabel('\Delta\delta_f (deg/step)'); xlabel('Time (s)'); ylim([-10 10]);

%% LOCAL FUNCTIONS

function A_c = dynMatrix_A(vx, Cf, Cr, m, Iz, l1, l2)
    A_c = [-(2*Cf+2*Cr)/(m*vx),        0,  -(2*Cf*l1-2*Cr*l2)/(m*vx)-vx,   0;
                0,                      0,   1,                               0;
           -(2*l1*Cf-2*l2*Cr)/(Iz*vx), 0,  -(2*l1^2*Cf+2*l2^2*Cr)/(Iz*vx), 0;
               -1,                     -vx,  0,                               0];
end

function B_c = dynMatrix_B(Cf, m, l1, Iz)
    B_c = [(2*Cf)/m; 0; (2*l1*Cf)/Iz; 0];
end