% LTV-MPC with 2-Input [v_x, delta] + UIO + Sparse Error Recovery
% UNIFIED GLOBAL POSE STATE for BOTH Kinematic & Dynamic Modes
% Uses 2D Scheduled UIO Gains

clc; clear; close all;

%% Vehicle & Controller Parameters
Cf = 30000;   Cr = 30000;
l1 = 1.4225;  l2 = 1.4225;
l  = l1 + l2;
m  = 1280;    Iz = 2500;

delta_t = 0.02;
Np      = 35;

% MPC weights
Q_x_kin = 500;  Q_psi_kin = 1000;  Q_y_kin = 500;  
R_dv_kin = 1000; R_du_kin = 500;

Q_x_dyn = 500;  Q_psi_dyn = 1000;  Q_y_dyn = 2000;  
R_dv_dyn = 1000; R_du_dyn = 500;

% Rate constraints (per step)
dv_max = 2.5 * delta_t;       
du_max = deg2rad(10);      

% Absolute constraints
v_min = 0.0;
v_max = 25.0;
u_max = deg2rad(27.1);

% Hysteresis switching thresholds
v_lo = 6.0;
v_hi = 7.0;

vx_min_dyn = 0.5;
vx_min_kin = 0.0;

% Attack / recovery parameters
tau          = 10;
p_attack_s   = 0.05;          
p_attack_a_v = 0.02;          
p_attack_a_d = 0.02;          
p_attack_a   = [p_attack_a_v; p_attack_a_d];  

alpha_lpf = 0.85; 
y_corrected_prev = zeros(3, 1);

%% Load Reference Trajectory
traj = readmatrix('C:\Users\soguchie\OneDrive - purdue.edu\ECE 699\Summer 2025\RoadRunner Projects\RoadRunner Project 1\Exports\BostonIntersection_design\Sedan_1.csv');

time     = traj(:,1);
X_path   = traj(:,2);
Y_path   = traj(:,3);
psi_path = traj(:,5);
vx_raw   = traj(:,8);

n_chk   = min(10, length(time)-1);
psi_num = atan2(mean(diff(Y_path(1:n_chk+1))), mean(diff(X_path(1:n_chk+1))));
if abs(angdiff(psi_num, psi_path(1))) > deg2rad(5)
    psi_path = psi_path + pi/2;
end

psi_unwrapped = unwrap(psi_path);
t_uniform     = (0 : delta_t : time(end))';

v_ref   = interp1(time, vx_raw,        t_uniform, 'linear');
X_ref   = interp1(time, X_path,        t_uniform, 'pchip');
Y_ref   = interp1(time, Y_path,        t_uniform, 'pchip');
psi_ref = interp1(time, psi_unwrapped, t_uniform, 'pchip');

N_total = length(t_uniform);

try
    pp_spline   = pchip(t_uniform, psi_ref);
    psi_dot_ref = ppval(fnder(pp_spline, 1), t_uniform);
catch
    psi_dot_ref = smooth(gradient(psi_ref, delta_t), 5);
end

beta_r_ref  = asin( min( abs(l2 .* psi_dot_ref ./ max(v_ref, vx_min_kin)), 1) .* sign(psi_dot_ref) );
delta_r_ref = atan( l * tan(beta_r_ref) / l2 );
delta_r_ref = smooth(delta_r_ref, 3);
delta_r_ref = max(min(delta_r_ref, u_max), -u_max);

%% Load 2D UIO Gain Schedule
load('observer_gain_schedule_with_M.mat');   

%% MPC Matrices & Output Setup
Q_kin_bar = kron(eye(Np), diag([Q_x_kin, Q_psi_kin, Q_y_kin]));
R_kin_bar = kron(eye(Np), diag([R_dv_kin, R_du_kin]));

Q_dyn_bar = kron(eye(Np), diag([Q_x_dyn, Q_psi_dyn, Q_y_dyn]));
R_dyn_bar = kron(eye(Np), diag([R_dv_dyn, R_du_dyn]));

pp_kin = 3;     
pp_dyn = 3;     
m_in   = 2;     

opt_lp = optimoptions('linprog',  'Display','none');
opt_qp = optimoptions('quadprog', 'Display','off');

%% Initialize
sim_steps = N_total - Np;

% State is tracked as PURE ERROR natively [e_vy; e_psi; e_r; e_Y; e_X]
x_kin = zeros(3, 1);    
x_dyn = zeros(5, 1);    

x_kin_est = zeros(3, 1);
x_dyn_est = zeros(5, 1);

x_kin_est_prev = x_kin_est;
x_dyn_mpc_prev = [v_ref(1)*sin(atan2(l2*tan(delta_r_ref(1)), l1+l2)); 0; psi_dot_ref(1); 0; 0];

X_global   = X_ref(1);
Y_global   = Y_ref(1);
psi_global = psi_ref(1);

z_uio = zeros(5, 1);

if v_ref(1) < v_lo
    active_model = 'kinematic';
else
    active_model = 'dynamic';
end

u_prev = [v_ref(1); delta_r_ref(1)];

vecyc_kin = zeros(tau * pp_kin, 1);
vecuc_kin = zeros((tau-1) * m_in, 1);

vecyc_dyn     = zeros(tau * pp_dyn, 1);
vec_known_dyn = zeros((tau-1) * 5, 1);

steps_in_mode = 0;
delta_U_guess = zeros(2*Np, 1);

% Logs
e_x_log         = zeros(sim_steps, 1);
e_psi_log       = zeros(sim_steps, 1);
e_y_log         = zeros(sim_steps, 1);
ys_ex_log       = zeros(sim_steps, 1);
ybar_c_ex_log   = zeros(sim_steps, 1);
ys_epsi_log     = zeros(sim_steps, 1);
ybar_c_epsi_log = zeros(sim_steps, 1);
ys_ey_log       = zeros(sim_steps, 1);
ybar_c_ey_log   = zeros(sim_steps, 1);
Y_log           = zeros(sim_steps, 1);
X_log           = zeros(sim_steps, 1);
psi_log         = zeros(sim_steps, 1);
v_log           = zeros(sim_steps, 1);
u_log           = zeros(sim_steps, 2);
ua_log          = zeros(sim_steps, 2);
du_log          = zeros(sim_steps, 2);
model_log       = strings(sim_steps, 1);

fprintf('UNIFIED Global-Pose MPC + 2D UIO (%d steps)...\n', sim_steps);

%% Main Simulation Loop
for k = 1 : sim_steps

    vx_now        = u_prev(1);
    steps_in_mode = steps_in_mode + 1;

    %% Model Switching
    if strcmp(active_model, 'kinematic') && vx_now >= v_hi
        active_model = 'dynamic';
        % Pure error transition
        x_dyn     = [0; x_kin(3); 0; x_kin(2); x_kin(1)];
        x_dyn_est = [0; x_kin_est(3); 0; x_kin_est(2); x_kin_est(1)];
        
        v_y_r_k = v_ref(k) * sin(atan2(l2 * tan(delta_r_ref(k)), l1+l2));
        x_dyn_mpc_prev = [x_dyn_est(1) + v_y_r_k; x_dyn_est(2); x_dyn_est(3) + psi_dot_ref(k); x_dyn_est(4); x_dyn_est(5)];

        cp_k = cos(psi_ref(k)); sp_k = sin(psi_ref(k));
        C_dyn_t = [0 0 0 sp_k cp_k; 0 1 0 0 0; 0 0 0 cp_k -sp_k];
        
        M_sw  = interpGain2D(v_ref(k), psi_ref(k), vx_grid, psi_grid, M_schedule);
        z_uio = x_dyn_est - M_sw * (C_dyn_t * x_dyn_est);

        vecyc_dyn     = zeros(tau * pp_dyn, 1);
        vec_known_dyn = zeros((tau-1) * 5, 1);
        steps_in_mode = 0;

    elseif strcmp(active_model, 'dynamic') && vx_now <= v_lo
        active_model = 'kinematic';

        x_kin = [x_dyn(5); x_dyn(4); x_dyn(2)];
        x_kin_est = [x_dyn_est(5); x_dyn_est(4); x_dyn_est(2)];
        x_kin_est_prev = x_kin_est;

        vecyc_kin = zeros(tau * pp_kin, 1);
        vecuc_kin = zeros((tau-1) * m_in, 1);
        steps_in_mode = 0;
    end

    enable_attacks = (steps_in_mode > tau);

    % Output maps (Global -> Sensors)
    cp_k = cos(psi_ref(k)); sp_k = sin(psi_ref(k));
    R_path = [ cp_k, sp_k, 0; 0, 0, 1; -sp_k, cp_k, 0];
    C_dyn_t = [0, 0, 0, sp_k, cp_k; 0, 1, 0, 0, 0; 0, 0, 0, cp_k, -sp_k];

    %% Plant Measurement & Attack Injection
    if strcmp(active_model, 'kinematic')
        y_true = R_path * x_kin;            
    else
        y_true = C_dyn_t * x_dyn;             
    end

    if enable_attacks
        Rho_s = diag(double(rand(pp_kin, 1) >= p_attack_s));
        rho_a = double(rand(m_in, 1) >= p_attack_a);   
    else
        Rho_s = eye(pp_kin); rho_a = ones(m_in, 1);
    end
    
    y_corrupted = Rho_s * y_true;
    e_s_true    = y_corrupted - y_true;

    %% Sparse Recovery + State Estimation
    if strcmp(active_model, 'kinematic')
        % GTZ LINEARIZATION
        [Ad_sr_kin, Bd_sr_kin] = gtzLinearizeKinematic(X_ref(k), Y_ref(k), psi_ref(k), v_ref(k), delta_r_ref(k), l, l2, delta_t);
        vecyc_kin = [y_corrupted;  vecyc_kin(1 : end - pp_kin)];

        if steps_in_mode >= tau
            C_seq = zeros(pp_kin, 3, tau);
            for i_tau = 1:tau
                psi_i = psi_ref(max(1, k - i_tau + 1));
                C_seq(:,:,i_tau) = [cos(psi_i), sin(psi_i), 0; 0, 0, 1; -sin(psi_i), cos(psi_i), 0];
            end
            
            [V_kin, Ob_kin] = buildMarkovObsv_TV(Ad_sr_kin, Bd_sr_kin, C_seq, tau, m_in, pp_kin);
            est_e_s_kin     = sparseRecover(vecyc_kin, vecuc_kin, V_kin, Ob_kin, tau, pp_kin, opt_lp);
        else
            est_e_s_kin = zeros(pp_kin, 1);
        end

        y_corrected_raw = y_corrupted - est_e_s_kin;
        y_corrected = alpha_lpf * y_corrected_raw + (1 - alpha_lpf) * y_corrected_prev;
        y_corrected_prev = y_corrected;

        ys_ex_log(k)       = y_true(1);
        ybar_c_ex_log(k)   = y_corrected(1);
        ys_epsi_log(k)     = y_true(2);
        ybar_c_epsi_log(k) = y_corrected(2);
        ys_ey_log(k)       = y_true(3);
        ybar_c_ey_log(k)   = y_corrected(3);

        ex_c = y_corrected(1); 
        ey_c = y_corrected(3);
        x_kin_est(1) = ex_c * cp_k - ey_c * sp_k;
        x_kin_est(2) = ex_c * sp_k + ey_c * cp_k;
        x_kin_est(3) = y_corrected(2);

        atk_sensor_log(k, :) = e_s_true'; 
        est_sensor_log(k, :) = est_e_s_kin';
        atk_act_log(k, :)    = ((rho_a - 1) .* u_prev)';
        est_err_log(k, 1:3)  = (x_kin - x_kin_est)';

    else
        % GTZ LINEARIZATION
        [Ad_sr_dyn, Bd_sr_dyn] = gtzLinearizeDynamic(X_ref(k), Y_ref(k), psi_ref(k), v_ref(k), psi_dot_ref(k), delta_r_ref(k), Cf, Cr, m, Iz, l1, l2, delta_t);
        
        vecyc_dyn = [y_corrupted;  vecyc_dyn(1 : end - pp_dyn)];

        if steps_in_mode >= tau
            C_seq_dyn = zeros(pp_dyn, 5, tau);
            for i_tau = 1:tau
                psi_i = psi_ref(max(1, k - i_tau + 1));
                c_i = cos(psi_i); s_i = sin(psi_i);
                C_seq_dyn(:,:,i_tau) = [0, 0, 0, s_i, c_i; 0, 1, 0, 0, 0; 0, 0, 0, c_i, -s_i];
            end
            
            [V_dyn, Ob_dyn] = buildMarkovObsv_TV(Ad_sr_dyn, eye(5), C_seq_dyn, tau, 5, pp_dyn);
            est_e_s_dyn     = sparseRecover(vecyc_dyn, vec_known_dyn, V_dyn, Ob_dyn, tau, pp_dyn, opt_lp);
        else
            est_e_s_dyn = zeros(pp_dyn, 1);
        end

        y_corrected_raw = y_corrupted - est_e_s_dyn;
        y_corrected = alpha_lpf * y_corrected_raw + (1 - alpha_lpf) * y_corrected_prev;
        y_corrected_prev = y_corrected;

        ys_ex_log(k)       = y_true(1);
        ybar_c_ex_log(k)   = y_corrected(1);
        ys_epsi_log(k)     = y_true(2);
        ybar_c_epsi_log(k) = y_corrected(2);
        ys_ey_log(k)       = y_true(3);
        ybar_c_ey_log(k)   = y_corrected(3);


        atk_sensor_log(k, :) = e_s_true'; 
        est_sensor_log(k, :) = est_e_s_dyn';
        atk_act_log(k, :)    = ((rho_a - 1) .* u_prev)';
        est_err_log(k, :)    = (x_dyn - x_dyn_est)';

        % GLOBAL-POSE UIO ESTIMATION
        M_now = interpGain2D(v_ref(k), psi_ref(k), vx_grid, psi_grid, M_schedule);
        x_dyn_est = z_uio + M_now * y_corrected;
    end

    %% Build MPC augmented state and prediction matrices
    if strcmp(active_model, 'kinematic')
        y_out_est = R_path * x_kin_est;
        dx_est    = x_kin_est - x_kin_est_prev;
        x_a       = [dx_est; y_out_est];
        Q_bar     = Q_kin_bar; R_bar = R_kin_bar; n_st = 3;
    else
        % RECONSTRUCT ABSOLUTE STATE FOR GTZ MPC
        v_y_r_k = v_ref(k) * sin(atan2(l2 * tan(delta_r_ref(k)), l1+l2));
        x_dyn_mpc = [x_dyn_est(1) + v_y_r_k; 
                     x_dyn_est(2); 
                     x_dyn_est(3) + psi_dot_ref(k); 
                     x_dyn_est(4); 
                     x_dyn_est(5)];

        y_out_est = C_dyn_t * x_dyn_est;
        dx_est    = x_dyn_mpc - x_dyn_mpc_prev;
        x_a       = [dx_est; y_out_est];
        x_dyn_mpc_prev = x_dyn_mpc;
        Q_bar     = Q_dyn_bar; R_bar = R_dyn_bar; n_st = 5;
    end

    e_x_log(k)   = y_out_est(1);
    e_psi_log(k) = y_out_est(2);
    e_y_log(k)   = y_out_est(3);
    Y_log(k)     = Y_global;
    X_log(k)     = X_global;
    psi_log(k)   = psi_global;
    v_log(k)     = vx_now;
    model_log(k) = active_model;

    Phi_a_seq   = cell(Np, 1); 
    Gamma_a_seq = cell(Np, 1);
    p = 3; 
    C_a = [zeros(p, n_st), eye(p)];

    for i = 1:Np
        idx     = min(k + i - 1, length(v_ref));
        v_r     = max(v_ref(idx), vx_min_kin);
        psi_r   = psi_ref(idx);
        delta_r = delta_r_ref(idx);

        if strcmp(active_model, 'kinematic')
            [Phi, Gamma] = gtzLinearizeKinematic(X_ref(idx), Y_ref(idx), psi_r, v_r, delta_r, l, l2, delta_t);
            C_mat_i = [cos(psi_r), sin(psi_r), 0; 0, 0, 1; -sin(psi_r), cos(psi_r), 0];
        else
            [Phi, Gamma] = gtzLinearizeDynamic(X_ref(idx), Y_ref(idx), psi_r, v_r, psi_dot_ref(idx), delta_r, Cf, Cr, m, Iz, l1, l2, delta_t);
            C_mat_i = [0, 0, 0, sin(psi_r), cos(psi_r); 0, 1, 0, 0, 0; 0, 0, 0, cos(psi_r), -sin(psi_r)];
        end

        Phi_a_seq{i}   = [Phi, zeros(n_st, p); C_mat_i * Phi, eye(p)];
        Gamma_a_seq{i} = [Gamma; C_mat_i * Gamma];
    end

    n_a = n_st + p; 
    W = zeros(p * Np, n_a); 
    Z = zeros(p * Np, m_in * Np);
    Phi_prod = eye(n_a);
    for i = 1:Np
        Phi_prod = Phi_a_seq{i} * Phi_prod;
        W((i-1)*p+1 : i*p, :) = C_a * Phi_prod;
        for j = 1:i
            if i == j, temp_prod = eye(n_a);
            else
                temp_prod = eye(n_a);
                for nn_idx = i:-1:j+1, temp_prod = temp_prod * Phi_a_seq{nn_idx}; end
            end
            Z((i-1)*p+1 : i*p, (j-1)*m_in+1 : j*m_in) = C_a * temp_prod * Gamma_a_seq{j};
        end
    end

    %% Constraints & Optimization
    lb_du = repmat([-dv_max; -du_max], Np, 1); ub_du = repmat([ dv_max;  du_max], Np, 1);
    E_mat = kron(ones(Np, 1), eye(m_in)); H_mat = kron(tril(ones(Np)), eye(m_in));
    
    slack = 1e-3;
    U_max_vec = repmat([v_max + slack;  u_max + slack], Np, 1); 
    U_min_vec = repmat([v_min - slack; -u_max - slack], Np, 1);
    
    A_ineq = [H_mat; -H_mat]; b_ineq = [U_max_vec - E_mat * u_prev; -U_min_vec + E_mat * u_prev];

    H_qp = Z' * Q_bar * Z + R_bar; 
    H_qp = (H_qp + H_qp') / 2 + 1e-5 * eye(m_in * Np); 
    f_qp = Z' * Q_bar * (W * x_a);
    
    [delta_U_opt, ~, exitflag] = quadprog(H_qp, f_qp, A_ineq, b_ineq, [], [], lb_du, ub_du, [], opt_qp);

    if exitflag <= 0
        warning('Optimization failed at step %d. Coasting.', k); du_now = [0; 0];
    else 
        du_now = delta_U_opt(1:2); 
    end

    u_command = u_prev + du_now; u_actuated = rho_a .* u_command;    

    %% Plant Propagation (Nonlinear Absolute Pose Integration)
    x_kin_est_prev = x_kin_est; x_dyn_est_prev = x_dyn_est;
    vx_act = u_actuated(1); delta_act = u_actuated(2);

    if strcmp(active_model, 'kinematic')
        vx_kin = max(vx_act, vx_min_kin); beta_now = atan2(l2 * tan(delta_act), l);
        ode_k = @(~, s) [vx_kin * cos(s(3) + beta_now); vx_kin * sin(s(3) + beta_now); (vx_kin / l2) * sin(beta_now)];
        [~, s] = ode23s(ode_k, [0, delta_t], [X_global; Y_global; psi_global]);
        X_global = s(end, 1); 
        Y_global = s(end, 2); 
        psi_global = s(end, 3);
        x_kin = [X_global - X_ref(k+1); Y_global - Y_ref(k+1); angdiff(psi_ref(k+1), psi_global)];

        u_sr_now  = u_command - [v_ref(k); delta_r_ref(k)];
        vecuc_kin = [u_sr_now;  vecuc_kin(1 : end - m_in)];
    else
        vx_pl = max(vx_act, vx_min_dyn);
        
        % Reconstruct Absolute States for Integration
        v_y_r_k = v_ref(k) * sin(atan2(l2 * tan(delta_r_ref(k)), l1+l2));
        v_y_abs = x_dyn(1) + v_y_r_k;
        psi_abs = x_dyn(2) + psi_ref(k); 
        r_abs   = x_dyn(3) + psi_dot_ref(k);
        Y_abs   = x_dyn(4) + Y_ref(k); 
        X_abs   = x_dyn(5) + X_ref(k);
        
        s0 = [v_y_abs; psi_abs; r_abs; Y_abs; X_abs];

        ode_dyn = @(~, xs) [ 
            -(2*Cf+2*Cr)/(m*vx_pl)*xs(1) - ((2*Cf*l1-2*Cr*l2)/(m*vx_pl) + vx_pl)*xs(3) + (2*Cf/m)*delta_act;
             xs(3);
            -(2*l1*Cf-2*l2*Cr)/(Iz*vx_pl)*xs(1) - (2*l1^2*Cf+2*l2^2*Cr)/(Iz*vx_pl)*xs(3) + (2*l1*Cf/Iz)*delta_act;
             vx_pl*sin(xs(2)) + xs(1)*cos(xs(2));
             vx_pl*cos(xs(2)) - xs(1)*sin(xs(2)) ];

        [~, s] = ode23s(ode_dyn, [0, delta_t], s0); 
        s_end = s(end,:)';
        psi_global = s_end(2); 
        Y_global = s_end(4); 
        X_global = s_end(5);
        
        v_y_r_k1 = v_ref(k+1) * sin(atan2(l2 * tan(delta_r_ref(k+1)), l1+l2));
        x_dyn = [s_end(1) - v_y_r_k1; 
                 angdiff(psi_ref(k+1), psi_global); 
                 s_end(3) - psi_dot_ref(k+1); 
                 Y_global - Y_ref(k+1); 
                 X_global - X_ref(k+1)];

        % Sparse recovery
        u_sr_now  = u_command - [v_ref(k); delta_r_ref(k)];
        
        % evaluate matrices at current step k to perfectly match sparse prediction buffer 
        d_now     = Bd_sr_dyn * u_sr_now;
        vec_known_dyn = [d_now; vec_known_dyn(1 : end - 5)];

        % GLOBAL-POSE UIO PROPAGATION
        L_prop = interpGain2D(v_ref(k), psi_ref(k), vx_grid, psi_grid, L_schedule);
        M_prop = interpGain2D(v_ref(k), psi_ref(k), vx_grid, psi_grid, M_schedule);
        P_t_global = eye(5) - M_prop * C_dyn_t;

        % Propagate using UIO dynamics
        z_uio = P_t_global * (Ad_sr_dyn * z_uio + Ad_sr_dyn * M_prop * y_corrected + Bd_sr_dyn * u_sr_now) ...
              + L_prop * (y_corrected - C_dyn_t * (z_uio + M_prop * y_corrected));
    end

    u_log(k, :) = u_command'; du_log(k, :) = du_now'; u_prev = u_command;
    delta_U_guess = [delta_U_opt(m_in+1:end); zeros(m_in, 1)];
end

fprintf('Simulation complete.\n');

%% Plots
t_plot  = t_uniform(1:sim_steps); 
kin_idx = model_log == "kinematic"; 
dyn_idx = model_log == "dynamic";

figure(1);
subplot(2,1,1);
plot(t_plot, v_ref(1:sim_steps), 'k--', 'LineWidth', 2); hold on;
scatter(t_plot(kin_idx), u_log(kin_idx,1), 15, 'b', 'filled');
scatter(t_plot(dyn_idx), u_log(dyn_idx,1), 15, 'r', 'filled');
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$v_x$ (m/s)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);
legend('Reference', 'Kinematic', 'Dynamic', 'FontName','Times New Roman' ,'FontSize',20, 'Location','best');

subplot(2,1,2);
plot(t_plot, rad2deg(u_log(:,2)), 'k-', 'LineWidth', 2); hold on;
yline( rad2deg(u_max),  'r--', 'LineWidth',2); yline(-rad2deg(u_max),  'r--', 'LineWidth',2);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$\delta_f$ (deg)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);
ylim([-35 35]);
xlabel('Time (s)', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 28);

% subplot(3,1,3); 
% plot(t_plot, rad2deg(du_log(:, 2)), 'k-', 'LineWidth', 2);
% ax = gca; ax.FontSize = 20; hold on; grid on;
% yline( rad2deg(du_max),  'r--', 'LineWidth', 2); 
% yline(-rad2deg(du_max),  'r--', 'LineWidth', 2);
% ylabel('$\Delta\delta_f$ (deg)', 'Interpreter', 'latex'); 
% xlabel('Time (s)', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 28);
% ylim([-15 15]);


% Figure 3: sensor attacks vs sparse recovery
figure(2);
subplot(3,1,1);
plot(t_plot, atk_sensor_log(:,1), 'r-', 'LineWidth', 1.5); hold on;
plot(t_plot, est_sensor_log(:,1), 'b--', 'LineWidth', 1.5);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$e_{s,1}$', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28); % e_x channel
legend('True drop', 'Estimated', 'FontName','Times New Roman' ,'FontSize',20, 'Location','best');

subplot(3,1,2);
plot(t_plot, atk_sensor_log(:,2), 'r-', 'LineWidth', 1.5); hold on;
plot(t_plot, est_sensor_log(:,2), 'b--', 'LineWidth', 1.5);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$e_{s,2}$', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);   % e_psi channel

subplot(3,1,3);
plot(t_plot, atk_sensor_log(:,3), 'r-', 'LineWidth', 1.5); hold on;
plot(t_plot, est_sensor_log(:,3), 'b--', 'LineWidth', 1.5);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$e_{s,3}$', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28); % e_y channel
xlabel('Time (s)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);


% Figure 4: output recovery
figure(3);
subplot(3,1,1);
plot(t_plot, ys_ex_log, 'b-', 'LineWidth', 1.5); hold on;
plot(t_plot, ybar_c_ex_log, 'r--', 'LineWidth', 1.5);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$y^s_{1}, \tilde{y}^s_{1}$ (m)', 'Interpreter', 'latex', 'FontName','Times New Roman' ,'FontSize',28); % e_x channel
legend('$y^s_{1}$ (True)', '$\tilde{y}^s_{1}$ (Estimated)', 'Interpreter', 'latex', 'FontName','Times New Roman' ,'FontSize',16, 'Location','best');

subplot(3,1,2);
plot(t_plot, ys_epsi_log, 'b-', 'LineWidth', 1.5); hold on;
plot(t_plot, ybar_c_epsi_log, 'r--', 'LineWidth', 1.5);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$y^s_{2}, \tilde{y}^s_{2}$ (rad)', 'Interpreter', 'latex', 'FontName','Times New Roman' ,'FontSize',28); % e_psi channel
legend('$y^s_{2}$ (True)', '$\tilde{y}^s_{2}$ (Estimated)', 'Interpreter', 'latex', 'FontName','Times New Roman' ,'FontSize',16, 'Location','best');

subplot(3,1,3);
plot(t_plot, ys_ey_log, 'b-', 'LineWidth', 1.5); hold on;
plot(t_plot, ybar_c_ey_log, 'r--', 'LineWidth', 1.5);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$y^s_{3}, \tilde{y}^s_{3}$ (m)', 'Interpreter', 'latex', 'FontName','Times New Roman' ,'FontSize',28); % e_y channel
legend('$y^s_{3}$ (True)', '$\tilde{y}^s_{3}$ (Estimated)', 'Interpreter', 'latex', 'FontName','Times New Roman' ,'FontSize',16, 'Location','best');
xlabel('Time (s)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28); 


%% ===== Helper Functions =====

function K = interpGain2D(vx_q, psi_q, vx_grid, psi_grid, K_schedule)
    psi_q = atan2(sin(psi_q), cos(psi_q));
    vx_q  = min(max(vx_q, vx_grid(1)), vx_grid(end));
    [~, i_vx]  = min(abs(vx_grid  - vx_q));
    [~, j_psi] = min(abs(psi_grid - psi_q));
    K = K_schedule(:, :, i_vx, j_psi);
end

function [Ad, Bd] = gtzLinearizeDynamic(X_r, Y_r, psi_r, v_r, psi_dot_r, delta_r, Cf, Cr, m, Iz, l1, l2, dt)
    if v_r < 0.5, v_r = 0.5; end
    l = l1 + l2; 
    beta_r = atan2(l2 * tan(delta_r), l); 
    v_y_r  = v_r * sin(beta_r);
    x_r_i  = [v_y_r; psi_r; psi_dot_r; Y_r; X_r]; 
    mu_r_i = [v_r; delta_r];
    cp = cos(psi_r); sp = sin(psi_r);

    A_c = zeros(5, 5);
    A_c(1,1) = -(2*Cf+2*Cr)/(m*v_r); A_c(1,3) = -(2*Cf*l1-2*Cr*l2)/(m*v_r) - v_r;
    A_c(2,3) = 1;
    A_c(3,1) = -(2*l1*Cf-2*l2*Cr)/(Iz*v_r); A_c(3,3) = -(2*l1^2*Cf+2*l2^2*Cr)/(Iz*v_r);
    A_c(4,1) = cp; A_c(4,2) = v_r*cp - v_y_r*sp;
    A_c(5,1) = -sp; A_c(5,2) = -v_r*sp - v_y_r*cp;

    df1_dvx = (2*Cf+2*Cr)/(m*v_r^2)*v_y_r + ((2*Cf*l1-2*Cr*l2)/(m*v_r^2)-1)*psi_dot_r;
    df3_dvx = (2*l1*Cf-2*l2*Cr)/(Iz*v_r^2)*v_y_r + (2*l1^2*Cf+2*l2^2*Cr)/(Iz*v_r^2)*psi_dot_r;
    B_v = [df1_dvx; 0; df3_dvx; sp; cp]; B_steer = [2*Cf/m; 0; 2*l1*Cf/Iz; 0; 0];
    B_c = [B_v, B_steer];

    f_r = [-(2*Cf+2*Cr)/(m*v_r)*v_y_r - ((2*Cf*l1-2*Cr*l2)/(m*v_r) + v_r)*psi_dot_r + (2*Cf/m)*delta_r;
           psi_dot_r;
          -(2*l1*Cf-2*l2*Cr)/(Iz*v_r)*v_y_r - (2*l1^2*Cf+2*l2^2*Cr)/(Iz*v_r)*psi_dot_r + (2*l1*Cf/Iz)*delta_r;
           v_r*sp + v_y_r*cp;
           v_r*cp - v_y_r*sp];

    phi_xT = f_r - A_c * x_r_i - B_c * mu_r_i; denom = x_r_i' * x_r_i;
    A_GTZ = A_c + (phi_xT * x_r_i') / denom; 

    sys_c_dyn = ss(A_GTZ, B_c, eye(5), zeros(5,2));
    sys_d_dyn = c2d(sys_c_dyn, dt, 'zoh');
    
    Ad = sys_d_dyn.A; 
    Bd = sys_d_dyn.B;

end

function [Ad, Bd] = gtzLinearizeKinematic(X_r, Y_r, psi_r, vr, delta_r, l, l2, dt)
    if vr < 0.5, vr = 0.5; end
    beta_r = atan2(l2 * tan(delta_r), l); 
    dbeta_ddelta = l2 * l / (l^2 * cos(delta_r)^2 + l2^2 * sin(delta_r)^2);

    % Continuous Taylor A and B Matrices
    A_T = [0, 0, -vr * sin(psi_r + beta_r);
           0, 0,  vr * cos(psi_r + beta_r);
           0, 0,  0];

    B_T = [cos(psi_r + beta_r),  -vr * sin(psi_r + beta_r) * dbeta_ddelta;
           sin(psi_r + beta_r),   vr * cos(psi_r + beta_r) * dbeta_ddelta;
           sin(beta_r) / l2,     (vr / l2) * cos(beta_r) * dbeta_ddelta];

    x_r_i  = [X_r; Y_r; psi_r];
    mu_r_i = [vr; delta_r];

    f_r = [vr * cos(psi_r + beta_r);
           vr * sin(psi_r + beta_r);
           (vr / l2) * sin(beta_r)];

    phi_xT = f_r - A_T * x_r_i - B_T * mu_r_i;
    denom  = x_r_i' * x_r_i;
    A_GTZ = A_T + (phi_xT * x_r_i') / denom;

    sys_c_kin = ss(A_GTZ, B_T, eye(3), zeros(3,2));
    sys_d_kin = c2d(sys_c_kin, dt, 'zoh');
    
    Ad = sys_d_kin.A; 
    Bd = sys_d_kin.B;
end

function [V, Ob] = buildMarkovObsv_TV(Ad, Bd, C_seq, tau, m1, pp)
    n_st = size(Ad, 1); 
    Ob = zeros(tau * pp, n_st);
    
    for i = 1:tau 
        Ob((i-1)*pp+1 : i*pp, :) = C_seq(:,:,i) * Ad^(tau-i); 
    end
    
    V = zeros(tau * pp, (tau-1) * m1);
    for i = 1:tau-1
        for j = 1:(tau-i)
            col_idx = i + j - 1; 
            V((i-1)*pp+1 : i*pp, (col_idx-1)*m1+1 : col_idx*m1) = C_seq(:,:,i) * Ad^(j-1) * Bd;
        end
    end
end

function est_e_s = sparseRecover(vecyc, vecuc, V, Ob, tau, pp, opt_lp)
    Q2 = null(Ob');
    
    if isempty(Q2) 
        est_e_s = zeros(pp, 1); 
        return; 
    end
    
    Zt = Q2' * (vecyc - V * vecuc); 
    Wt = Q2';
    n_var = tau * pp;
    
    E_split = linprog(ones(1, 2*n_var), [], [], [Wt, -Wt], Zt, zeros(2*n_var, 1), [], opt_lp);
    
    if ~isempty(E_split) 
        E_hat = E_split(1:n_var) - E_split(n_var+1:end); 
        est_e_s = E_hat(1:pp); 
    else 
        est_e_s = zeros(pp, 1); 
    end
end