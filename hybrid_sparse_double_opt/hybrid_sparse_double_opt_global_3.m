% MOST UP-TO-DATE
% LTV-MPC with 2-Input [v_x, delta] + UIO + Sparse Error Recovery
% UNIFIED GLOBAL POSE STATE for BOTH Kinematic & Dynamic Modes
% now, the plant output y_out is the vehicle's absolute pose (X,Y \psi)
% rotated by the reference heading into path-aligned axis (acceptable---what we want!)
% Uses 2D Scheduled UIO Gains & Error-Space Isolation for Stability

clc; clear; close all;

%% Vehicle & Controller Parameters
Cf = 30000;   Cr = 30000;
l1 = 1.4225;  l2 = 1.4225;
l  = l1 + l2;
m  = 1280;    Iz = 2500;

delta_t = 0.02;
Np      = 35;

% MPC weights
Q_x_kin = 2000;  Q_psi_kin = 1000;  Q_y_kin = 7500;  
R_dv_kin = 500; R_du_kin = 15000;

Q_x_dyn = 2000;  Q_psi_dyn = 1000;  Q_y_dyn = 7500;  
R_dv_dyn = 500; R_du_dyn = 15000;

% Rate constraints (per step)
dv_max = 2.5 * delta_t;       %acceleration. currently works. was 5 and works
du_max = deg2rad(5);      %steering rate. currently works. was 25 and works

% Absolute constraints
v_min = 0.0;
v_max = 25.0;
u_max = deg2rad(25);        %steering angle

% Hysteresis switching thresholds
v_lo = 6.0;
v_hi = 7.0;

vx_min_dyn = 0.5;
vx_min_kin = 0.0;

% Attack / recovery parameters
tau          = 10;
p_attack_s   = 0.02;          
p_attack_a_v = 0.00;          
p_attack_a_d = 0.01;          
p_attack_a   = [p_attack_a_v; p_attack_a_d];  

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
load("observer_gain_schedule_with_M0.05.mat"); 

%% MPC Matrices & Output Setup
Q_kin_bar = kron(eye(Np), diag([Q_x_kin, Q_psi_kin, Q_y_kin]));
R_kin_bar = kron(eye(Np), diag([R_dv_kin, R_du_kin]));

Q_dyn_bar = kron(eye(Np), diag([Q_x_dyn, Q_psi_dyn, Q_y_dyn]));
R_dyn_bar = kron(eye(Np), diag([R_dv_dyn, R_du_dyn]));

pp_kin = 3;     
pp_dyn = 3;     
m_in   = 2;     

opt_lp = optimoptions('linprog',  'Display','none');
opt_fmincon = optimoptions('fmincon', 'Display','none', 'Algorithm','sqp', 'SpecifyObjectiveGradient',true);

%% Initialize
sim_steps = N_total - Np;

X_global   = X_ref(1);
Y_global   = Y_ref(1);
psi_global = psi_ref(1);

% ABSOLUTE STATES
x_kin = [X_global; Y_global; psi_global];

beta_init  = atan2(l2 * tan(delta_r_ref(1)), l);
v_y_init   = v_ref(1) * sin(beta_init);
r_init     = v_ref(1) / l2 * sin(beta_init);
x_dyn      = [v_y_init; psi_global; r_init; Y_global; X_global];

x_kin_err_est = zeros(3, 1);
x_dyn_err_est = zeros(5, 1);

x_kin_est = x_kin;
x_dyn_est = x_dyn;

x_kin_est_prev = x_kin_est;
x_dyn_est_prev = x_dyn_est;

M0 = interpGain2D(v_ref(1), psi_ref(1), vx_grid, psi_grid, M_schedule);
C0_dyn = [0 0 0 sin(psi_ref(1)) cos(psi_ref(1)); 
          0 1 0 0 0; 
          0 0 0 cos(psi_ref(1)) -sin(psi_ref(1))];
z_uio_err = x_dyn_err_est - M0 * (C0_dyn * x_dyn_err_est);

if v_ref(1) < v_lo
    active_model = 'kinematic';
else
    active_model = 'dynamic';
end

u_prev = [v_ref(1); delta_r_ref(1)];

vecyc_err_kin = zeros(tau * pp_kin, 1);
vecuc_err_kin = zeros((tau-1) * m_in, 1);

vecyc_err_dyn     = zeros(tau * pp_dyn, 1);
vec_known_err_dyn = zeros((tau-1) * 5, 1);

steps_in_mode = 0;
delta_U_guess = zeros(2*Np, 1);

% Logs
e_x_log         = zeros(sim_steps, 1);
e_psi_log       = zeros(sim_steps, 1);
e_y_log         = zeros(sim_steps, 1);
ys_x_log        = zeros(sim_steps, 1);
ybar_c_x_log    = zeros(sim_steps, 1);
ys_psi_log      = zeros(sim_steps, 1);
ybar_c_psi_log  = zeros(sim_steps, 1);
ys_y_log        = zeros(sim_steps, 1);
ybar_c_y_log    = zeros(sim_steps, 1);
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

    %% Model Switching (Absolute Pose Transfer)
    if strcmp(active_model, 'kinematic') && vx_now >= v_hi
        active_model = 'dynamic';
        beta_prev  = atan2(l2 * tan(u_prev(2)), l);
        r_init     = max(vx_now, vx_min_dyn) / l2 * sin(beta_prev);  
        v_y_init   = max(vx_now, vx_min_dyn) * sin(beta_prev);  
        
        x_dyn     = [v_y_init; x_kin(3); r_init; x_kin(2); x_kin(1)];
        
        v_y_r_k = v_ref(k) * sin(atan2(l2 * tan(delta_r_ref(k)), l1+l2));
        x_dyn_err_est = [v_y_init - v_y_r_k; x_kin_err_est(3); r_init - psi_dot_ref(k); x_kin_err_est(2); x_kin_err_est(1)];
        
        x_ref_dyn = [v_y_r_k; psi_ref(k); psi_dot_ref(k); Y_ref(k); X_ref(k)];
        x_dyn_est = x_dyn_err_est + x_ref_dyn;
        x_dyn_est_prev = x_dyn_est;
        
        cp_k = cos(psi_ref(k)); sp_k = sin(psi_ref(k));
        C_mat_dyn = [0 0 0 sp_k cp_k; 
                    0 1 0 0 0; 
                    0 0 0 cp_k -sp_k];
        
        M_sw  = interpGain2D(v_ref(k), psi_ref(k), vx_grid, psi_grid, M_schedule);
        z_uio_err = x_dyn_err_est - M_sw * (C_mat_dyn * x_dyn_err_est);

        vecyc_err_dyn     = zeros(tau * pp_dyn, 1);
        vec_known_err_dyn = zeros((tau-1) * 5, 1);
        steps_in_mode = 0;

    elseif strcmp(active_model, 'dynamic') && vx_now <= v_lo
        active_model = 'kinematic';

        x_kin = [x_dyn(5); x_dyn(4); x_dyn(2)];
        x_kin_err_est = [x_dyn_err_est(5); x_dyn_err_est(4); x_dyn_err_est(2)];
        
        x_ref_kin = [X_ref(k); Y_ref(k); psi_ref(k)];
        x_kin_est = x_kin_err_est + x_ref_kin;
        x_kin_est_prev = x_kin_est;

        vecyc_err_kin = zeros(tau * pp_kin, 1);
        vecuc_err_kin = zeros((tau-1) * m_in, 1);
        steps_in_mode = 0;
    end

    % Path-frame setpoint at the CURRENT step:  y_ref = C * x_ref
    cp0 = cos(psi_ref(k));  sp0 = sin(psi_ref(k));
    y_ref_now = [ cp0*X_ref(k) + sp0*Y_ref(k);
                  psi_ref(k);
                 -sp0*X_ref(k) + cp0*Y_ref(k) ];

    enable_attacks = (steps_in_mode > tau);

    % Output maps (Global -> Sensors / Path-Aligned Output)
    C_mat_kin = [ cp0, sp0, 0; 
                0, 0, 1; 
                -sp0, cp0, 0];
    C_mat_dyn = [0, 0, 0, sp0, cp0; 
                0, 1, 0, 0, 0; 
                0, 0, 0, cp0, -sp0];

    %% Plant Measurement & Attack Injection
    if strcmp(active_model, 'kinematic')
        y_true = C_mat_kin * x_kin;          % Rotated Absolute Pose 
    else
        y_true = C_mat_dyn * x_dyn;             
    end

    if enable_attacks
        Rho_s = diag(double(rand(pp_kin, 1) >= p_attack_s));
        rho_a = double(rand(m_in, 1) >= p_attack_a);   
    else
        Rho_s = eye(pp_kin); rho_a = ones(m_in, 1);
    end
    
    y_corrupted     = Rho_s * y_true;       
    e_s_true        = y_corrupted - y_true;
    
    % ERROR-SPACE ISOLATION (Bypasses LP Float Limit)
    y_err_corrupted = y_corrupted - y_ref_now;
    u_err_now       = u_prev - [v_ref(k); delta_r_ref(k)];

    %% Sparse Recovery + State Estimation
    if strcmp(active_model, 'kinematic')
        [Ad_sr_kin, Bd_sr_kin] = gtzLinearizeKinematic(X_ref(k), Y_ref(k), psi_ref(k), v_ref(k), delta_r_ref(k), l, l2, delta_t);
        
        vecyc_err_kin = [y_err_corrupted;  vecyc_err_kin(1 : end - pp_kin)];

        if steps_in_mode >= tau
            A_seq_kin = zeros(3, 3, tau);
            B_seq_kin = zeros(3, 2, tau);
            C_seq = zeros(pp_kin, 3, tau);
            
            for i_tau = 1:tau
                idx_tau = max(1, k - i_tau + 1);
                [A_seq_kin(:,:,i_tau), B_seq_kin(:,:,i_tau)] = gtzLinearizeKinematic(...
                    X_ref(idx_tau), Y_ref(idx_tau), psi_ref(idx_tau), ...
                    v_ref(idx_tau), delta_r_ref(idx_tau), l, l2, delta_t);
                
                psi_i = psi_ref(idx_tau);
                C_seq(:,:,i_tau) = [cos(psi_i), sin(psi_i), 0; 0, 0, 1; -sin(psi_i), cos(psi_i), 0];
            end
            
            [V_kin, Ob_kin] = buildMarkovObsv_TV(A_seq_kin, B_seq_kin, C_seq, tau, m_in, pp_kin);
            est_e_s_kin     = sparseRecover(vecyc_err_kin, vecuc_err_kin, V_kin, Ob_kin, tau, pp_kin, opt_lp);
        else
            est_e_s_kin = zeros(pp_kin, 1);
        end

        y_err_corrected = y_err_corrupted - est_e_s_kin;

        % Log absolute corrected outputs for plotting
        y_corrected = y_err_corrected + y_ref_now;
        ys_x_log(k)       = y_true(1);
        ybar_c_x_log(k)   = y_corrected(1);
        ys_psi_log(k)     = y_true(2);
        ybar_c_psi_log(k) = y_corrected(2);
        ys_y_log(k)       = y_true(3);
        ybar_c_y_log(k)   = y_corrected(3);

        % Inverse rotate path-aligned error pose back to global error state
        ex_c = y_err_corrected(1); 
        ey_c = y_err_corrected(3);
        x_kin_err_est(1) = ex_c * cp0 - ey_c * sp0;
        x_kin_err_est(2) = ex_c * sp0 + ey_c * cp0;
        x_kin_err_est(3) = y_err_corrected(2);

        % RECONSTRUCT ABSOLUTE STATE FOR MPC
        x_ref_kin = [X_ref(k); Y_ref(k); psi_ref(k)];
        x_kin_est = x_kin_err_est + x_ref_kin;

        atk_sensor_log(k, :) = e_s_true'; 
        est_sensor_log(k, :) = est_e_s_kin';
        atk_act_log(k, :)    = ((rho_a - 1) .* u_prev)';
        est_err_log(k, 1:3)  = (x_kin - x_kin_est)';
        
        vecuc_err_kin = [u_err_now;  vecuc_err_kin(1 : end - m_in)];

    else
        [Ad_sr_dyn, Bd_sr_dyn] = gtzLinearizeDynamic(X_ref(k), Y_ref(k), psi_ref(k), v_ref(k), psi_dot_ref(k), delta_r_ref(k), Cf, Cr, m, Iz, l1, l2, delta_t);
        
        vecyc_err_dyn = [y_err_corrupted;  vecyc_err_dyn(1 : end - pp_dyn)];

        if steps_in_mode >= tau
            A_seq_dyn = zeros(5, 5, tau);
            B_seq_dyn = zeros(5, 5, tau); 
            C_seq_dyn = zeros(pp_dyn, 5, tau);
            
            for i_tau = 1:tau
                idx_tau = max(1, k - i_tau + 1);
                [A_seq_dyn(:,:,i_tau), ~] = gtzLinearizeDynamic(...
                    X_ref(idx_tau), Y_ref(idx_tau), psi_ref(idx_tau), ...
                    v_ref(idx_tau), psi_dot_ref(idx_tau), delta_r_ref(idx_tau), ...
                    Cf, Cr, m, Iz, l1, l2, delta_t);
                
                B_seq_dyn(:,:,i_tau) = eye(5);
                
                psi_i = psi_ref(idx_tau);
                c_i = cos(psi_i); s_i = sin(psi_i);
                C_seq_dyn(:,:,i_tau) = [0, 0, 0, s_i, c_i; 0, 1, 0, 0, 0; 0, 0, 0, c_i, -s_i];
            end
            
            [V_dyn, Ob_dyn] = buildMarkovObsv_TV(A_seq_dyn, B_seq_dyn, C_seq_dyn, tau, 5, pp_dyn);
            est_e_s_dyn     = sparseRecover(vecyc_err_dyn, vec_known_err_dyn, V_dyn, Ob_dyn, tau, pp_dyn, opt_lp);
        else
            est_e_s_dyn = zeros(pp_dyn, 1);
        end

        y_err_corrected = y_err_corrupted - est_e_s_dyn;

        y_corrected = y_err_corrected + y_ref_now;
        ys_x_log(k)       = y_true(1);
        ybar_c_x_log(k)   = y_corrected(1);
        ys_psi_log(k)     = y_true(2);
        ybar_c_psi_log(k) = y_corrected(2);
        ys_y_log(k)       = y_true(3);
        ybar_c_y_log(k)   = y_corrected(3);

        atk_sensor_log(k, :) = e_s_true'; 
        est_sensor_log(k, :) = est_e_s_dyn';
        atk_act_log(k, :)    = ((rho_a - 1) .* u_prev)';
        est_err_log(k, :)    = (x_dyn - x_dyn_est)';

        % ERROR-SPACE UIO ESTIMATION
        M_now = interpGain2D(v_ref(k), psi_ref(k), vx_grid, psi_grid, M_schedule);
        x_dyn_err_est = z_uio_err + M_now * y_err_corrected;
        
        % RECONSTRUCT ABSOLUTE STATE FOR MPC
        v_y_r_k = v_ref(k) * sin(atan2(l2 * tan(delta_r_ref(k)), l1+l2));
        x_ref_dyn = [v_y_r_k; psi_ref(k); psi_dot_ref(k); Y_ref(k); X_ref(k)];
        x_dyn_est = x_dyn_err_est + x_ref_dyn;
        
        d_err_now = Bd_sr_dyn * u_err_now;
        vec_known_err_dyn = [d_err_now; vec_known_err_dyn(1 : end - 5)];

        L_prop = interpGain2D(v_ref(k), psi_ref(k), vx_grid, psi_grid, L_schedule);
        M_prop = interpGain2D(v_ref(k), psi_ref(k), vx_grid, psi_grid, M_schedule);
        P_t_global = eye(5) - M_prop * C_mat_dyn;

        % Propagate using UIO error dynamics
        z_uio_err = P_t_global * (Ad_sr_dyn * z_uio_err + Ad_sr_dyn * M_prop * y_err_corrected + Bd_sr_dyn * u_err_now) ...
                  + L_prop * (y_err_corrected - C_mat_dyn * (z_uio_err + M_prop * y_err_corrected));
    end

    %% Build MPC augmented state and prediction matrices
    if strcmp(active_model, 'kinematic')
        y_out_est = C_mat_kin * x_kin_est;
        dx_est    = x_kin_est - x_kin_est_prev;
        x_a       = [dx_est; y_out_est];
        x_kin_est_prev = x_kin_est;
        Q_bar     = Q_kin_bar; R_bar = R_kin_bar; n_st = 3;
    else
        y_out_est = C_mat_dyn * x_dyn_est;
        dx_est    = x_dyn_est - x_dyn_est_prev;
        x_a       = [dx_est; y_out_est];
        x_dyn_est_prev = x_dyn_est;
        Q_bar     = Q_dyn_bar; R_bar = R_dyn_bar; n_st = 5;
    end

    e_x_log(k)   = y_out_est(1) - y_ref_now(1);
    e_psi_log(k) = angdiff(y_ref_now(2), y_out_est(2));
    e_y_log(k)   = y_out_est(3) - y_ref_now(3);
    Y_log(k)     = Y_global;
    X_log(k)     = X_global;
    psi_log(k)   = psi_global;
    v_log(k)     = vx_now;
    model_log(k) = active_model;

    %% Build Time-Varying Sequences for the Horizon
    Phi_a_seq   = cell(Np, 1); 
    Gamma_a_seq = cell(Np, 1);
    y_ref_seq   = cell(Np, 1);
    y_ref_acc   = y_ref_now; 

    p = 3; 
    C_a = [zeros(p, n_st), eye(p)];

    for i = 1:Np
        idx     = min(k + i - 1, length(v_ref));
        v_r     = max(v_ref(idx), vx_min_kin);
        psi_r   = psi_ref(idx);
        delta_r = delta_r_ref(idx);

        % Accumulate reference setpoint in the path frame step-by-step
        idx_next = min(idx + 1, N_total);
        cpa = cos(psi_r);  
        spa = sin(psi_r);                

        dX_ref   = X_ref(idx_next) - X_ref(idx);
        dY_ref   = Y_ref(idx_next) - Y_ref(idx);
        dpsi_ref = angdiff(psi_ref(idx), psi_ref(idx_next));

        y_ref_acc = y_ref_acc + [ cpa*dX_ref + spa*dY_ref;
                                  dpsi_ref;
                                 -spa*dX_ref + cpa*dY_ref ];
        y_ref_seq{i} = y_ref_acc;

        if strcmp(active_model, 'kinematic')
            [Phi, Gamma] = gtzLinearizeKinematic(X_ref(idx), Y_ref(idx), psi_r, v_r, delta_r, l, l2, delta_t);
            C_mat_i = [cpa, spa, 0; 0, 0, 1; -spa, cpa, 0];
        else
            [Phi, Gamma] = gtzLinearizeDynamic(X_ref(idx), Y_ref(idx), psi_r, v_r, psi_dot_ref(idx), delta_r, Cf, Cr, m, Iz, l1, l2, delta_t);
            C_mat_i = [0, 0, 0, spa, cpa; 0, 1, 0, 0, 0; 0, 0, 0, cpa, -spa];
        end

        Phi_a_seq{i}   = [Phi, zeros(n_st, p); C_mat_i * Phi, eye(p)];
        Gamma_a_seq{i} = [Gamma; C_mat_i * Gamma];
    end

    n_a = n_st + p; 
    W = zeros(p * Np, n_a); 
    Z = zeros(p * Np, m_in * Np);
    Y_ref_horizon = zeros(p * Np, 1);

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
        Y_ref_horizon((i-1)*p+1 : i*p) = y_ref_seq{i};
    end

    %% Constraints & Optimization
    lb_du = repmat([-dv_max; -du_max], Np, 1); 
    ub_du = repmat([ dv_max;  du_max], Np, 1);
    
    E_mat = kron(ones(Np, 1), eye(m_in)); 
    H_mat = kron(tril(ones(Np)), eye(m_in));
    
    U_max_vec = repmat([v_max;  u_max], Np, 1); 
    U_min_vec = repmat([v_min; -u_max], Np, 1);
    
    A_ineq = [H_mat; -H_mat]; 
    b_ineq = [U_max_vec - E_mat * u_prev; 
                -U_min_vec + E_mat * u_prev];

    %% fmincon implementation of custom slides formulation
    cost_func = @(DU) obj_with_gradient(DU, Y_ref_horizon, W * x_a, Z, Q_bar, R_bar);
    
    [delta_U_opt, ~, exitflag] = fmincon(cost_func, delta_U_guess, A_ineq, b_ineq, [], [], lb_du, ub_du, [], opt_fmincon);

    if exitflag <= 0
        warning('Optimization failed at step %d. Coasting.', k); du_now = [0; 0];
    else 
        du_now = delta_U_opt(1:2); 
    end

    u_command = u_prev + du_now; u_actuated = rho_a .* u_command;    

    %% Plant Propagation (Nonlinear Absolute Pose Integration)
    vx_act = u_actuated(1); delta_act = u_actuated(2);

    if strcmp(active_model, 'kinematic')
        vx_kin = max(vx_act, vx_min_kin); 
        beta_now = atan2(l2 * tan(delta_act), l);
        vc_now   = vx_kin / cos(beta_now); % Use CG velocity natively
        ode_k = @(~, s) [vc_now * cos(s(3) + beta_now); vc_now * sin(s(3) + beta_now); (vc_now / l2) * sin(beta_now)];
        [~, s] = ode23s(ode_k, [0, delta_t], x_kin);
        X_global = s(end, 1); 
        Y_global = s(end, 2); 
        psi_global = s(end, 3);
        x_kin = [X_global; Y_global; psi_global];
    else
        vx_pl = max(vx_act, vx_min_dyn);
        
        s0 = x_dyn; % Absolute state
        ode_dyn = @(~, xs) [ 
            -(2*Cf+2*Cr)/(m*vx_pl)*xs(1) - ((2*Cf*l1-2*Cr*l2)/(m*vx_pl) + vx_pl)*xs(3) + (2*Cf/m)*delta_act;
             xs(3);
            -(2*l1*Cf-2*l2*Cr)/(Iz*vx_pl)*xs(1) - (2*l1^2*Cf+2*l2^2*Cr)/(Iz*vx_pl)*xs(3) + (2*l1*Cf/Iz)*delta_act;
             vx_pl*sin(xs(2)) + xs(1)*cos(xs(2));
             vx_pl*cos(xs(2)) - xs(1)*sin(xs(2)) ];

        [~, s] = ode23s(ode_dyn, [0, delta_t], s0); 
        x_dyn = s(end,:)';
        psi_global = x_dyn(2); 
        Y_global = x_dyn(4); 
        X_global = x_dyn(5);
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
subplot(1,1,1);
plot(t_plot, v_ref(1:sim_steps), 'k--', 'LineWidth', 2); hold on;
scatter(t_plot(kin_idx), u_log(kin_idx,1), 15, 'b', 'filled');
scatter(t_plot(dyn_idx), u_log(dyn_idx,1), 15, 'r', 'filled');
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$v_x$ (m/s)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);
legend('Reference', 'Kinematic', 'Dynamic', 'FontName','Times New Roman' ,'FontSize',20, 'Location','best');

% subplot(2,1,2);
% plot(t_plot, rad2deg(u_log(:,2)), 'k-', 'LineWidth', 2); hold on;
% yline( rad2deg(u_max),  'r--', 'LineWidth',2); yline(-rad2deg(u_max),  'r--', 'LineWidth',2);
% ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
% grid on; ylabel('$\delta_f$ (deg)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);
% ylim([-35 35]);
xlabel('Time (s)', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 28);

figure(2);
subplot(3,1,1);
plot(t_plot, atk_sensor_log(:,1), 'r-', 'LineWidth', 2); hold on;
plot(t_plot, est_sensor_log(:,1), 'b--', 'LineWidth', 2);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$e_{s1}, \tilde{e}_{s1}$ (m)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);
legend('True drop', 'Estimated', 'FontName','Times New Roman' ,'FontSize',20, 'Location','best');

subplot(3,1,2);
plot(t_plot, atk_sensor_log(:,2), 'r-', 'LineWidth', 2); hold on;
plot(t_plot, est_sensor_log(:,2), 'b--', 'LineWidth', 2);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$e_{s2}, \tilde{e}_{s2}$ (rad)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);
legend('True drop', 'Estimated', 'FontName','Times New Roman' ,'FontSize',20, 'Location','best');

subplot(3,1,3);
plot(t_plot, atk_sensor_log(:,3), 'r-', 'LineWidth', 2); hold on;
plot(t_plot, est_sensor_log(:,3), 'b--', 'LineWidth', 2);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$e_{s3}, \tilde{e}_{s3}$ (m)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);
xlabel('Time (s)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);
legend('True drop', 'Estimated', 'FontName','Times New Roman' ,'FontSize',20, 'Location','best');

figure(3);
subplot(3,1,1);
plot(t_plot, ys_x_log, 'b-', 'LineWidth', 2); hold on;
plot(t_plot, ybar_c_x_log, 'r--', 'LineWidth', 2);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$y^s_{1}, \tilde{y}^s_{1}$ (m)', 'Interpreter', 'latex', 'FontName','Times New Roman' ,'FontSize',28);
legend('$y^s_{1}$ (True)', '$\tilde{y}^s_{1}$ (Estimated)', 'Interpreter', 'latex', 'FontName','Times New Roman' ,'FontSize',17, 'Location','best');

subplot(3,1,2);
plot(t_plot, ys_psi_log, 'b-', 'LineWidth', 2); hold on;
plot(t_plot, ybar_c_psi_log, 'r--', 'LineWidth', 2);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$y^s_{2}, \tilde{y}^s_{2}$ (rad)', 'Interpreter', 'latex', 'FontName','Times New Roman' ,'FontSize',28);
legend('$y^s_{2}$ (True)', '$\tilde{y}^s_{2}$ (Estimated)', 'Interpreter', 'latex', 'FontName','Times New Roman' ,'FontSize',17, 'Location','best');

subplot(3,1,3);
plot(t_plot, ys_y_log, 'b-', 'LineWidth', 2); hold on;
plot(t_plot, ybar_c_y_log, 'r--', 'LineWidth', 2);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; ylabel('$y^s_{3}, \tilde{y}^s_{3}$ (m)', 'Interpreter', 'latex', 'FontName','Times New Roman' ,'FontSize',28);
legend('$y^s_{3}$ (True)', '$\tilde{y}^s_{3}$ (Estimated)', 'Interpreter', 'latex', 'FontName','Times New Roman' ,'FontSize',17, 'Location','best');
xlabel('Time (s)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28); 


%% ===== Helper Functions =====
function [J, grad] = obj_with_gradient(DU, r_p, W_xa, Z, Q, R)
    Y_pred = W_xa + Z * DU;
    err = r_p - Y_pred;
    
    J = 0.5 * err' * Q * err + 0.5 * DU' * R * DU;
    if nargout > 1
        grad = -Z' * Q * err + R * DU;
    end
end

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

    phi_xT = f_r - A_c * x_r_i - B_c * mu_r_i; 
    denom = x_r_i' * x_r_i;
    A_GTZ = A_c + (phi_xT * x_r_i') / denom; 

    sys_c_dyn = ss(A_GTZ, B_c, eye(5), zeros(5,2));
    sys_d_dyn = c2d(sys_c_dyn, dt, 'zoh');
    
    Ad = sys_d_dyn.A; 
    Bd = sys_d_dyn.B;
end

function [Ad, Bd] = gtzLinearizeKinematic(X_r, Y_r, psi_r, vr, delta_r, l, l2, dt)
    beta_r = atan2(l2 * tan(delta_r), l); 
    dbeta_ddelta = l2 * l / (l^2 * cos(delta_r)^2 + l2^2 * sin(delta_r)^2);

    cp = cos(psi_r);   sp = sin(psi_r);
    tb = tan(beta_r);  sec2b = sec(beta_r)^2;
            
    A_T = [0, 0, -vr * (sp + cp * tb);
           0, 0,  vr * (cp - sp * tb);
           0, 0,  0];
                   
    B_T = [cp - sp * tb,  -vr * sp * sec2b * dbeta_ddelta;
           sp + cp * tb,   vr * cp * sec2b * dbeta_ddelta;
           tb / l2,       (vr / l2) * sec2b * dbeta_ddelta];

    x_r_i  = [X_r; Y_r; psi_r];
    mu_r_i = [vr; delta_r];

    f_r = [vr * (cp - sp * tb);
           vr * (sp + cp * tb);
           (vr / l2) * tb];

    phi_xT = f_r - A_T * x_r_i - B_T * mu_r_i;
    denom  = x_r_i' * x_r_i;
    A_GTZ = A_T + (phi_xT * x_r_i') / denom;

    sys_c_kin = ss(A_GTZ, B_T, eye(3), zeros(3,2));
    sys_d_kin = c2d(sys_c_kin, dt, 'zoh');
    
    Ad = sys_d_kin.A; 
    Bd = sys_d_kin.B;
end

function [V, Ob] = buildMarkovObsv_TV(A_seq, B_seq, C_seq, tau, m1, pp)
    n_st = size(A_seq, 1); 
    Ob = zeros(tau * pp, n_st);
    V = zeros(tau * pp, (tau-1) * m1);
    
    for i = 1:tau 
        Phi_i = eye(n_st);
        for m = tau : -1 : i+1
            Phi_i = A_seq(:,:,m) * Phi_i;
        end
        Ob((i-1)*pp+1 : i*pp, :) = C_seq(:,:,i) * Phi_i; 
        if i < tau
            for c = i : tau-1
                Phi_trans = eye(n_st);
                for m = c : -1 : i+1
                    Phi_trans = A_seq(:,:,m) * Phi_trans;
                end
                B_t = B_seq(:,:, c+1);
                V((i-1)*pp+1 : i*pp, (c-1)*m1+1 : c*m1) = C_seq(:,:,i) * Phi_trans * B_t;
            end
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