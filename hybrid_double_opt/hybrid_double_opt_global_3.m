% MOST UP-TO-DATE
% True LTV-MPC — 2-Input [vx, steer] Kinematic/Dynamic Switch (with kinematic mode updated)
% BOTH Models use ABSOLUTE GLOBAL POSE as States
% Augmented State formulation with GTZ Linearization
% now, the plant output y_out is the vehicle's absolute pose (X,Y \psi)
% rotated by the reference heading into path-aligned axis (aceeptable ---  what we want)
% Output formulation: y = C*x  (rotated ABSOLUTE pose, no tracking error in the
% state). The reference enters the COST as a setpoint Y_ref_horizon, built by
% accumulating the reference's path-frame increments with the model's per-step
% rotation (a one-shot C*x_ref would be biased under a time-varying path frame).

clc; clear; close all;

%% Vehicle & controller parameters
Cf = 30000;   Cr = 30000;
l1 = 1.4225;  l2 = 1.4225;
l  = l1 + l2;
m  = 1280;    Iz = 2500;

delta_t = 0.05;
Np      = 35;

% MPC Weights (Penalizing path-frame errors: e_x, e_psi, e_y)
Q_x_kin = 500;  Q_psi_kin = 1000;  Q_y_kin = 500;  
R_dv_kin = 1000; R_du_kin = 500;

Q_x_dyn = 500;  Q_psi_dyn = 1000;  Q_y_dyn = 2000;  
R_dv_dyn = 1000; R_du_dyn = 500;

% Rate Constraints (per step)
dv_max = 2.5 * delta_t;  % Max acceleration (m/s^2 * dt)
du_max = deg2rad(5);    % Max steering rate

% Absolute Constraints
v_min = 0.0;             
v_max = 25.0;            
u_max = deg2rad(25);   

v_switch_lo = 6.0;
v_switch_hi = 7.0;

vx_min_dyn = 0.5;
vx_min_kin = 0;   

%% Load reference trajectory
traj = readmatrix('C:\Users\soguchie\OneDrive - purdue.edu\ECE 699\Summer 2025\RoadRunner Projects\RoadRunner Project 1\Exports\BostonIntersection_design\Sedan_1.csv');
% traj = readmatrix('C:\Users\soguchie\OneDrive - purdue.edu\ECE 699\Summer 2025\RoadRunner Projects\RoadRunner Project 1\Exports\SingaporeIntersection_design\Sedan_1.csv');

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

beta_r_ref  = asin( min( abs(l2 .* psi_dot_ref ./ max(v_ref, vx_min_kin)), 1) .* sign(psi_dot_ref) );
delta_r_ref = atan( l * tan(beta_r_ref) / l2 );
delta_r_ref = max(min(delta_r_ref, u_max), -u_max);

%% Cost matrices 
Q_kin_bar = kron(eye(Np), diag([Q_x_kin, Q_psi_kin, Q_y_kin]));
R_kin_bar = kron(eye(Np), diag([R_dv_kin, R_du_kin]));

Q_dyn_bar = kron(eye(Np), diag([Q_x_dyn, Q_psi_dyn, Q_y_dyn]));
R_dyn_bar = kron(eye(Np), diag([R_dv_dyn, R_du_dyn]));

%% Initialize simulation
sim_steps = N_total - Np;

X_global   = X_ref(1);
Y_global   = Y_ref(1);
psi_global = psi_ref(1);

% ABSOLUTE STATES
% Kinematic = [X; Y; psi]
% Dynamic   = [v_y; psi; psi_dot; Y; X]
x_kin = [X_global; Y_global; psi_global];

beta_init  = atan2(l2 * tan(delta_r_ref(1)), l);
v_y_init   = v_ref(1) * sin(beta_init);
r_init     = v_ref(1) / l2 * sin(beta_init);
x_dyn      = [v_y_init; psi_global; r_init; Y_global; X_global]; 

x_kin_prev = x_kin;
x_dyn_prev = x_dyn;

if v_ref(1) < v_switch_lo
    active_model = 'kinematic';
else
    active_model = 'dynamic';
end

delta_U_guess = zeros(2 * Np, 1);
u_prev = [v_ref(1); delta_r_ref(1)];

% Logs
e_x_log    = zeros(sim_steps, 1);
e_psi_log  = zeros(sim_steps, 1);
e_y_log    = zeros(sim_steps, 1);
y_out_log  = zeros(sim_steps, 3);
% y_ref_log  = zeros(sim_steps, 3);
Y_log      = zeros(sim_steps, 1);
X_log      = zeros(sim_steps, 1);
psi_log    = zeros(sim_steps, 1);
u_log      = zeros(sim_steps, 2);
du_log     = zeros(sim_steps, 2);
v_log      = zeros(sim_steps, 1);
model_log  = strings(sim_steps, 1);

fprintf('Starting 2-Input LTV-MPC (%d steps)...\n', sim_steps);

%% Main simulation loop
for k = 1 : sim_steps

    vx_now = u_prev(1);

    % Model switching (absolute pose transfer)
    if strcmp(active_model, 'kinematic') && vx_now >= v_switch_hi
        active_model = 'dynamic';
        beta_prev  = atan2(l2 * tan(u_prev(2)), l);
        r_init     = max(vx_now, vx_min_dyn) / l2 * sin(beta_prev);  
        v_y_init   = max(vx_now, vx_min_dyn) * sin(beta_prev);       
        
        x_dyn      = [v_y_init; x_kin(3); r_init; x_kin(2); x_kin(1)];
        x_dyn_prev = x_dyn;

    elseif strcmp(active_model, 'dynamic') && vx_now <= v_switch_lo
        active_model = 'kinematic';
        x_kin      = [x_dyn(5); x_dyn(4); x_dyn(2)];
        x_kin_prev = x_kin;
    end

    % Path-frame setpoint at the CURRENT step:  y_ref = C * x_ref
    cp0 = cos(psi_ref(k));  sp0 = sin(psi_ref(k));
    y_ref_now = [ cp0*X_ref(k) + sp0*Y_ref(k);
                  psi_ref(k);
                 -sp0*X_ref(k) + cp0*Y_ref(k) ];

    % Extract outputs for logs and MPC state.
    % MPC output is now the ROTATED ABSOLUTE POSE:  y = C * x  (no error in the state).
    if strcmp(active_model, 'kinematic')
        C_mat = [ cos(psi_ref(k)), sin(psi_ref(k)), 0;
                  0,               0,               1;
                 -sin(psi_ref(k)), cos(psi_ref(k)), 0];

        y_out = C_mat * x_kin;          % y = C * x
        dx    = x_kin - x_kin_prev;
        x_a   = [dx; y_out];
        Q_bar = Q_kin_bar;
        R_bar = R_kin_bar;
        n_st  = 3;
    else
        cp = cos(psi_ref(k));  sp = sin(psi_ref(k));
        C_mat = [ 0,  0,  0,  sp,  cp;
                  0,  1,  0,  0,    0;
                  0,  0,  0,  cp, -sp ];

        y_out = C_mat * x_dyn;          % y = C * x
        dx    = x_dyn - x_dyn_prev;
        x_a   = [dx; y_out];
        Q_bar = Q_dyn_bar;
        R_bar = R_dyn_bar;
        n_st  = 5;
    end

    % Tracking error (path frame) -- for logging only:  e = y - y_ref
    e_x_log(k)   = y_out(1) - y_ref_now(1);
    e_psi_log(k) = angdiff(y_ref_now(2), y_out(2));
    e_y_log(k)   = y_out(3) - y_ref_now(3);
    y_out_log(k, :) = y_out';
    % y_ref_log(k, :) = y_ref_now';
    Y_log(k)     = Y_global;
    X_log(k)     = X_global;
    psi_log(k)   = psi_global;
    v_log(k)     = vx_now;
    model_log(k) = active_model;

    %% Build Time-Varying Sequences for the Horizon
    Phi_a_seq   = cell(Np, 1);
    Gamma_a_seq = cell(Np, 1);
    y_ref_seq   = cell(Np, 1);     % path-frame reference setpoint per predicted step
    y_ref_acc   = y_ref_now;       % accumulates C*x_ref using the model's per-step rotation

    p = size(C_mat, 1);
    C_a = [zeros(p, n_st), eye(p)];

    for i = 1:Np
        idx = k + i - 1;
        v_r = max(v_ref(idx), vx_min_kin);
        psi_r = psi_ref(idx);
        delta_r = delta_r_ref(idx);

        % Reference setpoint (path frame), accumulated with the SAME per-step
        % rotation the prediction model uses (C evaluated at psi_ref(idx)).
        % A one-shot C*x_ref(k+i) would be biased here because the rotation is
        % time-varying along the horizon -- this accumulation is the "reference
        % shift correction", now carried by the setpoint instead of the state.
        idx_next = min(idx + 1, N_total);
        cpa = cos(psi_r);  spa = sin(psi_r);                 % psi_r = psi_ref(idx)

        dX_ref   = X_ref(idx_next) - X_ref(idx);
        dY_ref   = Y_ref(idx_next) - Y_ref(idx);
        dpsi_ref = angdiff(psi_ref(idx), psi_ref(idx_next));

        y_ref_acc = y_ref_acc + [ cpa*dX_ref + spa*dY_ref;
                                  dpsi_ref;
                                 -spa*dX_ref + cpa*dY_ref ];
        y_ref_seq{i} = y_ref_acc;
        
        if strcmp(active_model, 'kinematic')
            beta_r = atan2(l2 * tan(delta_r), l);
            dbeta_ddelta = l2 * l / (l^2 * cos(delta_r)^2 + l2^2 * sin(delta_r)^2);
            
            cp = cos(psi_r);   sp = sin(psi_r);
            tb = tan(beta_r);  sec2b = sec(beta_r)^2;
            
            A_T = [0, 0, -v_r * (sp + cp * tb);
                   0, 0,  v_r * (cp - sp * tb);
                   0, 0,  0];
                   
            B_T = [cp - sp * tb,  -v_r * sp * sec2b * dbeta_ddelta;
                   sp + cp * tb,   v_r * cp * sec2b * dbeta_ddelta;
                   tb / l2,       (v_r / l2) * sec2b * dbeta_ddelta];

            x_r_i = [X_ref(idx); Y_ref(idx); psi_r];
            mu_r_i  = [v_r; delta_r];

            f_r = [v_r * (cp - sp * tb);
                   v_r * (sp + cp * tb);
                   (v_r / l2) * tb];

            phi_xT = f_r - A_T * x_r_i - B_T * mu_r_i;
            A_GTZ  = A_T + (phi_xT * x_r_i') / (x_r_i' * x_r_i);

            sys_c_kin = ss(A_GTZ, B_T, eye(3), zeros(3,2));
            sys_d_kin = c2d(sys_c_kin, delta_t, 'zoh');

            Phi = sys_d_kin.A;
            Gamma = sys_d_kin.B;
            % using expm() for discretization
            % M_kin = expm([A_GTZ, B_T; zeros(2, 3), zeros(2, 2)] * delta_t);
            % Phi   = M_kin(1:3, 1:3);
            % Gamma = M_kin(1:3, 4:5);
            
            C_mat_i = [ cos(psi_r), sin(psi_r), 0;
                        0,          0,          1;
                       -sin(psi_r), cos(psi_r), 0];
        else
            v_r_dyn = max(v_r, vx_min_dyn);
            beta_r  = atan2(l2 * tan(delta_r), l);
            v_y_r   = v_r_dyn * sin(beta_r);
            psi_dot_r = psi_dot_ref(idx);

            x_r_i  = [v_y_r; psi_r; psi_dot_r; Y_ref(idx); X_ref(idx)];
            mu_r_i = [v_r_dyn; delta_r];

            cp = cos(psi_r);  sp = sin(psi_r);

            A_c = zeros(5, 5);
            A_c(1,1) = -(2*Cf+2*Cr)/(m*v_r_dyn);
            A_c(1,3) = -(2*Cf*l1-2*Cr*l2)/(m*v_r_dyn) - v_r_dyn;
            A_c(2,3) = 1;
            A_c(3,1) = -(2*l1*Cf-2*l2*Cr)/(Iz*v_r_dyn);
            A_c(3,3) = -(2*l1^2*Cf+2*l2^2*Cr)/(Iz*v_r_dyn);
            A_c(4,1) = cp;
            A_c(4,2) = v_r_dyn*cp - v_y_r*sp;   
            A_c(5,1) = -sp;
            A_c(5,2) = -v_r_dyn*sp - v_y_r*cp;  

            df1_dvx = -(2*Cf*l1-2*Cr*l2)/(m*v_r_dyn^2)*(-1)*psi_dot_r - psi_dot_r + (2*Cf+2*Cr)/(m*v_r_dyn^2)*v_y_r;                    
            df3_dvx = (2*l1*Cf-2*l2*Cr)/(Iz*v_r_dyn^2)*v_y_r + (2*l1^2*Cf+2*l2^2*Cr)/(Iz*v_r_dyn^2)*psi_dot_r;
            df4_dvx = sp;   
            df5_dvx = cp;   
            B_v = [df1_dvx; 0; df3_dvx; df4_dvx; df5_dvx];

            B_steer = [2*Cf/m; 0; 2*l1*Cf/Iz; 0; 0];
            B_c = [B_v, B_steer];

            f_r = [ -(2*Cf+2*Cr)/(m*v_r_dyn)*v_y_r - ((2*Cf*l1-2*Cr*l2)/(m*v_r_dyn) + v_r_dyn)*psi_dot_r + (2*Cf/m)*delta_r;                                   
                    psi_dot_r;                                              
                   -(2*l1*Cf-2*l2*Cr)/(Iz*v_r_dyn)*v_y_r - (2*l1^2*Cf+2*l2^2*Cr)/(Iz*v_r_dyn)*psi_dot_r + (2*l1*Cf/Iz)*delta_r;                               
                    v_r_dyn*sp + v_y_r*cp;                                  
                    v_r_dyn*cp - v_y_r*sp ];                                

            phi_xT = f_r - A_c * x_r_i - B_c * mu_r_i;
            denom  = x_r_i' * x_r_i;     
            A_GTZ = A_c + (phi_xT * x_r_i') / denom;

            sys_c_dyn = ss(A_GTZ, B_c, eye(5), zeros(5,2));
            sys_d_dyn = c2d(sys_c_dyn, delta_t, 'zoh');
            Phi = sys_d_dyn.A;
            Gamma = sys_d_dyn.B;
            
            % using expm() for discretization
            % M_dyn = expm([A_GTZ, B_c; zeros(2, 5), zeros(2, 2)] * delta_t);
            % Phi   = M_dyn(1:5, 1:5);
            % Gamma = M_dyn(1:5, 6:7);

            C_mat_i = [ 0,  0,  0,  sp,  cp;
                        0,  1,  0,  0,    0;
                        0,  0,  0,  cp, -sp ];
        end
        
        Phi_a_seq{i}   = [Phi, zeros(n_st, p); 
                          C_mat_i*Phi, eye(p)];
        Gamma_a_seq{i} = [Gamma; 
                          C_mat_i*Gamma];
    end
    
    %% Build LTV Prediction Matrices (W, Z) and reference setpoint (Y_ref_horizon)
    n_a = n_st + p;
    W = zeros(p * Np, n_a);
    Z = zeros(p * Np, 2 * Np);
    Y_ref_horizon = zeros(p * Np, 1);

    Phi_prod = eye(n_a);
    for i = 1:Np
        Phi_prod = Phi_a_seq{i} * Phi_prod;
        W((i-1)*p+1 : i*p, :) = C_a * Phi_prod;
        for j = 1:i
            if i == j
                temp_prod = eye(n_a);
            else
                temp_prod = eye(n_a);
                for n_idx = i:-1:j+1
                    temp_prod = temp_prod * Phi_a_seq{n_idx};
                end
            end
            Z((i-1)*p+1 : i*p, (j-1)*2+1 : j*2) = C_a * temp_prod * Gamma_a_seq{j};
        end

        % Stack the path-frame setpoint for predicted step k+i
        Y_ref_horizon((i-1)*p+1 : i*p) = y_ref_seq{i};
    end
    
    %% Define Constraints
    lb_du = repmat([-dv_max; -du_max], Np, 1);
    ub_du = repmat([ dv_max;  du_max], Np, 1);
    
    E_mat = kron(ones(Np, 1), eye(2));
    H_mat = kron(tril(ones(Np)), eye(2));
    
    U_max_vec = repmat([v_max; u_max], Np, 1);
    U_min_vec = repmat([v_min; -u_max], Np, 1);
    
    A_ineq = [H_mat; -H_mat];
    b_ineq = [U_max_vec - E_mat * u_prev; 
             -U_min_vec + E_mat * u_prev];

    %% Solve Optimization Problem using fmincon
    % r_p = Y_ref_horizon;
    % Q = Q_bar;
    % R = R_bar;
    % W_xa = W * x_a;
    % 
    % cost_func = @(DU) obj_with_gradient(DU, r_p, W_xa, Z, Q, R);
    % options_fmincon = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', 'SpecifyObjectiveGradient', true);
    % 
    % [delta_U_opt, ~, exitflag] = fmincon(cost_func, delta_U_guess, A_ineq, b_ineq, ...
    %                                       [], [], lb_du, ub_du, [], options_fmincon);

    H_qp = Z' * Q_bar * Z + R_bar;
    f_qp = Z' * Q_bar * (W * x_a - Y_ref_horizon);

    H_qp = (H_qp + H_qp') / 2;

    options_qp = optimoptions('quadprog', 'Display', 'off');

    [delta_U_opt, ~, exitflag] = quadprog(H_qp, f_qp, A_ineq, b_ineq, ...
                                          [], [], lb_du, ub_du, delta_U_guess, options_qp);
    
    if exitflag <= 0 
        warning('Optimization failed at step %d. Coasting.', k);
        du_now = [0; 0]; 
    else
        du_now = delta_U_opt(1:2);
    end

    u_current = u_prev + du_now;

    %% Propagate Absolute Plant
    x_kin_prev = x_kin;
    x_dyn_prev = x_dyn;
    vx_act     = u_current(1);
    delta_act  = u_current(2);
    
    if strcmp(active_model, 'kinematic')
        vx_kin   = max(vx_act, vx_min_kin);
        beta_now = atan2(l2 * tan(delta_act), l);
        vc_now   = vx_kin / cos(beta_now);  % Convert vx control input to CG magnitude
        
        ode_kin  = @(~, s) [vc_now * cos(s(3) + beta_now);
                            vc_now * sin(s(3) + beta_now);
                            (vc_now / l2) * sin(beta_now)];
                            
        [~, s] = ode23s(ode_kin, [0, delta_t], x_kin);

        x_kin(1) = s(end,1);
        x_kin(2) = s(end,2);
        x_kin(3) = s(end,3);
        
        X_global   = x_kin(1);
        Y_global   = x_kin(2);
        psi_global = x_kin(3);

    else
        vx_pl = max(vx_act, vx_min_dyn);

        ode_dyn = @(~, xs) [ ...
            -(2*Cf+2*Cr)/(m*vx_pl)*xs(1) - ((2*Cf*l1-2*Cr*l2)/(m*vx_pl) + vx_pl)*xs(3) + (2*Cf/m)*delta_act;                                     
             xs(3);                                                     
            -(2*l1*Cf-2*l2*Cr)/(Iz*vx_pl)*xs(1) - (2*l1^2*Cf+2*l2^2*Cr)/(Iz*vx_pl)*xs(3) + (2*l1*Cf/Iz)*delta_act;                                 
             vx_pl*sin(xs(2)) + xs(1)*cos(xs(2));                       
             vx_pl*cos(xs(2)) - xs(1)*sin(xs(2)) ];                     

        [~, s]  = ode23s(ode_dyn, [0, delta_t], x_dyn);
        x_dyn   = s(end,:)';

        psi_global = x_dyn(2);
        Y_global   = x_dyn(4);
        X_global   = x_dyn(5);
    end

    u_log(k, :)  = u_current';
    du_log(k, :) = du_now';
    u_prev       = u_current;

    delta_U_guess = [delta_U_opt(3:end); 0; 0];
end

fprintf('Simulation Complete.\n');

%% Plots
t_plot  = t_uniform(1:sim_steps);
kin_idx = model_log == "kinematic";
dyn_idx = model_log == "dynamic";

figure(1);
subplot(3,1,1);
% plot(t_plot, v_ref(1:sim_steps), 'k--', 'LineWidth', 1.5); hold on;
scatter(t_plot(kin_idx), u_log(kin_idx, 1), 10, 'b', 'filled'); hold on;
scatter(t_plot(dyn_idx), u_log(dyn_idx, 1), 10, 'r', 'filled');
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2; box on;
grid on;
legend('Kinematic', 'Dynamic', 'FontName','Times New Roman' ,'FontSize',20, 'Location','best');
ylabel('$v$ (m/s)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);

subplot(3,1,2);
plot(t_plot, rad2deg(u_log(:,2)), 'k-', 'LineWidth', 2);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
yline( rad2deg(u_max),  'm--', 'LineWidth',2); yline(-rad2deg(u_max),  'm--', 'LineWidth',2);
grid on; ylabel('$\delta_f$(deg)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);  
ylim([-35,35]);

subplot(3,1,3);
plot(t_plot, rad2deg(du_log(:, 2)), 'k-', 'LineWidth', 2);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
hold on; grid on;
yline( rad2deg(du_max),  'm--', 'LineWidth', 2); 
yline(-rad2deg(du_max),  'm--', 'LineWidth', 2);
ylabel('$\Delta\delta_f$ (deg/step)', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 28);
xlabel('Time (s)', 'Interpreter', 'latex', 'FontName', 'Times New Roman', 'FontSize', 28);
ylim([-rad2deg(du_max)*1.2, rad2deg(du_max)*1.2]);

figure(2);
subplot(3,1,1);
plot(t_plot, e_x_log, 'k-', 'LineWidth', 2);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; 
ylabel('$e_x$ (m)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28); 

subplot(3,1,2);
plot(t_plot, e_y_log, 'b-', 'LineWidth', 2);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; 
ylabel('$e_y$ (m)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);

subplot(3,1,3);
plot(t_plot, e_psi_log, 'r-', 'LineWidth', 2);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; 
ylabel('$e_\psi$ (rad)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28); 
xlabel('Time (s)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);

figure(3);
subplot(3,1,1);
% plot(t_plot, y_ref_log(:,1), 'k--', 'LineWidth', 2); hold on;
plot(t_plot, y_out_log(:,1), 'b-', 'LineWidth', 1.5);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; 
% legend('Reference', 'Plant Output', 'FontName','Times New Roman' ,'FontSize',20, 'Location','best');
ylabel('Path $X$ (m)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28); 

subplot(3,1,2);
% plot(t_plot, y_ref_log(:,3), 'k--', 'LineWidth', 2); hold on;
plot(t_plot, y_out_log(:,3), 'b-', 'LineWidth', 1.5);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; 
ylabel('Path $Y$ (m)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);

subplot(3,1,3);
% plot(t_plot, rad2deg(y_ref_log(:,2)), 'k--', 'LineWidth', 2); hold on;
plot(t_plot, rad2deg(y_out_log(:,2)), 'r-', 'LineWidth', 1.5);
ax = gca; ax.XAxis.FontSize = 20; ax.YAxis.FontSize = 20; ax.FontSize = 20; ax.LineWidth = 2;
grid on; 
ylabel('Global $\psi$ (deg)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28); 
xlabel('Time (s)', 'Interpreter','latex', 'FontName','Times New Roman' ,'FontSize',28);


function [J, grad] = obj_with_gradient(DU, r_p, W_xa, Z, Q, R)
    Y_pred = W_xa + Z * DU;
    
    J = 0.5 * (r_p - Y_pred)' * Q * (r_p - Y_pred) + 0.5 * DU' * R * DU;
    if nargout > 1
        grad = (-(r_p - W_xa - Z*DU)' * Q * Z + DU' * R)';
    end
end