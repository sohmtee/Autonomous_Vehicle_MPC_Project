clc; clear; close all;

%% Vehicle & controller parameters
Cf = 30000;   Cr = 30000;
l1 = 1.4225;  l2 = 1.4225;
l  = l1 + l2;
m  = 1280;    Iz = 2500;

delta_t = 0.05;
Np      = 35;

% MPC weights
Q_x_kin = 500;  Q_psi_kin = 1000;  Q_y_kin = 500;
R_dv_kin = 1000; R_du_kin = 500;

% Rate constraints (per step)
dv_max = 3 * delta_t;   % max acceleration (m/s^2 * dt)
du_max = deg2rad(10);      % max steering rate

% Absolute constraints
v_min = 0.0;
v_max = 25.0;
u_max = deg2rad(25);

vx_min_kin = 0;

traj = readmatrix("C:\Users\soguchie\OneDrive - purdue.edu\ECE 699\Summer 2025\RoadRunner Projects\RoadRunner Project 1\Exports\BostonIntersection_design2\Sedan_1.csv");

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
    pp_sp       = pchip(t_uniform, psi_ref);
    psi_dot_ref = ppval(fnder(pp_sp, 1), t_uniform);
catch
    psi_dot_ref = smooth(gradient(psi_ref, delta_t), 5);
end

beta_r_ref  = asin( min( abs(l2 .* psi_dot_ref ./ max(v_ref, vx_min_kin)), 1) .* sign(psi_dot_ref) );
delta_r_ref = atan( l * tan(beta_r_ref) / l2 );
delta_r_ref = max(min(delta_r_ref, u_max), -u_max);

%% Cost matrices
Q_bar = kron(eye(Np), diag([Q_x_kin, Q_psi_kin, Q_y_kin]));
R_bar = kron(eye(Np), diag([R_dv_kin, R_du_kin]));

%% Initialize simulation
sim_steps = N_total - Np;

X_global   = X_ref(1);
Y_global   = Y_ref(1);
psi_global = psi_ref(1);

% Controller pose state: [X; Y; psi]
x_kin      = [X_global; Y_global; psi_global];
x_kin_prev = x_kin;

beta_init = atan2(l2 * tan(delta_r_ref(1)), l);
v_y_init  = v_ref(1) * sin(beta_init);
r_init    = v_ref(1) / l2 * sin(beta_init);
x_dyn     = [v_y_init; psi_global; r_init; Y_global; X_global];

delta_U_guess = zeros(2 * Np, 1);
u_prev = [v_ref(1); delta_r_ref(1)];

n_st = 3; 
p    = 3;                    % output dimension
C_a  = [zeros(p, n_st), eye(p)];

% Logs
e_x_log     = zeros(sim_steps, 1);
e_psi_log   = zeros(sim_steps, 1);
e_y_log     = zeros(sim_steps, 1);
y_out_log   = zeros(sim_steps, 3);
X_log       = zeros(sim_steps, 1);
Y_log       = zeros(sim_steps, 1);
psi_log     = zeros(sim_steps, 1);
u_log       = zeros(sim_steps, 2);
du_log      = zeros(sim_steps, 2);
v_log       = zeros(sim_steps, 1);

fprintf('Kinematic-only LTV-MPC | %d steps...\n', sim_steps);

%% Main simulation loop
for k = 1 : sim_steps

    vx_now = u_prev(1);

    % Path-frame setpoint at the CURRENT step:  y_ref = C * x_ref
    cp0 = cos(psi_ref(k));  sp0 = sin(psi_ref(k));
    y_ref_now = [ cp0*X_ref(k) + sp0*Y_ref(k);
                  psi_ref(k);
                 -sp0*X_ref(k) + cp0*Y_ref(k) ];

    C_mat = [ cp0,  sp0, 0;
              0,    0,   1;
             -sp0,  cp0, 0];

    y_out = C_mat * x_kin;          % y = C * x
    dx    = x_kin - x_kin_prev;
    x_a   = [dx; y_out];

    % Tracking error (path frame)
    e_x_log(k)      = y_out(1) - y_ref_now(1);
    e_psi_log(k)    = angdiff(y_ref_now(2), y_out(2));
    e_y_log(k)      = y_out(3) - y_ref_now(3);
    y_out_log(k, :) = y_out';
    X_log(k)        = X_global;
    Y_log(k)        = Y_global;
    psi_log(k)      = psi_global;
    v_log(k)        = vx_now;

    %% Build time-varying sequences over the horizon (Kinematic GTZ only)
    Phi_a_seq   = cell(Np, 1);
    Gamma_a_seq = cell(Np, 1);
    y_ref_seq   = cell(Np, 1);
    y_ref_acc   = y_ref_now;

    for i = 1:Np
        idx     = k + i - 1;
        v_r     = max(v_ref(idx), vx_min_kin);
        psi_r   = psi_ref(idx);
        delta_r = delta_r_ref(idx);

        % Accumulate the reference setpoint in the path frame, using the same
        % per-step rotation the prediction model uses.
        idx_next = min(idx + 1, N_total);
        cpa = cos(psi_r);  spa = sin(psi_r);

        dX_ref   = X_ref(idx_next) - X_ref(idx);
        dY_ref   = Y_ref(idx_next) - Y_ref(idx);
        dpsi_ref = angdiff(psi_ref(idx), psi_ref(idx_next));

        y_ref_acc = y_ref_acc + [ cpa*dX_ref + spa*dY_ref;
                                  dpsi_ref;
                                 -spa*dX_ref + cpa*dY_ref ];
        y_ref_seq{i} = y_ref_acc;

        % Kinematic GTZ linearization
        beta_r       = atan2(l2 * tan(delta_r), l);
        dbeta_ddelta = l2 * l / (l^2 * cos(delta_r)^2 + l2^2 * sin(delta_r)^2);

        cp = cos(psi_r);   sp = sin(psi_r);
        tb = tan(beta_r);  sec2b = sec(beta_r)^2;

        A_T = [0, 0, -v_r * (sp + cp * tb);
               0, 0,  v_r * (cp - sp * tb);
               0, 0,  0];

        B_T = [cp - sp * tb,  -v_r * sp * sec2b * dbeta_ddelta;
               sp + cp * tb,   v_r * cp * sec2b * dbeta_ddelta;
               tb / l2,       (v_r / l2) * sec2b * dbeta_ddelta];

        x_r_i  = [X_ref(idx); Y_ref(idx); psi_r];
        mu_r_i = [v_r; delta_r];

        f_r = [v_r * (cp - sp * tb);
               v_r * (sp + cp * tb);
               (v_r / l2) * tb];

        phi_xT = f_r - A_T * x_r_i - B_T * mu_r_i;
        A_GTZ  = A_T + (phi_xT * x_r_i') / (x_r_i' * x_r_i);

        sys_c_kin = ss(A_GTZ, B_T, eye(3), zeros(3,2));
        sys_d_kin = c2d(sys_c_kin, delta_t, 'zoh');

        Phi   = sys_d_kin.A;
        Gamma = sys_d_kin.B;

        C_mat_i = [ cp,  sp, 0;
                    0,   0,  1;
                   -sp,  cp, 0];

        Phi_a_seq{i}   = [Phi, zeros(n_st, p);
                          C_mat_i*Phi, eye(p)];
        Gamma_a_seq{i} = [Gamma;
                          C_mat_i*Gamma];
    end

    %% Build LTV prediction matrices (W, Z) and reference setpoint
    n_a = n_st + p;
    W = zeros(p * Np, n_a);
    Z = zeros(p * Np, 2 * Np);
    Y_ref_horizon = zeros(p * Np, 1);

    Phi_prod = eye(n_a);
    for i = 1:Np
        Phi_prod = Phi_a_seq{i} * Phi_prod;
        W((i-1)*p+1 : i*p, :) = C_a * Phi_prod;
        for j = 1:i
            temp_prod = eye(n_a);
            if i ~= j
                for n_idx = i:-1:j+1
                    temp_prod = temp_prod * Phi_a_seq{n_idx};
                end
            end
            Z((i-1)*p+1 : i*p, (j-1)*2+1 : j*2) = C_a * temp_prod * Gamma_a_seq{j};
        end
        Y_ref_horizon((i-1)*p+1 : i*p) = y_ref_seq{i};
    end

    %% Constraints
    lb_du = repmat([-dv_max; -du_max], Np, 1);
    ub_du = repmat([ dv_max;  du_max], Np, 1);

    E_mat = kron(ones(Np, 1), eye(2));
    H_mat = kron(tril(ones(Np)), eye(2));

    U_max_vec = repmat([v_max;  u_max], Np, 1);
    U_min_vec = repmat([v_min; -u_max], Np, 1);

    A_ineq = [H_mat; -H_mat];
    b_ineq = [U_max_vec - E_mat * u_prev;
             -U_min_vec + E_mat * u_prev];

    %% Solve QP
    H_qp = Z' * Q_bar * Z + R_bar;
    f_qp = Z' * Q_bar * (W * x_a - Y_ref_horizon);
    H_qp = (H_qp + H_qp') / 2;

    options_qp = optimoptions('quadprog', 'Display', 'off');

    [delta_U_opt, ~, exitflag] = quadprog(H_qp, f_qp, A_ineq, b_ineq, ...
                                          [], [], lb_du, ub_du, delta_U_guess, options_qp);

    if exitflag <= 0
        warning('Optimization failed at step %d. Coasting.', k);
        du_now = [0; 0];
        delta_U_opt = zeros(2*Np, 1);
    else
        du_now = delta_U_opt(1:2);
    end

    u_current = u_prev + du_now;

    %% Propagate plant
    x_kin_prev = x_kin;
    vx_act     = u_current(1);
    delta_act  = u_current(2);

    beta_cmd = atan2(l2 * tan(delta_act), l);

    vx_pl = max(vx_act, eps);

    ode_dyn = @(~, xs) [-(2*Cf+2*Cr)/(m*vx_pl)*xs(1) - ((2*Cf*l1-2*Cr*l2)/(m*vx_pl) + vx_pl)*xs(3) + (2*Cf/m)*delta_act;
         xs(3);
        -(2*l1*Cf-2*l2*Cr)/(Iz*vx_pl)*xs(1) - (2*l1^2*Cf+2*l2^2*Cr)/(Iz*vx_pl)*xs(3) + (2*l1*Cf/Iz)*delta_act;
         vx_pl*sin(xs(2)) + xs(1)*cos(xs(2));
         vx_pl*cos(xs(2)) - xs(1)*sin(xs(2)) ];

    [~, s] = ode23s(ode_dyn, [0, delta_t], x_dyn);
    x_dyn  = s(end, :)';

    psi_global = x_dyn(2);
    Y_global   = x_dyn(4);
    X_global   = x_dyn(5);

    % Feed the dynamic plant's pose back to the kinematic controller
    x_kin = [X_global; Y_global; psi_global];

    u_log(k, :)  = u_current';
    du_log(k, :) = du_now';
    u_prev       = u_current;

    delta_U_guess = [delta_U_opt(3:end); 0; 0];

end

fprintf('Simulation complete.\n\n');

t_plot = t_uniform(1:sim_steps);

%% Plots
figure(1);
subplot(3,1,1);
plot(t_plot, v_ref(1:sim_steps), 'k--', 'LineWidth', 2); hold on;
plot(t_plot, u_log(:,1), 'b-', 'LineWidth', 2);
ax = gca; ax.FontSize = 20; ax.LineWidth = 2; box on; grid on;
legend('Reference', 'Kinematic', 'FontName','Times New Roman', 'FontSize',20, 'Location','best');
ylabel('$v$ (m/s)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);

subplot(3,1,2);
plot(t_plot, rad2deg(u_log(:,2)), 'k-', 'LineWidth', 2); hold on;
yline( rad2deg(u_max), 'm--', 'LineWidth', 2); yline(-rad2deg(u_max), 'm--', 'LineWidth', 2);
ax = gca; ax.FontSize = 20; ax.LineWidth = 2; grid on;
ylabel('$\delta_f$ (deg)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);
ylim([-35, 35]);

subplot(3,1,3);
plot(t_plot, rad2deg(du_log(:,2)), 'k-', 'LineWidth', 2); hold on;
yline( rad2deg(du_max), 'm--', 'LineWidth', 2); yline(-rad2deg(du_max), 'm--', 'LineWidth', 2);
ax = gca; ax.FontSize = 20; ax.LineWidth = 2; grid on;
ylabel('$\Delta\delta_f$ (deg/step)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);
xlabel('Time (s)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);
ylim([-rad2deg(du_max)*1.2, rad2deg(du_max)*1.2]);

figure(2);
subplot(3,1,1);
plot(t_plot, e_x_log, 'k-', 'LineWidth', 2);
ax = gca; ax.FontSize = 20; ax.LineWidth = 2; grid on;
ylabel('$e_x$ (m)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);

subplot(3,1,2);
plot(t_plot, e_y_log, 'b-', 'LineWidth', 2);
ax = gca; ax.FontSize = 20; ax.LineWidth = 2; grid on;
ylabel('$e_y$ (m)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);

subplot(3,1,3);
plot(t_plot, e_psi_log, 'r-', 'LineWidth', 2);
ax = gca; ax.FontSize = 20; ax.LineWidth = 2; grid on;
ylabel('$e_\psi$ (rad)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);
xlabel('Time (s)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);

figure(3);
subplot(3,1,1);
plot(t_plot, X_ref(1:sim_steps), 'k--', 'LineWidth', 2); hold on;
plot(t_plot, X_log, 'b-', 'LineWidth', 2);
ax = gca; ax.FontSize = 20; ax.LineWidth = 2; grid on;
ylabel('$x_g$ (m)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);
legend('Reference', 'Plant', 'FontName','Times New Roman', 'FontSize',18, 'Location','best');

subplot(3,1,2);
plot(t_plot, psi_ref(1:sim_steps), 'k--', 'LineWidth', 2); hold on;
plot(t_plot, psi_log, 'b-', 'LineWidth', 2);
ax = gca; ax.FontSize = 20; ax.LineWidth = 2; grid on;
ylabel('$\psi_g$ (rad)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);

subplot(3,1,3);
plot(t_plot, Y_ref(1:sim_steps), 'k--', 'LineWidth', 2); hold on;
plot(t_plot, Y_log, 'b-', 'LineWidth', 2);
ax = gca; ax.FontSize = 20; ax.LineWidth = 2; grid on;
ylabel('$y_g$ (m)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);
xlabel('Time (s)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);

figure(4);
plot(X_ref(1:sim_steps), Y_ref(1:sim_steps), 'k--', 'LineWidth', 2); hold on;
plot(X_log, Y_log, 'b-', 'LineWidth', 2);
axis equal; grid on;
ax = gca; ax.FontSize = 20; ax.LineWidth = 2;
xlabel('$X_g$ (m)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);
ylabel('$Y_g$ (m)', 'Interpreter','latex', 'FontName','Times New Roman', 'FontSize',28);
legend('Reference', 'Kinematic-only MPC', 'FontName','Times New Roman', 'FontSize',18, 'Location','best');