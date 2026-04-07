%  MPC — GTZ Kinematic / Dynamic Bicycle Model Switch

clc; clear; close all;

%% Vehicle & controller parameters
Cf = 30000;   Cr = 30000;
l1 = 1.2;     l2 = 1.22;
l  = l1 + l2;
m  = 1280;    Iz = 2500;

delta_t = 0.05;
Np      = 35;

Q_psi_kin = 1000;  Q_y_kin = 400;  R_du_kin = 500;
Q_psi_dyn = 1000;  Q_y_dyn = 400;  R_du_dyn = 2000;

du_max = deg2rad(20);
u_max  = deg2rad(30);

v_switch_lo = 4.0;
v_switch_hi = 5.0;

vx_min_dyn = 0.5;
vx_min_kin = 0;  

%% Load reference trajectory
traj = readmatrix('C:\Users\soguchie\OneDrive - purdue.edu\ECE 699\Summer 2025\RoadRunner Projects\RoadRunner Project 1\Exports\SingaporeIntersection_design_2026-03-24_small\Sedan_1_newest2.csv');

time     = traj(:,1);
X_path   = traj(:,2);
Y_path   = traj(:,3);
psi_path = traj(:,5);

% Yaw convention check
n_chk   = min(10, length(time)-1);
psi_num = atan2(mean(diff(Y_path(1:n_chk+1))), mean(diff(X_path(1:n_chk+1))));
if abs(angdiff(psi_num, psi_path(1))) > deg2rad(5)
    warning('Applying +pi/2 RoadRunner yaw offset.');
    psi_path = psi_path + pi/2;
end

psi_unwrapped = unwrap(psi_path);
t_uniform     = (0 : delta_t : time(end))';

X_ref   = interp1(time, X_path,        t_uniform, 'pchip');
Y_ref   = interp1(time, Y_path,        t_uniform, 'pchip');
psi_ref = interp1(time, psi_unwrapped, t_uniform, 'pchip');

dX_raw = diff(X_path) ./ diff(time);
dY_raw = diff(Y_path) ./ diff(time);
v_geom = sqrt(dX_raw.^2 + dY_raw.^2);
t_mid  = 0.5*(time(1:end-1) + time(2:end));
v_ref  = interp1(t_mid, v_geom, t_uniform, 'pchip', 'extrap');
v_ref  = smooth(v_ref, 5);

N_total = length(t_uniform);

try
    pp          = pchip(t_uniform, psi_ref);
    psi_dot_ref = ppval(fnder(pp, 1), t_uniform);
    fprintf('Using spline derivative for psi_dot_ref.\n');
catch
    psi_dot_ref = smooth(gradient(psi_ref, delta_t), 5);
end

% Reference steering feedforward
delta_r_ref = atan2(l * psi_dot_ref, max(v_ref, vx_min_kin));
delta_r_ref = smooth(delta_r_ref, 3);
delta_r_ref = max(min(delta_r_ref, u_max), -u_max);

%% Cost matrices & constraints
Q_kin = kron(eye(Np), diag([Q_psi_kin, Q_y_kin]));
R_kin = kron(eye(Np), R_du_kin);

Q_dyn = kron(eye(Np), diag([Q_psi_dyn, Q_y_dyn]));
R_dyn = kron(eye(Np), R_du_dyn);

lb_du = -du_max * ones(Np, 1);
ub_du =  du_max * ones(Np, 1);
A_cs  = tril(ones(Np));

options = optimoptions('fmincon', ...
    'Algorithm',     'sqp', ...
    'Display',       'off', ...
    'MaxIterations', 100);

% Output matrices
Cd_kin = [0 0 1;
          0 1 0];

Cd_dyn = [0 1 0 0; 
          0 0 0 1];

%% Initialize simulation
sim_steps = N_total - Np;

e_kin = zeros(3, 1);
x_dyn = zeros(4, 1);

X_global   = X_ref(1);
Y_global   = Y_ref(1);
psi_global = psi_ref(1);

if v_ref(1) < v_switch_lo
    active_model = 'kinematic';
else
    active_model = 'dynamic';
end

u_prev        = delta_r_ref(1);
delta_U_guess = zeros(Np, 1);

e_psi_log  = zeros(sim_steps, 1);
e_y_log    = zeros(sim_steps, 1);
Y_log      = zeros(sim_steps, 1);
X_log      = zeros(sim_steps, 1);
psi_log    = zeros(sim_steps, 1);
u_log      = zeros(sim_steps, 1);
du_log     = zeros(sim_steps, 1);
v_log      = zeros(sim_steps, 1);
model_log  = strings(sim_steps, 1);

fprintf('Starting Hybrid MPC (%d steps)...\n', sim_steps);
fprintf('  KIN below %.1f m/s  |  DYN above %.1f m/s\n', v_switch_lo, v_switch_hi);

%% Main simulation loop
for k = 1 : sim_steps

    vx_now = v_ref(k);

    % Model switching with hysteresis
    if strcmp(active_model, 'kinematic') && vx_now >= v_switch_hi
        active_model = 'dynamic';
        r_init = max(vx_now, vx_min_dyn) * tan(u_prev) / l;
        x_dyn  = [0; e_kin(3); r_init; e_kin(2)];
        fprintf('  KIN to DYN at t=%.2fs (v=%.2f m/s)\n', t_uniform(k), vx_now);

    elseif strcmp(active_model, 'dynamic') && vx_now <= v_switch_lo
        active_model = 'kinematic';
        e_kin = [0; x_dyn(4); x_dyn(2)];
        fprintf('  DYN to KIN at t=%.2fs (v=%.2f m/s)\n', t_uniform(k), vx_now);
    end

    % Log
    if strcmp(active_model, 'kinematic')
        e_psi_log(k) = e_kin(3);
        e_y_log(k)   = e_kin(2);
    else
        e_psi_log(k) = x_dyn(2);
        e_y_log(k)   = x_dyn(4);
    end
    Y_log(k)     = Y_global;
    X_log(k)     = X_global;
    psi_log(k)   = psi_global;
    v_log(k)     = vx_now;
    model_log(k) = active_model;

    % Extract horizon data
    idx_h       = k : k + Np - 1;
    vx_fut_kin  = max(v_ref(idx_h), vx_min_kin);
    vx_fut_dyn  = max(v_ref(idx_h), vx_min_dyn);
    delta_r_fut = delta_r_ref(idx_h);
    psi_dot_fut = psi_dot_ref(idx_h);

    eps_r_future = [X_ref(idx_h), Y_ref(idx_h), psi_ref(idx_h)];

    % Constraints
    b_upper = ( u_max - u_prev) * ones(Np, 1);
    b_lower = ( u_max + u_prev) * ones(Np, 1);
    A_ineq  = [ A_cs; -A_cs];
    b_ineq  = [b_upper; b_lower];

    % Solve MPC
    if strcmp(active_model, 'kinematic')

        cost_fn = @(dU) costKinematic_GTZ(dU, e_kin, u_prev, Q_kin, R_kin, Np, ...
                                           Cd_kin, eps_r_future, vx_fut_kin, ...
                                           delta_r_fut, delta_t, l);
    else

        cost_fn = @(dU) costDynamic(dU, x_dyn, u_prev, Q_dyn, R_dyn, Np, ...
                                     Cd_dyn, psi_dot_fut, vx_fut_dyn, delta_t, ...
                                     Cf, Cr, m, Iz, l1, l2);
    end

    delta_U_opt = fmincon(cost_fn, delta_U_guess, A_ineq, b_ineq, ...
                          [], [], lb_du, ub_du, [], options);

    du_now    = delta_U_opt(1);
    u_current = u_prev + du_now;

    % Propagate plant
    if strcmp(active_model, 'kinematic')

        ode_kin = @(~, s) [vx_now * cos(s(3));
                            vx_now * sin(s(3));
                            vx_now * tan(u_current) / l];
        [~, s] = ode23s(ode_kin, [0, delta_t], [X_global; Y_global; psi_global]);

        X_global   = s(end,1);
        Y_global   = s(end,2);
        psi_global = s(end,3);

        e_kin(1) = X_global - X_ref(k+1);
        e_kin(2) = Y_global - Y_ref(k+1);
        e_kin(3) = angdiff(psi_ref(k+1), psi_global);

    else

        vx_pl     = max(vx_now, vx_min_dyn);
        A_c       = dynMatrix_A(vx_pl, Cf, Cr, m, Iz, l1, l2);
        B_c       = dynMatrix_B(Cf, m, l1, Iz);
        W_c       = [0; -psi_dot_ref(k); 0; 0];

        ode_dyn   = @(~, xs) A_c * xs + B_c * u_current + W_c;
        [~, s]    = ode23s(ode_dyn, [0, delta_t], x_dyn);
        x_dyn     = s(end,:)';

        psi_global = psi_ref(k+1) + x_dyn(2);
        Y_global   = Y_ref(k+1)   + x_dyn(4) * cos(psi_ref(k+1));
        X_global   = X_ref(k+1)   - x_dyn(4) * sin(psi_ref(k+1));
    end

    % Store & warm-start
    u_log(k)      = u_current;
    du_log(k)     = du_now;
    u_prev        = u_current;
    delta_U_guess = [delta_U_opt(2:end); 0];

end

fprintf('Done.\n');

%% Plots
t_plot  = t_uniform(1:sim_steps);
kin_idx = model_log == "kinematic";
dyn_idx = model_log == "dynamic";

figure(1);
subplot(4,1,1);
plot(t_plot, v_log, 'k-', 'LineWidth', 1.5); hold on;
scatter(t_plot(kin_idx), v_log(kin_idx), 15, 'b', 'filled');
scatter(t_plot(dyn_idx), v_log(dyn_idx), 15, 'r', 'filled');
grid on; ylabel('$v$ (m/s)', 'Interpreter','latex');
legend('Speed','Kinematic','Dynamic');

subplot(4,1,2);
plot(t_plot, Y_ref(1:sim_steps), 'r--', 'LineWidth', 2); hold on;
plot(t_plot, Y_log, 'b-', 'LineWidth', 1.5);
grid on; legend('Y_{ref}','Y_{actual}'); ylabel('$Y$ (m)', 'Interpreter','latex');

subplot(4,1,3);
plot(t_plot, psi_ref(1:sim_steps), 'r--', 'LineWidth', 2); hold on;
plot(t_plot, psi_log, 'b-', 'LineWidth', 1.5);
grid on; legend('\psi_{ref}','\psi_{actual}'); ylabel('$\psi$ (rad)', 'Interpreter','latex');

subplot(4,1,4);
plot(t_plot, rad2deg(u_log), 'k-', 'LineWidth', 1.5);
grid on; ylabel('$\delta_f$ (deg)', 'Interpreter','latex');
xlabel('Time (s)'); ylim([-30 30]);

figure(2);
subplot(3,1,1);
plot(t_plot, e_y_log, 'b-', 'LineWidth', 1.5);
grid on; ylabel('Y error $e_y$ (m)', 'Interpreter','latex');

subplot(3,1,2);
plot(t_plot, e_psi_log, 'r-', 'LineWidth', 1.5);
grid on; ylabel('Psi error $e_\psi$ (rad)', 'Interpreter','latex');

subplot(3,1,3);
plot(t_plot, rad2deg(du_log), 'k-', 'LineWidth', 1.5);
grid on; ylabel('$\Delta\delta_f$ (deg/step)', 'Interpreter','latex');
xlabel('Time (s)'); ylim([-10 10]);


%%  LOCAL FUNCTIONS

function J = costKinematic_GTZ(delta_U_seq, e_current, u_last, ...
                                Q, R, Np, Cd, ...
                                eps_r_future, vx_future, delta_r_future, dt, l)

    cost   = 0;
    e_pred = e_current;
    u_pred = u_last;

    for i = 1 : Np
        u_pred = u_pred + delta_U_seq(i);

        % Full reference state at step i (global coordinates)
        eps_r_i   = eps_r_future(i, :)';   % [X_r; Y_r; psi_r]
        vr_i      = vx_future(i);
        delta_r_i = delta_r_future(i);
        mu_r_i    = [vr_i; delta_r_i];

        % Taylor Jacobians at (eps_r_i, mu_r_i)
        A_T = [0, 0, -vr_i * sin(eps_r_i(3));
               0, 0,  vr_i * cos(eps_r_i(3));
               0, 0,  0                      ];

        B_T = [cos(eps_r_i(3)),                       0;
               sin(eps_r_i(3)),                       0;
               tan(delta_r_i)/l,  vr_i/(l*cos(delta_r_i)^2)];

        % Nonlinear dynamics at reference (exact evaluation)
        f_r = [vr_i * cos(eps_r_i(3));
               vr_i * sin(eps_r_i(3));
               vr_i * tan(delta_r_i) / l];

        % GTZ residual term, absorbs into A
        phi_xT = f_r - A_T * eps_r_i - B_T * mu_r_i;

        norm_sq     = eps_r_i' * eps_r_i;
        A_GTZ = A_T + (phi_xT * eps_r_i') / norm_sq;

        % Forward-Euler discretization
        Ad = eye(3) + A_GTZ * dt;
        Bd = B_T * dt;

        % Control deviation from reference
        mu_tilde = [0; u_pred - delta_r_i];

        % Propagate error state
        e_pred = Ad * e_pred + Bd * mu_tilde;

        % Output cost
        y_pred = Cd * e_pred;
        Q_i    = Q((i-1)*2+1 : i*2, (i-1)*2+1 : i*2);
        cost   = cost + y_pred' * Q_i * y_pred;
    end

    J = cost + delta_U_seq' * R * delta_U_seq;
end


function J = costDynamic(delta_U_seq, x_current, u_last, ...
                          Q, R, Np, Cd, psi_dot_fut, vx_fut, dt, ...
                          Cf, Cr, m, Iz, l1, l2)
    cost   = 0;
    x_pred = x_current;
    u_pred = u_last;
    B_c    = dynMatrix_B(Cf, m, l1, Iz);

    for i = 1 : Np
        u_pred = u_pred + delta_U_seq(i);
        A_c    = dynMatrix_A(vx_fut(i), Cf, Cr, m, Iz, l1, l2);
        Ad     = eye(4) + A_c * dt;
        Bd     = B_c * dt;
        W_d    = [0; -psi_dot_fut(i)*dt; 0; 0];

        x_pred = Ad * x_pred + Bd * u_pred + W_d;
        y_pred = Cd * x_pred;
        Q_i    = Q((i-1)*2+1 : i*2, (i-1)*2+1 : i*2);
        cost   = cost + y_pred' * Q_i * y_pred;
    end
    J = cost + delta_U_seq' * R * delta_U_seq;
end


function A_c = dynMatrix_A(vx, Cf, Cr, m, Iz, l1, l2)
    A_c = [-(2*Cf+2*Cr)/(m*vx),        0,  -(2*Cf*l1-2*Cr*l2)/(m*vx)-vx,   0;
                0,                      0,   1,                               0;
           -(2*l1*Cf-2*l2*Cr)/(Iz*vx), 0,  -(2*l1^2*Cf+2*l2^2*Cr)/(Iz*vx), 0;
               -1,                     -vx,  0,                               0];
end

function B_c = dynMatrix_B(Cf, m, l1, Iz)
    B_c = [(2*Cf)/m; 0; (2*l1*Cf)/Iz; 0];
end