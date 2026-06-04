clc; clear; close all;

fprintf('Creating observer gain schedule (2D: v_x x psi_ref)...\n');

% traj = readmatrix('C:\Users\soguchie\OneDrive - purdue.edu\ECE 699\Summer 2025\RoadRunner Projects\RoadRunner Project 1\Exports\SingaporeIntersection_design_2026-03-24_small\Sedan_1.csv');
traj = readmatrix('C:\Users\soguchie\OneDrive - purdue.edu\ECE 699\Summer 2025\RoadRunner Projects\RoadRunner Project 1\Exports\BostonIntersection_design\Sedan_1.csv');

time     = traj(:,1);
psi_path = traj(:,5);
vx_raw   = traj(:,8);

delta_t = 0.05;
t_uniform = (0 : delta_t : time(end))';

% (psi_unwrap used only to inspect the reference; gain grid is over [-pi, pi])
psi_unwrapped = unwrap(psi_path);
vx_ref  = interp1(time, vx_raw,         t_uniform, 'linear');
psi_ref = interp1(time, psi_unwrapped,  t_uniform, 'pchip');

% Vehicle parameters
vehicle_params = struct();
vehicle_params.Cf = 30000;
vehicle_params.Cr = 30000;
vehicle_params.l1 = 1.4225;
vehicle_params.l2 = 1.4225;
vehicle_params.m  = 1280;
vehicle_params.Iz = 2500;

% Dimensions
nn = 5;   % states
pp = 3;   % output: [e_x_path; e_psi; e_y_path]       (path-frame, time-varying C)
mm = 2;   % input:  [v_x, delta_f]

% --- Grids ---
% v_x grid: same as before
vx_min_dyn = 0.5;
vx_grid    = vx_min_dyn : 0.1 : max(vx_ref);
N_vx       = length(vx_grid);

psi_step  = pi/18;
psi_grid  = (-pi : psi_step : pi)';
N_psi     = length(psi_grid);

fprintf('Grid sizes: N_vx = %d, N_psi = %d, total LMIs = %d\n', ...
        N_vx, N_psi, N_vx * N_psi);

psi_dot_nom = 0;
v_y_nom     = 0;

% --- UIO feasibility check over the 2D grid ---
disp('--- UIO feasibility check across (v_x, psi_ref) ---');
for i = 1:N_vx
    for j = 1:N_psi
        [A_c, B_c, C_c] = computeVehicleModel(vx_grid(i), psi_grid(j), ...
                                              vehicle_params, psi_dot_nom, v_y_nom);
        sysd = c2d(ss(A_c, B_c, C_c, zeros(pp, mm)), delta_t);
        CB   = sysd.C * sysd.B;
        if rank(CB) ~= rank(sysd.B)
            error('Rank condition fails at vx = %.2f, psi_ref = %.3f rad', ...
                  vx_grid(i), psi_grid(j));
        end
    end
end
disp('Rank condition satisfied across entire (v_x, psi_ref) grid');

% --- LMI-based gain synthesis ---
L_schedule = zeros(nn, pp, N_vx, N_psi);
M_schedule = zeros(nn, pp, N_vx, N_psi);

n_total = N_vx * N_psi;
count   = 0;
for i = 1:N_vx
    for j = 1:N_psi
        count = count + 1;
        vx_i  = vx_grid(i);
        psi_j = psi_grid(j);

        [A_c, B_c, C_c] = computeVehicleModel(vx_i, psi_j, vehicle_params, ...
                                              psi_dot_nom, v_y_nom);
        sysd = c2d(ss(A_c, B_c, C_c, zeros(pp, mm)), delta_t);
        Ad = sysd.A;  Bd = sysd.B;  Cd = sysd.C;

        % M = B_d * pinv(C_d * B_d)
        M_ij = Bd * pinv(Cd * Bd);
        M_schedule(:,:,i,j) = M_ij;

        % A_1 = (I - M*C) * A
        A_1 = (eye(nn) - M_ij * Cd) * Ad;

        % LMI for L:
        %   [ -P,        A_1'*P - C'*Y' ;
        %     P*A_1 - Y*C,        -P    ] <= 0,    P > 0
        cvx_begin sdp quiet
            variable P_ij(nn, nn) symmetric
            variable Y_ij(nn, pp)
            [-P_ij,                  A_1'*P_ij - Cd'*Y_ij'; 
              P_ij*A_1 - Y_ij*Cd,                    -P_ij] <= 0;
            P_ij >= eps * eye(nn);
        cvx_end

        if strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')
            L_schedule(:,:,i,j) = P_ij \ Y_ij;
            if mod(count, 50) == 0 || count == n_total
                fprintf('  [%4d / %d]  vx = %5.2f  psi = %+6.3f  (%s)\n', ...
                        count, n_total, vx_i, psi_j, cvx_status);
            end
        else
            error('CVX failed at vx = %.2f, psi_ref = %.3f rad (status: %s)', ...
                  vx_i, psi_j, cvx_status);
        end
    end
end

save('observer_gain_schedule_with_M.mat', 'vx_grid', 'psi_grid', ...
     'L_schedule', 'M_schedule');
disp('--- gains saved (2D schedule) ---');

%% =========================================================================
function [A_c, B_c, C_c] = computeVehicleModel(vx, psi_ref, params, ...
                                                psi_dot_ref, v_y_ref)
    Cf = params.Cf;
    Cr = params.Cr;
    l1 = params.l1;
    l2 = params.l2;
    m  = params.m;
    Iz = params.Iz;

    if vx < 0.5
        vx = 0.5;
    end

    cp = cos(psi_ref);
    sp = sin(psi_ref);

    % --- A matrix ---
    A_c = zeros(5, 5);
    A_c(1,1) = -(2*Cf + 2*Cr)/(m*vx);
    A_c(1,3) = -(2*Cf*l1 - 2*Cr*l2)/(m*vx) - vx;
    A_c(2,3) =  1;
    A_c(3,1) = -(2*l1*Cf - 2*l2*Cr)/(Iz*vx);
    A_c(3,3) = -(2*l1^2*Cf + 2*l2^2*Cr)/(Iz*vx);
    A_c(4,1) =  cp;
    A_c(4,2) =  vx*cp - v_y_ref*sp;
    A_c(5,1) = -sp;
    A_c(5,2) = -vx*sp - v_y_ref*cp;

    % --- B matrix ---
    df1_dvx = ((2*Cf + 2*Cr)/(m*vx^2)) * v_y_ref ...
            + ((2*Cf*l1 - 2*Cr*l2)/(m*vx^2) - 1) * psi_dot_ref;
    df3_dvx = ((2*l1*Cf - 2*l2*Cr)/(Iz*vx^2)) * v_y_ref ...
            + ((2*l1^2*Cf + 2*l2^2*Cr)/(Iz*vx^2)) * psi_dot_ref;

    B_v     = [df1_dvx; 0; df3_dvx; sp; cp];
    B_steer = [(2*Cf)/m; 0; (2*l1*Cf)/Iz; 0; 0];
    B_c     = [B_v, B_steer];

    % --- C matrix ---
    C_c = [ 0, 0, 0,  sp,  cp;       % e_x_path
            0, 1, 0,   0,   0;       % e_psi
            0, 0, 0,  cp, -sp ];     % e_y_path
end