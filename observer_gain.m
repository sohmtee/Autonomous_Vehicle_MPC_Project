clc; clear; close all;

fprintf('Creating observer gain schedule...\n');

% traj = readmatrix('C:\Users\soguchie\OneDrive - purdue.edu\ECE 699\Summer 2025\RoadRunner Projects\RoadRunner Project 1\Exports\SingaporeIntersection_design_diff_velocities\Sedan3_3.csv');
traj = readmatrix('C:\Users\soguchie\OneDrive - purdue.edu\ECE 699\Summer 2025\RoadRunner Projects\RoadRunner Project 1\Exports\SingaporeIntersection_design_varying_velocities2_for_hybrid_models\Sedan3_3a.csv');
% traj = readmatrix('C:\Users\soguchie\OneDrive - purdue.edu\ECE 699\Summer 2025\RoadRunner Projects\RoadRunner Project 1\Exports\BostonIntersection_design_varying_velocities_for_hybrid\Sedan_1.csv');

time = traj(:,1);
vx = traj(:,8);

delta_t = 0.05;         % formerly 0.01. but 0.05 works
t_uniform = 0:delta_t:time(end);

vx_ref = interp1(time, vx, t_uniform, 'linear');
vx_ref = smooth(vx_ref, 25);    % formerly 10 for varying velo for dynamic model only

% define vehicle parameters
vehicle_params = struct();
vehicle_params.Cf = 30000; 
vehicle_params.Cr = 30000;
vehicle_params.l1 = 1.2;
vehicle_params.l2 = 1.22;
vehicle_params.m = 1280;
vehicle_params.Iz = 2500;

Cf = vehicle_params.Cf;
Cr = vehicle_params.Cr;
l1 = vehicle_params.l1;
l2 = vehicle_params.l2;
m = vehicle_params.m;
Iz = vehicle_params.Iz;

C_c = [0 1 0 0; 0 0 0 1];
D_c = zeros(2,1);

nn = 4; pp = 2;

% check UIO existence
disp('--- UIO feasibility check ---')

% vx_test = linspace(min(vx_ref), max(vx_ref), 20);
vx_grid = 0:0.1:max(vx_ref);      % formerly was min:1:max  min(vx_ref):0.5:max(vx_ref)
for i = 1:length(vx_grid)
    [A_c_i, B_c_i] = computeVehicleModel(vx_grid(i), vehicle_params);
    sysd_i = c2d(ss(A_c_i, B_c_i, C_c, D_c), delta_t);
    if rank(sysd_i.C * sysd_i.B) ~= rank(sysd_i.B)
        error('Rank condition fails at vx = %.2f m/s', vx_grid(i));
    end
end
disp('Rank condition satisfied across entire velocity range')

N_schedule = length(vx_grid);   % number of velocity points
L_schedule = zeros(nn, pp, N_schedule);
M_schedule = zeros(nn, pp, N_schedule);

% solve LMI at each velocity point
for i = 1:N_schedule
    vx_i = vx_grid(i);

    [A_c_i, B_c_i] = computeVehicleModel(vx_i, vehicle_params);
    sysd_i = c2d(ss(A_c_i, B_c_i, C_c, D_c), delta_t);

    % compute M for this velocity
    M_schedule(:,:,i) = sysd_i.B * pinv(sysd_i.C * sysd_i.B);

    % compute A_1 for this velocity
    P_t_i = eye(nn) - M_schedule(:,:,i) * sysd_i.C;
    A_1_i = P_t_i * sysd_i.A;

    % solve LMI for L at this velocity
    cvx_begin sdp quiet
    variable P_i(nn, nn) symmetric
    variable Y_i(nn, pp)
    [-P_i, A_1_i'*P_i - sysd_i.C'*Y_i'; 
         P_i*A_1_i - Y_i*sysd_i.C, -P_i] <= 0;
    P_i >= eps*eye(nn)
    cvx_end

    if strcmp(cvx_status, 'Solved') || strcmp(cvx_status, 'Inaccurate/Solved')
        L_schedule(:,:,i) = P_i \ Y_i;
        fprintf('Designed observer gain at vx = %.2f m/s (Status: %s)\n', vx_i, cvx_status);
    else
        error('CVX failed at vx = %.2f m/s (Status: %s)', vx_i, cvx_status);
    end
end

save('observer_gain_schedule.mat', 'vx_grid', 'L_schedule', 'M_schedule');
disp('--- gains saved ---')



%% helper functions

function [A_c, B_c] = computeVehicleModel(vx, params)
    % Compute continuous-time vehicle model matrices for given velocity
    
    Cf = params.Cf;
    Cr = params.Cr;
    l1 = params.l1;
    l2 = params.l2;
    m = params.m;
    Iz = params.Iz;
    
    % Avoid division by zero
    if vx < 0.1
        vx = 0.1;
    end
    
    % State vector: [vy, psi, r, Y]
    A_c = [-(2*Cf + 2*Cr)/(m*vx),           0,    -(2*Cf*l1 - 2*Cr*l2)/(m*vx) - vx,    0;
                0,                          0,     1,                                   0;
           -(2*l1*Cf - 2*l2*Cr)/(Iz*vx),    0,    -(2*l1^2*Cf + 2*l2^2*Cr)/(Iz*vx),   0;
                -1,                        -vx,    0,                                   0];
    
    B_c = [(2*Cf)/m; 0; (2*l1*Cf)/Iz; 0];
end