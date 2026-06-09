clear; clc; close all;

%% =========================================================
%% DIFFERENT_GRAFES - single file, no external functions
%% 3 figures:
%% 1) Cubic Spline
%% 2) Potential Field
%% 3) Vector Step
%% Each figure: position, velocity, acceleration
%% =========================================================

%% ROBOT
robot = loadrobot("quanserQArm", DataFormat="row");
robot.Gravity = [0 0 -9.81];
eeName = robot.BodyNames{end};

q0 = homeConfiguration(robot);
q0 = q0(:)';   % force row 1x4

%% PARAMETERS
Ts = 0.05;
numSamples = 80;              % reduced to make it faster
tol = 0.01;                   % 1 cm
dqMax = deg2rad([90 90 90 90]);
safetyFactor = 1.2;

% target point
p_target = [0.50; 0.20; 0.1];

%% INVERSE KINEMATICS FOR FINAL CONFIGURATION
ik = inverseKinematics('RigidBodyTree', robot);
weights = [0.2 0.2 0.2 1 1 1];
T_target = trvec2tform(p_target') * eul2tform([0 pi 0]);

[q_end, ~] = ik(eeName, T_target, weights, q0);
q_end = q_end(:)';

%% TIME VECTOR
dq = abs(q_end - q0);
T_min = max(dq ./ dqMax);
T = safetyFactor * T_min;
t_vec = linspace(0, T, numSamples);
N = numSamples;

%% =========================================================
%% 1) CUBIC SPLINE
%% =========================================================
waypoints = [q0', q_end'];
[q_cub, ~, ~] = cubicpolytraj(waypoints, [0 T], t_vec);

%% =========================================================
%% 2) VECTOR STEP
%% =========================================================
alpha = 0.03;

q_vec = zeros(4, N);
q_vec(:,1) = q0(:);
q_prev = q0;

for k = 2:N
    Tcurr = getTransform(robot, q_prev, eeName);
    p_curr = Tcurr(1:3,4);

    e = p_target - p_curr;
    d = norm(e);

    if d < tol
        q_vec(:,k:end) = repmat(q_prev(:), 1, N-k+1);
        break;
    end

    step = alpha * e / max(d,1e-9);

    if norm(step) > d
        step = e;
    end

    p_cmd = p_curr + step;
    T_cmd = trvec2tform(p_cmd') * eul2tform([0 pi 0]);

    [q_new, ~] = ik(eeName, T_cmd, weights, q_prev);
    q_prev = q_new(:)';
    q_vec(:,k) = q_prev(:);
end

%% =========================================================
%% 3) POTENTIAL FIELD
%% =========================================================
q_pf = zeros(4, N);
q_pf(:,1) = q0(:);
q_prev = q0;

for k = 2:N
    Tcurr = getTransform(robot, q_prev, eeName);
    p_curr = Tcurr(1:3,4);

    e = p_target - p_curr;
    d = norm(e);

    if d < tol
        q_pf(:,k:end) = repmat(q_prev(:), 1, N-k+1);
        break;
    end

    % local function at end of this file
    p_cmd = potentialFieldPlanner(p_curr, p_target, Ts);

    T_cmd = trvec2tform(p_cmd') * eul2tform([0 pi 0]);
    [q_new, ~] = ik(eeName, T_cmd, weights, q_prev);

    q_prev = q_new(:)';
    q_pf(:,k) = q_prev(:);
end

%% =========================================================
%% FORWARD KINEMATICS
%% =========================================================
p_cub = zeros(3,N);
p_vec = zeros(3,N);
p_pf  = zeros(3,N);

for k = 1:N
    T1 = getTransform(robot, q_cub(:,k)', eeName);
    T2 = getTransform(robot, q_vec(:,k)', eeName);
    T3 = getTransform(robot, q_pf(:,k)', eeName);

    p_cub(:,k) = T1(1:3,4);
    p_vec(:,k) = T2(1:3,4);
    p_pf(:,k)  = T3(1:3,4);
end

%% =========================================================
%% VELOCITY AND ACCELERATION IN TASK SPACE
%% =========================================================
v_cub = zeros(3,N); a_cub = zeros(3,N);
v_vec = zeros(3,N); a_vec = zeros(3,N);
v_pf  = zeros(3,N); a_pf  = zeros(3,N);

v_cub(:,2:end) = diff(p_cub,1,2) / Ts;
v_vec(:,2:end) = diff(p_vec,1,2) / Ts;
v_pf(:,2:end)  = diff(p_pf,1,2) / Ts;

a_cub(:,2:end) = diff(v_cub,1,2) / Ts;
a_vec(:,2:end) = diff(v_vec,1,2) / Ts;
a_pf(:,2:end)  = diff(v_pf,1,2) / Ts;

speed_cub = vecnorm(v_cub,2,1);
speed_vec = vecnorm(v_vec,2,1);
speed_pf  = vecnorm(v_pf,2,1);

acc_cub = vecnorm(a_cub,2,1);
acc_vec = vecnorm(a_vec,2,1);
acc_pf  = vecnorm(a_pf,2,1);

%% =========================================================
%% FIGURE 1 - CUBIC SPLINE
%% =========================================================
figure('Name','Cubic Spline','Color','w','Position',[100 80 1000 800]);

subplot(3,1,1);
plot(t_vec, p_cub(1,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, p_cub(2,:), 'r', 'LineWidth', 1.8);
plot(t_vec, p_cub(3,:), 'g', 'LineWidth', 1.8);
grid on;
ylabel('Position [m]');
title('Cubic Spline - End-Effector Position');
legend('X','Y','Z','Location','best');

subplot(3,1,2);
plot(t_vec, v_cub(1,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, v_cub(2,:), 'r', 'LineWidth', 1.8);
plot(t_vec, v_cub(3,:), 'g', 'LineWidth', 1.8);
plot(t_vec, speed_cub, 'k--', 'LineWidth', 1.4);
grid on;
ylabel('Velocity [m/s]');
title('Cubic Spline - End-Effector Velocity');
legend('V_x','V_y','V_z','|V|','Location','best');

subplot(3,1,3);
plot(t_vec, a_cub(1,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, a_cub(2,:), 'r', 'LineWidth', 1.8);
plot(t_vec, a_cub(3,:), 'g', 'LineWidth', 1.8);
plot(t_vec, acc_cub, 'k--', 'LineWidth', 1.4);
grid on;
ylabel('Acceleration [m/s^2]');
xlabel('Time [s]');
title('Cubic Spline - End-Effector Acceleration');
legend('A_x','A_y','A_z','|A|','Location','best');

sgtitle('Cubic Spline Trajectory');

%% =========================================================
%% FIGURE 2 - POTENTIAL FIELD
%% =========================================================
figure('Name','Potential Field','Color','w','Position',[150 100 1000 800]);

subplot(3,1,1);
plot(t_vec, p_pf(1,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, p_pf(2,:), 'r', 'LineWidth', 1.8);
plot(t_vec, p_pf(3,:), 'g', 'LineWidth', 1.8);
grid on;
ylabel('Position [m]');
title('Potential Field - End-Effector Position');
legend('X','Y','Z','Location','best');

subplot(3,1,2);
plot(t_vec, v_pf(1,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, v_pf(2,:), 'r', 'LineWidth', 1.8);
plot(t_vec, v_pf(3,:), 'g', 'LineWidth', 1.8);
plot(t_vec, speed_pf, 'k--', 'LineWidth', 1.4);
grid on;
ylabel('Velocity [m/s]');
title('Potential Field - End-Effector Velocity');
legend('V_x','V_y','V_z','|V|','Location','best');

subplot(3,1,3);
plot(t_vec, a_pf(1,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, a_pf(2,:), 'r', 'LineWidth', 1.8);
plot(t_vec, a_pf(3,:), 'g', 'LineWidth', 1.8);
plot(t_vec, acc_pf, 'k--', 'LineWidth', 1.4);
grid on;
ylabel('Acceleration [m/s^2]');
xlabel('Time [s]');
title('Potential Field - End-Effector Acceleration');
legend('A_x','A_y','A_z','|A|','Location','best');

sgtitle('Potential Field Trajectory');

%% =========================================================
%% FIGURE 3 - VECTOR STEP
%% =========================================================
figure('Name','Vector Step','Color','w','Position',[200 120 1000 800]);

subplot(3,1,1);
plot(t_vec, p_vec(1,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, p_vec(2,:), 'r', 'LineWidth', 1.8);
plot(t_vec, p_vec(3,:), 'g', 'LineWidth', 1.8);
grid on;
ylabel('Position [m]');
title('Vector Step - End-Effector Position');
legend('X','Y','Z','Location','best');

subplot(3,1,2);
plot(t_vec, v_vec(1,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, v_vec(2,:), 'r', 'LineWidth', 1.8);
plot(t_vec, v_vec(3,:), 'g', 'LineWidth', 1.8);
plot(t_vec, speed_vec, 'k--', 'LineWidth', 1.4);
grid on;
ylabel('Velocity [m/s]');
title('Vector Step - End-Effector Velocity');
legend('V_x','V_y','V_z','|V|','Location','best');

subplot(3,1,3);
plot(t_vec, a_vec(1,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, a_vec(2,:), 'r', 'LineWidth', 1.8);
plot(t_vec, a_vec(3,:), 'g', 'LineWidth', 1.8);
plot(t_vec, acc_vec, 'k--', 'LineWidth', 1.4);
grid on;
ylabel('Acceleration [m/s^2]');
xlabel('Time [s]');
title('Vector Step - End-Effector Acceleration');
legend('A_x','A_y','A_z','|A|','Location','best');

sgtitle('Vector Step Trajectory');

%% =========================================================
%% LOCAL FUNCTION
%% Keep this at the end of the script
%% =========================================================
function p_cmd = potentialFieldPlanner(p_current, p_target, Ts)
% Simple attractive potential field
% Embedded here to avoid separate files

k_att = 2.0;      % attraction gain
v_max = 0.08;     % max Cartesian speed [m/s]
eps_n = 1e-9;

e = p_target - p_current;
d = norm(e);

if d < 1e-4
    p_cmd = p_current;
    return;
end

dir = e / max(d, eps_n);
v = min(k_att * d, v_max) * dir;
p_cmd = p_current + Ts * v;
end