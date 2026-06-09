%% =========================================================
%% END-EFFECTOR: position, velocity, acceleration
%% comparison of 3 methods
%% =========================================================

N = length(t_vec);

% Cartesian positions
p_cub = zeros(3,N);
p_vec = zeros(3,N);
p_pf  = zeros(3,N);

for k = 1:N
    T_cub = getTransform(robot, q_cub(:,k)', eeName);
    T_vec = getTransform(robot, q_vec(:,k)', eeName);
    T_pf  = getTransform(robot, q_pf(:,k)',  eeName);

    p_cub(:,k) = T_cub(1:3,4);
    p_vec(:,k) = T_vec(1:3,4);
    p_pf(:,k)  = T_pf(1:3,4);
end

% Cartesian velocities
v_cub = zeros(3,N);
v_vec = zeros(3,N);
v_pf  = zeros(3,N);

v_cub(:,2:end) = diff(p_cub,1,2) / Ts;
v_vec(:,2:end) = diff(p_vec,1,2) / Ts;
v_pf(:,2:end)  = diff(p_pf,1,2)  / Ts;

% Cartesian accelerations
a_cub = zeros(3,N);
a_vec = zeros(3,N);
a_pf  = zeros(3,N);

a_cub(:,2:end) = diff(v_cub,1,2) / Ts;
a_vec(:,2:end) = diff(v_vec,1,2) / Ts;
a_pf(:,2:end)  = diff(v_pf,1,2)  / Ts;

% Norms
speed_cub = vecnorm(v_cub,2,1);
speed_vec = vecnorm(v_vec,2,1);
speed_pf  = vecnorm(v_pf,2,1);

accnorm_cub = vecnorm(a_cub,2,1);
accnorm_vec = vecnorm(a_vec,2,1);
accnorm_pf  = vecnorm(a_pf,2,1);

%% =========================================================
%% 3D TRAJECTORY
%% =========================================================
figure('Name','End-Effector 3D Trajectory','Position',[100 100 900 700]);
plot3(p_cub(1,:), p_cub(2,:), p_cub(3,:), 'b', 'LineWidth', 2); hold on;
plot3(p_vec(1,:), p_vec(2,:), p_vec(3,:), 'g--', 'LineWidth', 2);
plot3(p_pf(1,:),  p_pf(2,:),  p_pf(3,:),  'r', 'LineWidth', 2);
grid on; axis equal;
xlabel('X Position [m]');
ylabel('Y Position [m]');
zlabel('Z Position [m]');
title('3D End-Effector Trajectory Comparison');
legend('Cubic Spline','Vector Step','Potential Field','Location','best');

%% =========================================================
%% POSITION
%% =========================================================
figure('Name','End-Effector Position','Position',[100 100 1200 800]);

subplot(3,1,1);
plot(t_vec, p_cub(1,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, p_vec(1,:), 'g--', 'LineWidth', 1.8);
plot(t_vec, p_pf(1,:),  'r', 'LineWidth', 1.8);
grid on;
ylabel('X Position [m]');
title('End-Effector Position Comparison');
legend('Cubic Spline','Vector Step','Potential Field','Location','best');

subplot(3,1,2);
plot(t_vec, p_cub(2,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, p_vec(2,:), 'g--', 'LineWidth', 1.8);
plot(t_vec, p_pf(2,:),  'r', 'LineWidth', 1.8);
grid on;
ylabel('Y Position [m]');
title('Y-Axis Position');

subplot(3,1,3);
plot(t_vec, p_cub(3,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, p_vec(3,:), 'g--', 'LineWidth', 1.8);
plot(t_vec, p_pf(3,:),  'r', 'LineWidth', 1.8);
grid on;
ylabel('Z Position [m]');
xlabel('Time [s]');
title('Z-Axis Position');

%% =========================================================
%% VELOCITY
%% =========================================================
figure('Name','End-Effector Velocity','Position',[120 120 1200 850]);

subplot(4,1,1);
plot(t_vec, v_cub(1,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, v_vec(1,:), 'g--', 'LineWidth', 1.8);
plot(t_vec, v_pf(1,:),  'r', 'LineWidth', 1.8);
grid on;
ylabel('V_x [m/s]');
title('End-Effector Velocity Comparison');
legend('Cubic Spline','Vector Step','Potential Field','Location','best');

subplot(4,1,2);
plot(t_vec, v_cub(2,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, v_vec(2,:), 'g--', 'LineWidth', 1.8);
plot(t_vec, v_pf(2,:),  'r', 'LineWidth', 1.8);
grid on;
ylabel('V_y [m/s]');
title('Velocity along Y Axis');

subplot(4,1,3);
plot(t_vec, v_cub(3,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, v_vec(3,:), 'g--', 'LineWidth', 1.8);
plot(t_vec, v_pf(3,:),  'r', 'LineWidth', 1.8);
grid on;
ylabel('V_z [m/s]');
title('Velocity along Z Axis');

subplot(4,1,4);
plot(t_vec, speed_cub, 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, speed_vec, 'g--', 'LineWidth', 1.8);
plot(t_vec, speed_pf,  'r', 'LineWidth', 1.8);
grid on;
ylabel('|V| [m/s]');
xlabel('Time [s]');
title('End-Effector Speed Magnitude');

%% =========================================================
%% ACCELERATION
%% =========================================================
figure('Name','End-Effector Acceleration','Position',[140 140 1200 850]);

subplot(4,1,1);
plot(t_vec, a_cub(1,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, a_vec(1,:), 'g--', 'LineWidth', 1.8);
plot(t_vec, a_pf(1,:),  'r', 'LineWidth', 1.8);
grid on;
ylabel('A_x [m/s^2]');
title('End-Effector Acceleration Comparison');
legend('Cubic Spline','Vector Step','Potential Field','Location','best');

subplot(4,1,2);
plot(t_vec, a_cub(2,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, a_vec(2,:), 'g--', 'LineWidth', 1.8);
plot(t_vec, a_pf(2,:),  'r', 'LineWidth', 1.8);
grid on;
ylabel('A_y [m/s^2]');
title('Acceleration along Y Axis');

subplot(4,1,3);
plot(t_vec, a_cub(3,:), 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, a_vec(3,:), 'g--', 'LineWidth', 1.8);
plot(t_vec, a_pf(3,:),  'r', 'LineWidth', 1.8);
grid on;
ylabel('A_z [m/s^2]');
title('Acceleration along Z Axis');

subplot(4,1,4);
plot(t_vec, accnorm_cub, 'b', 'LineWidth', 1.8); hold on;
plot(t_vec, accnorm_vec, 'g--', 'LineWidth', 1.8);
plot(t_vec, accnorm_pf,  'r', 'LineWidth', 1.8);
grid on;
ylabel('|A| [m/s^2]');
xlabel('Time [s]');
title('End-Effector Acceleration Magnitude');
