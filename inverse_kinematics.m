robot = loadrobot("quanserQArm", DataFormat="row");
robot.Gravity = [0 0 -9.81];

q0 = homeConfiguration(robot);
eeName = robot.BodyNames{end};

p_des = [0.3, 0.1, 0.2];     % [x y z]

x = p_des(1);
y = p_des(2);
z = p_des(3);

r = norm(p_des);
rMax = 0.8 * 0.780;

% 1) fuori dal cerchio  OR  2) (x<0 E z<0) → NON va bene
if (r >= rMax) || ( (x < 0) && (z < 0) )
    disp("The configuration is NOT reachable");
    q_sol = q0;
else
    ik = inverseKinematics('RigidBodyTree', robot);

    T_des = trvec2tform(p_des);
    weights = [0.25 0.25 0.25  1 1 1];

    [q_sol, solInfo] = ik(eeName, T_des, weights, q0);

    q_sol
    solInfo.Status
end