%documentation: https://it.mathworks.com/help/robotics/ref/rigidbodytree.html

robot = loadrobot("quanserQArm", DataFormat="row");
robot.Gravity = [0 0 -9.81];
disp(robot.BodyNames)
allNames = robot.BodyNames;

q0 = homeConfiguration(robot);
tot = 0;

%% calcolo la lunghezza di ogni collegamento
T_base = robot.Bodies{1}.Joint.JointToParentTransform;
disp(T_base(1:3, 4))  % offset x, y, z

for i = 2:robot.NumBodies
    % Posizione del body corrente
    T_curr = getTransform(robot, q0, allNames{i});
    p_curr = T_curr(1:3,4);

    % Posizione del body precedente
    T_prev = getTransform(robot, q0, allNames{i-1});
    p_prev = T_prev(1:3,4);

    % Distanza euclidea tra i due
    lunghezza = norm(p_curr - p_prev);
    tot = tot+lunghezza;
    fprintf('%s → %s : %.4f, tot= %.4f m\n', allNames{i-1}, allNames{i}, lunghezza, tot);
    

end

figure; show(robot, homeConfiguration(robot));

%% workplace - work envelope
%generation of the obstacles for the enviroment

%generation of the table
sizeX = 1.0;   sizeY = 0.6;   sizeZ = 0.05;
xC = 0.2;   yC = 0.0;      zC = -0.525;

%documentation function || env = exampleHelperCreateWorkbench(0.4,1.5,0.05,[0.5 0.0 0.1]); env = exampleHelperCreateWorkbench(length_x,length_y,length_z,[centre_x centre_y centre_z]);

tableBox = collisionBox(sizeX, sizeY, sizeZ);
tableBox.Pose = trvec2tform([xC, yC, zC]);
env = {tableBox};


showCollisionArray(env);
hold on
show(robot,[0 0 0 0]);

%Generate the workspace for the robot in the collision environment.

rng default
[ws,configs] = generateRobotWorkspace(robot,env); %ws in in x,y,z; configs is in q0 q1 q2 q3

size(ws)
min(ws)
max(ws)

%Plot the workspace as an alpha shape.

wsAlpha = alphaShape(ws(:,1),ws(:,2),ws(:,3));
a = plot(wsAlpha,FaceAlpha=0.45,EdgeColor="none");
title("Robot Workspace in Specified Environment")
hold off
