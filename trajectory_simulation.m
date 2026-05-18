robot = loadrobot("quanserQArm", DataFormat="row");
robot.Gravity = [0 0 -9.81];
disp(robot.BodyNames)


% HOME POSITION: INITIAL POSITION           
        
q0 = homeConfiguration(robot);
%INITIAL RADIANT POSITION
%q0 = [0, -pi/4, pi/4, 0];  % [joint1, joint2, joint3, joint4]

%   3d model visualization
figure; show(robot, q0);
title('QArm - Configurazione Home');
axis([-0.5 0.5 -0.5 0.5 0 0.8]);

% Mostra info sui joint
showdetails(robot)

%% usando il joint workspace, generiamo le triaettorie con le diverse interpolazioni
%fissando come target un punto nel joint position
% a q0= end is used and then the interpolation between begining and end is
% performed using trapezoidal, cubic, quint and linear then, position,
% velocity and accelleration at each joint is extracted and plot with
% different colors

% Waypoint in joint space 
q0    = homeConfiguration(robot);
%END POSITION
%q_end = [-pi/6, -pi/4, pi/3, -pi/12];
q_end = [-2.8198, -1.4830, -0.0606, -0.3218];
%q_end = [-pi/6, -pi/4, pi/3, -pi/12];

waypoints  = [q0', q_end'];   % 4×2 — due colonne: inizio e fine

%calcolaiamo il tempo necessario mettendo come massimo 90 gradi per giunt
%al secondo
% velocità massime dei giunti (rad/s)
dqMax = deg2rad([90 90 90 90]);   % puoi affinare con i veri valori

% 1) variazione per ogni giunto
dq = abs(q_end - q0);        % 1x4

% 2) tempo minimo per rispettare i limiti
T_min = max(dq ./ dqMax);

% 3) scegli il tempo effettivo del segmento
safetyFactor = 1.2;               % un po' più lento del minimo
T = safetyFactor * T_min;
T
%% stampiamo T e lo impostiamo come massimo punto 
timePoints = [0, T];          % da 0 a 3 secondi
numSamples = 300;
t_vec = linspace(0, 3, numSamples);

% =====================================================
%  GENERA TUTTE LE TRAIETTORIE
% =====================================================

% 1. Trapezoidale
[q_trap, qd_trap, qdd_trap] = trapveltraj(waypoints, numSamples, 'EndTime', 3);


% 2. Cubica (polinomio grado 3, C1 — velocità continue)
[q_cub, qd_cub, qdd_cub] = cubicpolytraj(waypoints, timePoints, t_vec);

% 3. Quintica (polinomio grado 5, C2 — acc. continue)
[q_qui, qd_qui, qdd_qui] = quinticpolytraj(waypoints, timePoints, t_vec);

% 4. LERP — interpolazione lineare (solo posizioni, no vel/acc)
q_lerp = zeros(4, numSamples);
for j = 1:4
    q_lerp(j,:) = linspace(q0(j), q_end(j), numSamples);
end

% =====================================================
%  PLOT — 3 righe (pos, vel, acc) x 4 colonne (joint)
% =====================================================
jointNames = {'Joint 1 — Base', 'Joint 2 — Shoulder', 'Joint 3 — Elbow', 'Joint 4 — Wrist'};
colors = {'r', 'b', 'g', 'm'};
labels = {'Trapezoidale', 'Cubica', 'Quintica', 'Lineare (LERP)'};

figure('Name', 'Confronto Traiettorie QArm', 'Position', [50 50 1600 900]);

for j = 1:4

    % --- Posizione ---
    subplot(3, 4, j);
    hold on;
    plot(t_vec, q_trap(j,:),  'r',  LineWidth=1.8);
    plot(t_vec,  q_cub(j,:),   'b',  LineWidth=1.8);
    plot(t_vec,  q_qui(j,:),   'g',  LineWidth=1.8);
    plot(t_vec,  q_lerp(j,:),  'm--',LineWidth=1.8);
    title(jointNames{j}); ylabel('pos [rad]');
    if j == 1; legend(labels, Location="best"); end
    grid on; hold off;

    % --- Velocità ---
    subplot(3, 4, 4+j);
    hold on;
    plot(t_vec, qd_trap(j,:), 'r',  LineWidth=1.8);
    plot(t_vec,  qd_cub(j,:),  'b',  LineWidth=1.8);
    plot(t_vec,  qd_qui(j,:),  'g',  LineWidth=1.8);
    % LERP ha velocità costante
    plot(t_vec, ones(1,numSamples)*(q_end(j)-q0(j))/3, 'm--', LineWidth=1.8);
    ylabel('vel [rad/s]'); grid on; hold off;

    % --- Accelerazione ---
    subplot(3, 4, 8+j);
    hold on;
    plot(t_vec, qdd_trap(j,:), 'r', LineWidth=1.8);
    plot(t_vec,  qdd_cub(j,:),  'b', LineWidth=1.8);
    plot(t_vec,  qdd_qui(j,:),  'g', LineWidth=1.8);
    ylabel('acc [rad/s²]'); xlabel('tempo [s]'); grid on; hold off;

end

sgtitle('Confronto Traiettorie — QArm: Trapezoidale / Cubica / Quintica / Lineare', ...
        FontSize=14, FontWeight='bold');

%% robot 3d plot
%the traslation to the cartesian position given the joint angle is needed

trajs      = {q_trap, q_cub, q_qui, q_lerp};
trajNames  = {'Trapezoidale', 'Cubica', 'Quintica', 'Lineare'};
trajColors = {'r', 'b', 'g', 'm'};
trajStyles = {'-', '-', '-', '--'};

% --- Nomi presi DIRETTAMENTE dal modello (sicuro al 100%) ---
allNames  = robot.BodyNames;
disp('Tutti i body nel modello:');
disp(allNames)  % stampa per verifica

bodyNames  = {allNames{3}, allNames{4}, allNames{5}};
bodyTitles = {
    [allNames{3}, ' — varia con YAW'], ...
    [allNames{4}, ' — varia con YAW e SHOULDER'], ...
    [allNames{5}, ' — posizione finale']
};

% =====================================================
%  CALCOLA xyz per ogni body, ogni traiettoria
% =====================================================
xyz_all = cell(3, 4);

for b = 1:3
    for t = 1:4
        xyz = zeros(numSamples, 3);
        for i = 1:numSamples
            T = getTransform(robot, trajs{t}(:,i)', bodyNames{b});
            xyz(i,:) = T(1:3,4)';
        end

        % Verifica di sanità: i valori devono essere < 2m
        if max(abs(xyz(:))) > 2
            warning('Valori anomali per body %s — controlla il nome!', bodyNames{b});
        end

        xyz_all{b,t} = xyz;
    end
end

% =====================================================
%  3 GRAFICI — uno per body
% =====================================================
figure('Name', 'Percorsi 3D per Body', ...
       'Position', [50 50 1500 500], ...
       'NumberTitle', 'off');

for b = 1:3
    subplot(1, 3, b);
    hold on;

    for t = 1:4
        xyz = xyz_all{b,t};

        plot3(xyz(:,1), xyz(:,2), xyz(:,3), ...
              [trajColors{t}, trajStyles{t}], ...
              LineWidth=2, ...
              DisplayName=trajNames{t});

        % Punto iniziale
        plot3(xyz(1,1), xyz(1,2), xyz(1,3), 'o', ...
              Color=trajColors{t}, MarkerSize=7, ...
              MarkerFaceColor=trajColors{t}, ...
              HandleVisibility='off');

        % Punto finale
        plot3(xyz(end,1), xyz(end,2), xyz(end,3), '^', ...
              Color=trajColors{t}, MarkerSize=7, ...
              MarkerFaceColor=trajColors{t}, ...
              HandleVisibility='off');
    end

    title(bodyTitles{b});
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    legend(Location='best');
    grid on; axis equal;
    view(45, 25);
    hold off;
end

sgtitle('Percorso Cartesiano dei Body — 4 Tipi di Traiettoria', ...
        FontSize=14, FontWeight='bold');