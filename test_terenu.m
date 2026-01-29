addpath("kinematika_MR");

clc; clear; close all;

%% Load map
coords = [280 400; 280 520; 
         400 520; 400 400];
map = druhy('image.jpg', coords);

%% rest

dt = 0.01;
T  = 10;
N  = T/dt;

res = 1;        % map resolution [m]
mapH = map;       % height matrix
[ny, nx] = size(mapH);
[X, Y] = meshgrid( ...
    (0:nx-1) * res, ...
    (0:ny-1) * res );

state = [ ...
    10.0;   % x [m]
    51.0;   % y [m]
    0.0;   % theta [rad]
    0.0;   % v [m/s]
    0.0;   % omega [rad/s]
    0.20;  % z (clearance)
    0.0];  % zdot

M = [1.1; 1; 1.2; 1]; % motor torques

history = zeros(7, N);
z_abs_hist = zeros(1, N);

for k = 1:N
    % --- Robot dynamics (terrain-aware) ---
    state = trackedRobotDynamicsWithTerrain(state, M, dt, mapH, res);

    % --- Terrain height for visualization ---
    ix = max(1, min(round(state(1)/res)+1, size(mapH,2)));
    iy = max(1, min(round(state(2)/res)+1, size(mapH,1)));
    z_ground = mapH(iy, ix);

    z_abs_hist(k) = z_ground + state(6);
    history(:,k) = state;
end



figure;
surf(X, Y, mapH, 'EdgeColor', 'none');
colormap(gray);
hold on;

plot3(history(1,:), history(2,:), z_abs_hist, ...
      'r', 'LineWidth', 2);

xlabel('X [m]');
ylabel('Y [m]');
zlabel('Height [m]');
title('Tracked robot on terrain');
view(45,35);
axis equal;
camlight; lighting gouraud;
