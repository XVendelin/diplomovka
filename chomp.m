clc; clear; close all;

%% Load map
map = druhy('image.jpg', [250 370; 220 550; 450 550; 450 350]);

%% Define waypoints
waypoints = [40 45; 200 30; 40 65; 200 65; 40 90; 200 85; 40 120; 200 120; 40 150; 200 150; 40 size(map,2)-10;
             size(map,1)-10, size(map,2)-10]; % goal

%% Display map with waypoints
figure;
imshow(map, []); colormap('turbo'); colorbar;
hold on;
plot(waypoints(:,2), waypoints(:,1), 'ro', 'MarkerSize',10,'MarkerFaceColor','r');
text(waypoints(:,2)+3, waypoints(:,1), ...
     arrayfun(@num2str,1:size(waypoints,1),'UniformOutput',false),'Color','w');
title('Height Map with Waypoints (1 = too steep)');

%% Compute CHOMP paths between all waypoints
fullPath = [];
for i = 1:(size(waypoints,1)-1)
    start = waypoints(i,:);
    goal = waypoints(i+1,:);
    segment = chomp_height(map, start, goal);
    if isempty(segment)
        disp(['CHOMP failed between waypoint ', num2str(i), ' and ', num2str(i+1)]);
        break;
    end
    if i > 1
        segment = segment(2:end,:);
    end
    fullPath = [fullPath; segment];
end

%% Display full path
figure;
imshow(map, []); colormap('turbo'); colorbar;
hold on;
plot(fullPath(:,2), fullPath(:,1), 'w-', 'LineWidth', 2);
plot(waypoints(:,2), waypoints(:,1), 'go', 'MarkerSize',10,'MarkerFaceColor','g');
title('CHOMP Optimized Path with Height-Aware Cost');

%% find path
function path = chomp_height(map, start, goal)
    [rows, cols] = size(map);

    if map(start(1), start(2)) >= 1 || map(goal(1), goal(2)) >= 1
        path = [];
        warning('Start or goal is in an obstacle!');
        return;
    end

    % ==== Parameters ====
    N = 150;
    lambda = 2000;
    alpha = 70;
    eta = 0.0005;
    maxIter = 400;

    % ==== Initialize straight line ====
    path = [linspace(start(1), goal(1), N)', linspace(start(2), goal(2), N)'];

    % ==== Laplacian (smoothness operator) ====
    K = eye(N);
    for i = 2:N-1
        K(i, i-1:i+1) = [1 -2 1];
    end

    % ==== Optimization loop ====
    for iter = 1:maxIter
        grad_smooth = alpha * (K' * K * path);
        grad_terrain = zeros(size(path));

        for j = 2:N-1
            r = round(path(j,1));
            c = round(path(j,2));
            if r < 1 || r > rows || c < 1 || c > cols
                continue;
            end

            h = map(r,c);

            if h >= 1
                grad_terrain(j,:) = 1000 * rand(1,2);
                continue;
            end

            dr = 0; dc = 0;
            if r > 1 && r < rows
                dr = (map(r+1,c) - map(r-1,c)) / 2;
            end
            if c > 1 && c < cols
                dc = (map(r,c+1) - map(r,c-1)) / 2;
            end

            grad_terrain(j,:) = lambda * h * [dr, dc];
        end

        grad_total = grad_smooth + grad_terrain;
        path = path - eta * grad_total;

        path(1,:) = start;
        path(end,:) = goal;

        path(:,1) = min(max(path(:,1), 1), rows);
        path(:,2) = min(max(path(:,2), 1), cols);
    end
end

