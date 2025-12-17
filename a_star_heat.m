clc; clear; close all;

%% Load map
% map = druhy('image.jpg', [250 370; 220 550; 450 550; 450 350]);

%% Define waypoints
% waypoints = [40 45; 200 30; 40 65; 200 65; 40 90; 200 85; 40 120; 200 120; 40 150; 200 150; 40 size(map,2)-10;
             % size(map,1)-10, size(map,2)-10]; % goal
             
%% Load map
coords = [280 400; 280 520; 
         400 520; 400 400];
map = druhy('image.jpg', coords);

nmap = map; 
nmap(nmap ~= 1) = 0;

%% Define waypoints
waypoints = [10 20;
            100 10]; % goal

%% Compute path through all waypoints
fullPath = [];
for i = 1:(size(waypoints,1)-1)
    start = waypoints(i,:);
    goal = waypoints(i+1,:);
    segment = astar_height(map, start, goal);
    if isempty(segment)
        disp(['No path found between waypoint ', num2str(i), ' and ', num2str(i+1)]);
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
% map(map ~= 1) = 0;
% imshow(map, []);
hold on;
plot(fullPath(:,2), fullPath(:,1), 'w-', 'LineWidth', 2);
plot(waypoints(:,2), waypoints(:,1), 'go', 'MarkerSize',10,'MarkerFaceColor','g');
title('A* Path Considering Height Difficulty');

%% --- Height-aware A* function ---
function path = astar_height(map, start, goal)
    [rows, cols] = size(map);
    if map(start(1), start(2)) == 1 || map(goal(1), goal(2)) == 1
        path = [];
        return;
    end

    alpha = 40;

    openSet = false(rows, cols);
    cameFrom = zeros(rows, cols, 2);
    gScore = inf(rows, cols);
    fScore = inf(rows, cols);

    gScore(start(1), start(2)) = 0;
    fScore(start(1), start(2)) = heuristic(start, goal);
    openSet(start(1), start(2)) = true;

    while any(openSet(:))
        [~, idx] = min(fScore(:) + ~openSet(:)*1e6);
        [current_r, current_c] = ind2sub(size(map), idx);
        current = [current_r, current_c];

        if all(current == goal)
            path = current;
            while any(cameFrom(path(1,1), path(1,2),:))
                prev = squeeze(cameFrom(path(1,1), path(1,2),:))';
                path = [prev; path];
            end
            return;
        end

        openSet(current_r, current_c) = false;

        for dr = -1:1
            for dc = -1:1
                if dr == 0 && dc == 0, continue; end
                nr = current_r + dr; nc = current_c + dc;
                if nr < 1 || nr > rows || nc < 1 || nc > cols, continue; end
                if map(nr,nc) == 1, continue; end

                baseCost = sqrt(dr^2 + dc^2);

                elevationDiff = abs(map(nr,nc) - map(current_r,current_c));
                elevationCost = alpha * elevationDiff;

                tentative_g = gScore(current_r, current_c) + baseCost + elevationCost;

                if tentative_g < gScore(nr,nc)
                    cameFrom(nr,nc,:) = current;
                    gScore(nr,nc) = tentative_g;
                    fScore(nr,nc) = tentative_g + heuristic([nr,nc], goal);
                    openSet(nr,nc) = true;
                end
            end
        end
    end
    path = [];
end

%% --- Heuristic (Manhattan) ---
function h = heuristic(p, goal)
    h = abs(p(1)-goal(1)) + abs(p(2)-goal(2));
end
