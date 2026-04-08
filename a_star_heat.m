clc; clear; close all;

%% Load map
% map = druhy('image.jpg', [250 370; 220 550; 450 550; 450 350]);

%% Define waypoints
% waypoints = [40 45; 200 30; 40 65; 200 65; 40 90; 200 85; 40 120; 200 120; 40 150; 200 150; 40 size(map,2)-10;
%              size(map,1)-10, size(map,2)-10]; % goal
             
%% Load map
% coords = [280 400; 280 520; 
%          400 520; 400 400];
% map = druhy('image.jpg', coords);
% 
% nmap = map; 
% nmap(nmap ~= 1) = 0;

%% Define waypoints
% waypoints = [10 20;
%             100 10]; % goal

%% new waypoints
waypoints = [15 13; 102 12; 97 28; 58 32; 42 45; 45 57; 102 60; 100 78; 77 78; 34 77];
waypoints2 = [87 10; 85 88; 75 90; 76 26; 64 26; 60 95];
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
imshow(map, []); 
% map(map ~= 1) = 0;
% imshow(map, []);
hold on;
plot(fullPath(:,2), fullPath(:,1), 'g-', 'LineWidth', 2);
plot(waypoints(:,2), waypoints(:,1), 'r*', 'MarkerSize', 18, 'LineWidth', 2);
title('A* Path Considering Height Difficulty');


ax = figure(1);
drawnow; pause(0.01);
frame = getframe(ax);
im = frame2im(frame);
imwrite(im, 'astar_test1.png');

%% ========== A* METRICS CALCULATION ==========
fprintf('\n=== Computing Metrics for A* Path ===\n');

% Preallocate metric arrays
num_steps = size(fullPath, 1);
metrics_steps = 1:num_steps;
metrics_safety = zeros(1, num_steps);
metrics_hitcount = zeros(1, num_steps);
metrics_distance = zeros(1, num_steps);
metrics_speed = zeros(1, num_steps);
metrics_smooth = zeros(1, num_steps);

% Define local window size for safety calculation (e.g., 5x5 grid)
window_radius = 2; 
final_goal = waypoints(end, :);

for step = 1:num_steps
    % 1. Current Position
    r = fullPath(step, 1); % Row (Y)
    c = fullPath(step, 2); % Col (X)
    
    % 2. Safety & Hitcount (Extract local terrain window)
    r_min = max(1, r - window_radius);
    r_max = min(size(map, 1), r + window_radius);
    c_min = max(1, c - window_radius);
    c_max = min(size(map, 2), c + window_radius);
    
    local_window = map(r_min:r_max, c_min:c_max);
    
    % Assuming '1' or high values represent obstacles/elevation
    hitbox = (local_window >= 0.8); % Adjust threshold if map is not purely 0 and 1
    hit_count = sum(hitbox(:));
    
    metrics_safety(step) = hit_count / numel(hitbox); 
    metrics_hitcount(step) = hit_count;
    
    % 3. Distance to final goal
    dist_to_goal = hypot(final_goal(2) - c, final_goal(1) - r);
    metrics_distance(step) = dist_to_goal;
    
    % 4. Smoothness (Change in heading angle) & Speed
    if step > 2
        % Vector of previous step and current step
        v_prev = fullPath(step-1, :) - fullPath(step-2, :);
        v_curr = fullPath(step, :) - fullPath(step-1, :);
        
        % Calculate angles (Heading)
        theta_prev = atan2(v_prev(1), v_prev(2));
        theta_curr = atan2(v_curr(1), v_curr(2));
        
        % Direction change (Smoothness)
        % wrapToPi ensures that a turn from 179 deg to -179 deg is 2 deg, not 358.
        metrics_smooth(step) = abs(wrapToPi(theta_curr - theta_prev));
        
        % Speed calculation
        metrics_speed(step) = norm(v_curr);
    elseif step == 2
        v_curr = fullPath(step, :) - fullPath(step-1, :);
        metrics_speed(step) = norm(v_curr);
        metrics_smooth(step) = 0; % No change yet
    else
        metrics_speed(step) = 0;
        metrics_smooth(step) = 0;
    end
end

%% ========== FINAL PERFORMANCE SUMMARY ==========
fprintf('Plotting Performance Summaries...\n');

% Figure 2: Direction Change (Heading Jitter)
figure(2);
plot(metrics_steps, smoothdata(rad2deg(metrics_smooth), 'movmean', 10), 'r', 'LineWidth', 1);
grid on;
title('Path Smoothness (Heading Change)');
xlabel('Path Step'); 
ylabel('Change in Direction (deg)');

% Figure 3: Cumulative Safety
figure(3);
plot(metrics_steps, metrics_safety * 100, 'b', 'LineWidth', 1.5);
average_safety = sum(metrics_safety) / length(metrics_safety);
hold on;
plot([1, num_steps], [average_safety, average_safety] * 100, 'r--', 'LineWidth', 1);
hold off;
title('Safety Metric Along Path');
xlabel('Path Step'); ylabel('Obstacle Density in Local Window (%)');
legend('Current Safety', sprintf('Average: %.1f%%', average_safety*100));
grid on;


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
