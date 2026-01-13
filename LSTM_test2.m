%% LSTM Robot Navigation Training - A* Path Following
% Uses A* to generate optimal paths between WAYPOINTS, then trains LSTM to follow them

clear; clc; close all;
addpath("kinematika_MR");

%% ========== LOAD MAP ==========
coords = [280 400; 280 520; 400 520; 400 400];
map = druhy('image.jpg', coords);
res = 0.1;  % map resolution [m/cell]

fprintf('Map size: %d x %d\n', size(map,1), size(map,2));
fprintf('Map range: [%.3f, %.3f]\n', min(map(:)), max(map(:)));

%% ========== WAYPOINTS ==========
waypoints = [ ...
    10 20;
    100 10;
    100 18;
    10 30;
    10 40;
    100 28;
    100 40;
    10 50;
    10 65;
    100 55;
    100 65;
    10 75;
    10 85;
    100 75;
    100 85;
    10 95;
    10 103;
    100 95;
    10 size(map,2)-10;
    size(map,1)-20, size(map,2)-15
];

%% ========== GENERATE MULTI-WAYPOINT A* PATH ==========
fprintf('\nGenerating multi-waypoint A* path...\n');

optimalPath = [];

for i = 1:size(waypoints,1)-1
    start_wp = waypoints(i,:);
    goal_wp  = waypoints(i+1,:);
    
    segmentPath = astar_height(map, start_wp, goal_wp);
    
    if isempty(segmentPath)
        error('No valid path between waypoint %d and %d', i, i+1);
    end
    
    % Avoid duplicate junction points
    if i > 1
        segmentPath = segmentPath(2:end,:);
    end
    
    optimalPath = [optimalPath; segmentPath];
end

fprintf('Total path length: %d waypoints\n', size(optimalPath,1));

%% ========== CONVERT TO WORLD COORDINATES ==========
pathWorld = [(optimalPath(:,1) - 1) * res, ...
             (optimalPath(:,2) - 1) * res];

%% ========== VISUALIZATION ==========
figure('Position', [100, 100, 1200, 500]);

subplot(1,2,1);
imshow(map, []); colormap('turbo'); colorbar;
hold on;
plot(optimalPath(:,2), optimalPath(:,1), 'w-', 'LineWidth', 3);
plot(waypoints(:,2), waypoints(:,1), 'ro', ...
     'MarkerSize', 8, 'MarkerFaceColor', 'r');
title('A* Path Through Waypoints');
hold off;

subplot(1,2,2);
plot(pathWorld(:,2), pathWorld(:,1), 'b-', 'LineWidth', 2);
hold on;
plot(pathWorld(1,2), pathWorld(1,1), 'go', 'MarkerSize', 15, 'MarkerFaceColor', 'g');
plot(pathWorld(end,2), pathWorld(end,1), 'r*', 'MarkerSize', 20, 'LineWidth', 2);
grid on; axis equal;
title('Path in World Coordinates');
xlabel('Y [m]'); ylabel('X [m]');
hold off;
drawnow;

%% ========== CONFIGURATION ==========
start_pos = pathWorld(1,:)';
goal_pos  = pathWorld(end,:)';

obs_radius = 1;
obs_size   = 10;

num_episodes = 100;
max_steps    = 20000;
dt           = 0.01;
path_follow_distance = 100.0;

%% ========== DATA COLLECTION ==========
fprintf('\nCollecting training data...\n');

all_sequences = {};
all_targets   = {};
success_count = 0;

for ep = 1:num_episodes
    
    start_idx = randi([1, min(10, size(pathWorld,1))]);
    init_pos  = pathWorld(start_idx,:)';
    init_theta = randn * 0.2;
    
    state = [init_pos; init_theta; 0; 0; 0.22; 0];
    sequence = [];
    targets  = [];
    current_path_idx = start_idx;

    for step = 1:max_steps
        x = state(1); y = state(2); theta = state(3);

        [current_path_idx, target_point] = findPathTarget( ...
            pathWorld, [x y], current_path_idx, path_follow_distance);

        local_terrain = extractLocalTerrain(x, y, map, res, obs_radius, obs_size);

        dx = target_point(1) - x;
        dy = target_point(2) - y;
        dist_to_target = hypot(dx, dy);
        heading_error  = wrapToPi(atan2(dy, dx) - theta);
        dist_to_goal   = norm(goal_pos - [x; y]);

        terrain_vec = reshape(local_terrain, [], 1);

        features = [
            sin(theta);
            cos(theta);
            dist_to_target / path_follow_distance;
            sin(heading_error);
            cos(heading_error);
            dist_to_goal / norm(goal_pos - start_pos);
            state(4) / 5;
            state(5) / 5;
            terrain_vec
            ];

        exploitation_rate = 0.8 + 0.2*(ep/num_episodes);

        if rand < exploitation_rate
            M = pathFollowingController(state, target_point, local_terrain);
        else
            M = randn(4,1)*2.5 + 1.5;
        end

        M = max(min(M,10), -10);

        sequence = [sequence; features'];
        targets  = [targets; M'];

        state_new = trackedRobotDynamicsWithTerrain(state, M, dt, map, res);

        if norm(state_new(1:2) - state(1:2)) < 1e-3
            break;
        end

        state = state_new;

        if dist_to_goal < 0.1 || current_path_idx >= size(pathWorld,1)-10
            success_count = success_count + 1;
            break;
        end
    end

    if current_path_idx - start_idx > 20 || dist_to_goal < 0.1
        all_sequences{end+1} = sequence;
        all_targets{end+1}   = targets;
    end
end

fprintf('Collected %d sequences | Success: %d\n', ...
    numel(all_sequences), success_count);

%% ========== PREPARE TRAINING DATA ==========
X_train = cellfun(@(x)x', all_sequences, 'UniformOutput', false);
Y_train = cellfun(@(y)y', all_targets,   'UniformOutput', false);

input_size = size(X_train{1},1);

%% ========== BUILD & TRAIN LSTM ==========

% layers = [
%     sequenceInputLayer(input_size)
% 
%     lstmLayer(512, 'OutputMode', 'sequence')  % Increased from 256
%     batchNormalizationLayer()
%     dropoutLayer(0.2)
% 
%     lstmLayer(256, 'OutputMode', 'sequence')  % Increased from 128
%     batchNormalizationLayer()
%     dropoutLayer(0.3)
% 
%     lstmLayer(128, 'OutputMode', 'sequence')  % Increased from 64
%     dropoutLayer(0.3)
% 
%     lstmLayer(64, 'OutputMode', 'sequence')   % Add 4th LSTM layer
% 
%     fullyConnectedLayer(32)                    % Add intermediate FC layer
%     reluLayer()
%     fullyConnectedLayer(4)
%     regressionLayer
% ];

layers = buildLSTM(input_size);

options = trainingOptions('adam', ...
    'MaxEpochs', 1000, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(X_train, Y_train, layers, options);

%% ========== TEST ==========
fprintf('\n=== Testing ===\n');

state = [10; 18.2; 0; 0; 0; 2.2; 0] * res;
trajectory = state(1:2)';
current_path_idx = 0.99;

figure('Position', [100, 100, 1500, 600]);

for step = 1:max_steps
    x = state(1);
    y = state(2);
    theta = state(3);

    [current_path_idx, target_point] = findPathTarget( ...
        pathWorld, [x, y], current_path_idx, path_follow_distance);

    local_terrain = extractLocalTerrain(x, y, map, res, obs_radius, obs_size);

    dist_to_goal = norm(goal_pos - [x; y]);

    M = pathFollowingController(state, target_point, local_terrain);
    M = max(min(M(:), 10), -10);

    state = trackedRobotDynamicsWithTerrain(state, M, dt, map, res);
    trajectory = [trajectory; state(1:2)'];

    % --- VISUALIZATION ---
    if mod(step, 1) == 0
        subplot(1,3,1);
        imagesc(map); colormap gray; hold on;
        plot(pathWorld(:,2)/res, pathWorld(:,1)/res, 'c-', 'LineWidth', 2);
        plot(trajectory(:,2)/res, trajectory(:,1)/res, 'g-', 'LineWidth', 2);
        plot(start_pos(2)/res, start_pos(1)/res, 'go', ...
            'MarkerSize', 12, 'MarkerFaceColor', 'g');
        plot(goal_pos(2)/res, goal_pos(1)/res, 'r*', ...
            'MarkerSize', 18, 'LineWidth', 2);
        plot(y/res, x/res, 'yo', ...
            'MarkerSize', 10, 'MarkerFaceColor', 'y');
        plot(target_point(2)/res, target_point(1)/res, 'mx', ...
            'MarkerSize', 12, 'LineWidth', 3);
        title(sprintf('Step %d | Goal: %.2fm', step, dist_to_goal));
        axis equal; axis tight; hold off;

        subplot(1,3,2);
        bar(M); ylim([-12, 12]); grid on;
        title('Motor Torques'); ylabel('Nm');
        xticklabels({'FL','FR','RL','RR'});

        subplot(1,3,3);
        imagesc(local_terrain, [min(map(:)) max(map(:))]);
        colormap(gca, 'gray'); colorbar;
        title('Local View');
        axis equal; axis tight;

        drawnow;
    end

    if dist_to_goal < 0.1
        fprintf('SUCCESS! Goal reached in %d steps (%.2fm)\n', step, dist_to_goal);
        break;
    end

    if step > 200 && norm(trajectory(end,:) - trajectory(end-50,:)) < 0.5
        fprintf('Stuck at step %d (%.2fm from goal)\n', step, dist_to_goal);
        break;
    end
end



%% ========== HELPER FUNCTIONS ==========

function [idx, target] = findPathTarget(path, pos, current_idx, lookahead)
    % Find closest point on path
    dists = sqrt(sum((path - pos).^2, 2));
    [~, closest] = min(dists);
    
    % Start from closest point
    idx = max(current_idx, closest);
    
    % Look ahead along path
    for i = idx:size(path,1)
        if norm(path(i,:) - pos) >= lookahead
            target = path(i,:);
            idx = i;
            return;
        end
    end
    
    % If near end, target is goal
    target = path(end,:);
    idx = size(path,1);
end

function local_map = extractLocalTerrain(y,x, map, res, radius, grid_size)
    half = floor(grid_size/2);
    local_map = zeros(grid_size, grid_size);
    
    for i = 1:grid_size
        for j = 1:grid_size
            wx = x + (j - half - 1) * radius / half;
            wy = y + (i - half - 1) * radius / half;
            
            ix = round(wx/res) + 1;
            iy = round(wy/res) + 1;
            
            if ix >= 1 && ix <= size(map,2) && iy >= 1 && iy <= size(map,1)
                local_map(i,j) = map(iy, ix);
            else
                local_map(i,j) = 1.0;
            end
        end
    end
end

function M = pathFollowingController(state, target, terrain)
    x = state(1); y = state(2); theta = state(3); v = state(4);
    
    dx = target(1) - x;
    dy = target(2) - y;
    angle_to_target = atan2(dy, dx);
    heading_error = wrapToPi(angle_to_target - theta);
    dist = sqrt(dx^2 + dy^2);
    
    % Check terrain
    center = ceil(size(terrain,1)/2);
    terrain_ahead = terrain(center-1:center+1, center+1:end);
    obstacle_ahead = mean(terrain_ahead(:)) > 0.7;
    
    if obstacle_ahead
        % Emergency avoidance
        terrain_left = mean(terrain(1:center-1, center-1:center+1), 'all');
        terrain_right = mean(terrain(center+2:end, center-1:center+1), 'all');
        
        if terrain_right < terrain_left
            M = [-4; 7; -4; 7];  % Turn right
        else
            M = [7; -4; 7; -4];  % Turn left
        end
    else
        % Pure pursuit control
        if abs(heading_error) > deg2rad(20)
            % Large heading error - turn in place
            turn_strength = 5.0;
            M_turn = turn_strength * sign(heading_error);
            M = [1 - M_turn; 1 + M_turn; 1 - M_turn; 1 + M_turn];
        else
            % Small error - smooth following
            base = 6.0;
            K_turn = 3.0;
            M_turn = K_turn * heading_error;
            M = [base - M_turn; base + M_turn; base - M_turn; base + M_turn];
            
            % Slow near target
            if dist < 5.0
                M = M * (0.3 + 0.7 * dist/5.0);
            end
        end
    end
end

%% --- A* function ---
function path = astar_height(map, start, goal)
    [rows, cols] = size(map);
    if map(start(1), start(2)) == 1 || map(goal(1), goal(2)) == 1
        path = []; return;
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

function h = heuristic(p, goal)
    h = abs(p(1)-goal(1)) + abs(p(2)-goal(2));
end