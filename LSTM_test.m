%% LSTM Robot Navigation Training - A* Path Following
% Uses A* to generate optimal paths, then trains LSTM to follow them

clear; clc; close all;
addpath("kinematika_MR");

%% ========== LOAD MAP AND GENERATE PATH ==========
coords = [280 400; 280 520; 400 520; 400 400];
map = druhy('image.jpg', coords);
res = 0.1;  % map resolution [m/cell]

fprintf('Map size: %d x %d\n', size(map,1), size(map,2));
fprintf('Map range: [%.3f, %.3f]\n', min(map(:)), max(map(:)));

% Define start and goal in MAP COORDINATES (pixels)
start_map = [10 20];  % [row, col] - corresponds to ~[10, 20] in world
goal_map = [100 8];  % [row, col] - corresponds to ~[8, 100] in world

% Generate optimal path using A*
fprintf('\nGenerating A* path...\n');
optimalPath = astar_height(map, start_map, goal_map);

if isempty(optimalPath)
    error('No valid path found! Check start/goal positions and map.');
end

fprintf('Path length: %d waypoints\n', size(optimalPath, 1));

% Convert path to world coordinates
pathWorld = [(optimalPath(:,1) - 1) * res, (optimalPath(:,2) - 1) * res];
% pathWorld = [pathWorld(:,2) pathWorld(:,1)];

% Visualization
figure('Position', [100, 100, 1200, 500]);
subplot(1,2,1);
imshow(map, []); colormap('turbo'); colorbar;
hold on;
plot(optimalPath(:,2), optimalPath(:,1), 'w-', 'LineWidth', 3);
plot(start_map(2), start_map(1), 'go', 'MarkerSize', 15, 'MarkerFaceColor', 'g');
plot(goal_map(2), goal_map(1), 'r*', 'MarkerSize', 20, 'LineWidth', 2);
title('A* Optimal Path');
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
goal_pos = pathWorld(end,:)';

% Local observation
obs_radius = 1;
obs_size = 10;

% Training parameters
num_episodes = 150;
max_steps = 20000;
dt = 0.01;
path_follow_distance = 100.0;  % Look-ahead distance for path following

fprintf('\nStart: [%.2f, %.2f]\n', start_pos(1), start_pos(2));
fprintf('Goal: [%.2f, %.2f]\n', goal_pos(1), goal_pos(2));

%% ========== DATA COLLECTION ==========
fprintf('\nCollecting training data...\n');

all_sequences = {};
all_targets = {};
success_count = 0;

for ep = 1:num_episodes
    % Random starting position along early part of path
    start_idx = max(1, min(10, size(pathWorld,1)));
    start_idx = randi([1, start_idx]);
    
    init_pos = pathWorld(start_idx, :)';
    init_theta = randn * 0.2;
    state = [init_pos; init_theta; 0; 0; 0.22; 0];
    
    sequence = [];
    targets = [];
    current_path_idx = start_idx;
    
    try
        for step = 1:max_steps
            x = state(1);
            y = state(2);
            theta = state(3);
            
            % Find nearest point on path and look-ahead target
            [current_path_idx, target_point] = findPathTarget(pathWorld, [x, y], ...
                current_path_idx, path_follow_distance);
            
            % Extract local terrain
            local_terrain = extractLocalTerrain(x, y, map, res, obs_radius, obs_size);
            
            % Calculate features relative to path target
            dx = target_point(1) - x;
            dy = target_point(2) - y;
            dist_to_target = sqrt(dx^2 + dy^2);
            angle_to_target = atan2(dy, dx);
            heading_error = wrapToPi(angle_to_target - theta);
            
            % Distance to final goal
            dist_to_goal = norm(goal_pos - [x; y]);
            
            % Features: [dist_to_target, heading_error, dist_to_goal, v, omega, terrain]
            terrain_vec = reshape(local_terrain, [], 1);
            features = [theta; dist_to_target; heading_error; dist_to_goal; 
                        state(4); state(5); terrain_vec];

            
            % Path-following controller with exploration
            exploitation_rate = 0.8 + 0.2*(ep/num_episodes);
            
            if rand < exploitation_rate
                M = pathFollowingController(state, target_point, local_terrain);
            else
                M = randn(4,1) * 2.5 + 1.5;  % Exploration
            end
            
            M = max(min(M, 10), -10);
            
            % Store data
            sequence = [sequence; features'];
            targets = [targets; M'];
            
            % Simulate
            state_new = trackedRobotDynamicsWithTerrain(state, M, dt, map, res);
            
            % Check if stuck
            if norm(state_new(1:2) - state(1:2)) < 0.001
                break;
            end
            
            state = state_new;
            
            % Success if reached goal
            if dist_to_goal < 0.1
                fprintf('Episode %d: Reached goal in %d steps!\n', ep, step);
                success_count = success_count + 1;
                break;
            end
            
            % Also succeed if made good progress along path
            if current_path_idx >= size(pathWorld,1) - 10
                fprintf('Episode %d: Reached end of path!\n', ep);
                success_count = success_count + 1;
                break;
            end
        end
    catch ME
        fprintf('Episode %d error: %s\n', ep, ME.message);
        continue;
    end
    
    % Accept if made reasonable progress (moved forward on path)
    progress = current_path_idx - start_idx;
    if progress > 20 || dist_to_goal < 0.1
        all_sequences{end+1} = sequence;
        all_targets{end+1} = targets;
    end
    
    if mod(ep, 25) == 0
        fprintf('Episodes: %d/%d | Sequences: %d | Success: %d\n', ...
            ep, num_episodes, length(all_sequences), success_count);
    end
end

fprintf('\n=== Collection Summary ===\n');
fprintf('Sequences: %d | Success: %d\n', length(all_sequences), success_count);

if isempty(all_sequences)
    error('No training data collected!');
end

%% ========== PREPARE TRAINING DATA ==========
X_train = cell(length(all_sequences), 1);
Y_train = cell(length(all_sequences), 1);

for i = 1:length(all_sequences)
    X_train{i} = all_sequences{i}';
    Y_train{i} = all_targets{i}';
end

input_size = size(X_train{1}, 1);
fprintf('Input features: %d | Training sequences: %d\n', input_size, length(X_train));

%% ========== BUILD & TRAIN LSTM ==========
fprintf('\nBuilding LSTM...\n');

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
    'MaxEpochs', 100, ...
    'MiniBatchSize', 8, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 12, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

fprintf('Training...\n');
net = trainNetwork(X_train, Y_train, layers, options);

%% ========== TEST ==========
fprintf('\n=== Testing ===\n');

state = [10; 18.2; 0; 0; 0; 2.2; 0]*res;
trajectory = state(1:2)';
current_path_idx = 0.99;

% Initialize sequence buffer for LSTM
sequence_buffer = [];
buffer_length = 100;  % Can increase for temporal context

figure('Position', [100, 100, 1500, 600]);

for step = 1:max_steps
    x = state(1);
    y = state(2);
    theta = state(3);
    
    [current_path_idx, target_point] = findPathTarget(pathWorld, [x, y], ...
        current_path_idx, path_follow_distance);
    
    local_terrain = extractLocalTerrain(x, y, map, res, obs_radius, obs_size);
    
    dx = target_point(1) - x;
    dy = target_point(2) - y;
    dist_to_target = sqrt(dx^2 + dy^2);
    angle_to_target = atan2(dy, dx);
    heading_error = wrapToPi(angle_to_target - theta);
    dist_to_goal = norm(goal_pos - [x; y]);
    
    terrain_vec = reshape(local_terrain, [], 1);
    features = [theta; dist_to_target; heading_error; dist_to_goal; 
            state(4); state(5); terrain_vec];

    % Add to sequence buffer
    sequence_buffer = [sequence_buffer; features'];
    if size(sequence_buffer, 1) > buffer_length
        sequence_buffer = sequence_buffer(end-buffer_length+1:end, :);
    end
    
    % LSTM prediction - input format: [features x timesteps]
    M_pred = predict(net, sequence_buffer');
    
    % Extract motor commands from last timestep
    if size(M_pred, 2) > 1
        M = M_pred(:, end);  % Last timestep
    else
        M = M_pred;
    end
    M = max(min(M(:), 10), -10);

    
    state = trackedRobotDynamicsWithTerrain(state, M, dt, map, res);
    trajectory = [trajectory; state(1:2)'];
    
    if mod(step, 5) == 0
        subplot(1,3,1);
        imagesc(map); colormap gray; hold on;
        plot(pathWorld(:,2)/res, pathWorld(:,1)/res, 'c-', 'LineWidth', 2);
        plot(trajectory(:,2)/res, trajectory(:,1)/res, 'g-', 'LineWidth', 2);
        plot(start_pos(2)/res, start_pos(1)/res, 'go', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
        plot(goal_pos(2)/res, goal_pos(1)/res, 'r*', 'MarkerSize', 18, 'LineWidth', 2);
        plot(y/res, x/res, 'yo', 'MarkerSize', 10, 'MarkerFaceColor', 'y');
        plot(target_point(2)/res, target_point(1)/res, 'mx', 'MarkerSize', 12, 'LineWidth', 3);
        legend('A* Path', 'Robot Path', 'Start', 'Goal', 'Robot', 'Target');
        title(sprintf('Step %d | Goal: %.1fm', step, dist_to_goal));
        axis equal; axis tight; hold off;
        
        subplot(1,3,2);
        bar(M); ylim([-12, 12]); grid on;
        title('Motor Torques'); ylabel('Nm');
        xticklabels({'FL', 'FR', 'RL', 'RR'});
        
        subplot(1,3,3);
        imagesc(local_terrain); colormap(gca, 'gray'); colorbar;
        title('Local View'); axis equal; axis tight;
        
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