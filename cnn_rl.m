%% SAC Robot Navigation Training - A* Path Following
% Uses Soft Actor-Critic (SAC) reinforcement learning to train robot navigation

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
    100 10
    % 100 18;
    % 10 30;
    % 10 40;
    % 100 28;
    % 100 40;
    % 10 50;
    % 10 65;
    % 100 55;
    % 100 65;
    % 10 75;
    % 10 85;
    % 100 75;
    % 100 85;
    % 10 95;
    % 10 103;
    % 100 95;
    % 10 size(map,2)-10;
    % size(map,1)-20, size(map,2)-15
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

obs_radius = 0.5;
obs_size   = 10;

dt           = 0.05;
path_follow_distance = 100.0;

%% ========== CREATE RL ENVIRONMENT ==========
fprintf('\nCreating RL environment...\n');

% Define observation info
local_terrain_size = obs_size * obs_size;
numObs = 8 + local_terrain_size;  % sin/cos theta, distances, velocities, terrain

obsInfo = rlNumericSpec([numObs 1]);
obsInfo.Name = 'Robot State';
obsInfo.Description = 'Robot pose, velocities, and local terrain';

% Define action info (4 motor torques)
actInfo = rlNumericSpec([4 1], 'LowerLimit', -10, 'UpperLimit', 10);
actInfo.Name = 'Motor Torques';

% Create environment data structure to pass to functions
envData = struct();
envData.map = map;
envData.res = res;
envData.pathWorld = pathWorld;
envData.start_pos = start_pos;
envData.goal_pos = goal_pos;
envData.obs_radius = obs_radius;
envData.obs_size = obs_size;
envData.dt = dt;
envData.path_follow_distance = path_follow_distance;

% Create custom environment
env = rlFunctionEnv(obsInfo, actInfo, ...
    @(action,loggedSignals) stepFcn(action, loggedSignals, envData), ...
    @() resetFcn(envData));

%% ========== CREATE SAC AGENT ==========
fprintf('Building SAC agent...\n');

% Create critic networks (Q-functions)
criticNetwork1 = buildCriticNetwork(numObs);
criticNetwork2 = buildCriticNetwork(numObs);

critic1 = rlQValueFunction(criticNetwork1, obsInfo, actInfo);
critic2 = rlQValueFunction(criticNetwork2, obsInfo, actInfo);

% Create actor network (policy)
actorNetwork = buildActorNetwork(numObs);
actor = rlContinuousGaussianActor(actorNetwork, obsInfo, actInfo, ...
    'ObservationInputNames', 'observation', ...
    'ActionMeanOutputNames', 'mean_scale', ...
    'ActionStandardDeviationOutputNames', 'std_softplus');

% SAC agent options
agentOpts = rlSACAgentOptions( ...
    'SampleTime', dt, ...
    'DiscountFactor', 0.99, ...
    'ExperienceBufferLength', 1e6, ...
    'MiniBatchSize', 256, ...
    'NumWarmStartSteps', 5000, ...
    'TargetSmoothFactor', 0.005, ...
    'TargetUpdateFrequency', 1);

agentOpts.ActorOptimizerOptions.LearnRate = 3e-4;
agentOpts.ActorOptimizerOptions.GradientThreshold = 1;

agentOpts.CriticOptimizerOptions(1).LearnRate = 3e-4;
agentOpts.CriticOptimizerOptions(1).GradientThreshold = 1;
agentOpts.CriticOptimizerOptions(2).LearnRate = 3e-4;
agentOpts.CriticOptimizerOptions(2).GradientThreshold = 1;


% Create agent
agent = rlSACAgent(actor, [critic1, critic2], agentOpts);

%% ========== TRAINING OPTIONS ==========
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 5000, ...
    'MaxStepsPerEpisode', 500, ...
    'ScoreAveragingWindowLength', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 1000, ...
    'UseParallel', true);

%% ========== TRAIN AGENT ==========
fprintf('\n=== Starting SAC Training ===\n');
trainingStats = train(agent, env, trainOpts);

%% ========== TEST TRAINED AGENT ==========
fprintf('\n=== Testing Trained Agent ===\n');

state = [10; 18.2; 0; 0; 0; 2.2; 0] * res;
trajectory = state(1:2)';
current_path_idx = 0.99;

max_test_steps = 20000;

figure('Position', [100, 100, 1500, 600]);

for step = 1:max_test_steps
    x = state(1);
    y = state(2);
    theta = state(3);

    [current_path_idx, target_point] = findPathTarget( ...
        pathWorld, [x, y], current_path_idx, path_follow_distance);

    local_terrain = extractLocalTerrain(x, y, map, res, obs_radius, obs_size);
    
    dx = target_point(1) - x;
    dy = target_point(2) - y;
    dist_to_target = hypot(dx, dy);
    heading_error  = wrapToPi(atan2(dy, dx) - theta);
    dist_to_goal   = norm(goal_pos - [x; y]);

    terrain_vec = reshape(local_terrain, [], 1);

    obs = [
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

    % Get action from trained agent
    M = getAction(agent, obs);
    M = M{1};  % Extract from cell array
    M = max(min(M(:), 10), -10);

    state = trackedRobotDynamicsWithTerrain(state, M, dt, map, res);
    trajectory = [trajectory; state(1:2)'];

    % --- VISUALIZATION ---
    if mod(step, 10) == 0
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

%% ========== ENVIRONMENT FUNCTIONS ==========

function [nextObs, reward, isDone, loggedSignals] = stepFcn(action, loggedSignals, envData)
    persistent state current_path_idx trajectory step_count
    
    if isempty(state)
        [state, current_path_idx, trajectory, step_count] = resetEnv(envData);
    end
    
    step_count = step_count + 1;
    
    % Apply action
    M = action;
    state = trackedRobotDynamicsWithTerrain(state, M, envData.dt, envData.map, envData.res);
    
    x = state(1); y = state(2); theta = state(3);
    trajectory = [trajectory; x, y];
    
    % Get observation
    [current_path_idx, target_point] = findPathTarget( ...
        envData.pathWorld, [x, y], current_path_idx, envData.path_follow_distance);
    
    local_terrain = extractLocalTerrain(x, y, envData.map, envData.res, ...
                                       envData.obs_radius, envData.obs_size);
    
    dx = target_point(1) - x;
    dy = target_point(2) - y;
    dist_to_target = hypot(dx, dy);
    heading_error  = wrapToPi(atan2(dy, dx) - theta);
    dist_to_goal   = norm(envData.goal_pos - [x; y]);
    
    terrain_vec = reshape(local_terrain, [], 1);
    
    nextObs = [
        sin(theta);
        cos(theta);
        dist_to_target / envData.path_follow_distance;
        sin(heading_error);
        cos(heading_error);
        dist_to_goal / norm(envData.goal_pos - envData.start_pos);
        state(4) / 5;
        state(5) / 5;
        terrain_vec
    ];
    
    % Calculate reward
    reward = calculateReward(state, target_point, envData.goal_pos, local_terrain, ...
                            dist_to_goal, heading_error, dist_to_target);
    
    % Check termination
    isDone = false;
    if dist_to_goal < 0.1
        reward = reward + 1000;  % Big bonus for reaching goal
        isDone = true;
    elseif step_count >= 5000
        isDone = true;
    elseif norm(state(1:2)) < 0.1 || any(isnan(state)) || any(isinf(state))
        reward = reward - 100;  % Penalty for bad state
        isDone = true;
    end
    
    % Check if stuck
    if size(trajectory, 1) > 100
        if norm(trajectory(end,:) - trajectory(end-50,:)) < 0.3
            reward = reward - 50;
            isDone = true;
        end
    end
    
    loggedSignals = [];
    
    if isDone
        state = [];
    end
end

function [initObs, loggedSignals] = resetFcn(envData)
    [state, current_path_idx, ~, ~] = resetEnv(envData);
    
    x = state(1); y = state(2); theta = state(3);
    
    [~, target_point] = findPathTarget( ...
        envData.pathWorld, [x, y], current_path_idx, envData.path_follow_distance);
    
    local_terrain = extractLocalTerrain(x, y, envData.map, envData.res, ...
                                       envData.obs_radius, envData.obs_size);
    
    dx = target_point(1) - x;
    dy = target_point(2) - y;
    dist_to_target = hypot(dx, dy);
    heading_error  = wrapToPi(atan2(dy, dx) - theta);
    dist_to_goal   = norm(envData.goal_pos - [x; y]);
    
    terrain_vec = reshape(local_terrain, [], 1);
    
    initObs = [
        sin(theta);
        cos(theta);
        dist_to_target / envData.path_follow_distance;
        sin(heading_error);
        cos(heading_error);
        dist_to_goal / norm(envData.goal_pos - envData.start_pos);
        state(4) / 5;
        state(5) / 5;
        terrain_vec
    ];
    
    loggedSignals = [];
end

function [state, current_path_idx, trajectory, step_count] = resetEnv(envData)
    start_idx = randi([1, min(10, size(envData.pathWorld,1))]);
    init_pos  = envData.pathWorld(start_idx,:)';
    init_theta = randn * 0.2;
    
    state = [init_pos; init_theta; 0; 0; 0.22; 0];
    current_path_idx = start_idx;
    trajectory = init_pos';
    step_count = 0;
end

%% ========== HELPER FUNCTIONS ==========

function reward = calculateReward(state, target_point, goal_pos, terrain, ...
                                 dist_to_goal, heading_error, dist_to_target)
    x = state(1); y = state(2);
    
    % Progress toward goal
    reward = -0.1 * dist_to_goal;
    
    % Path following
    reward = reward - 0.05 * dist_to_target;
    
    % Heading alignment
    reward = reward - 0.1 * abs(heading_error);
    
    % % Collision penalty
    % center = ceil(size(terrain,1)/2);
    % if terrain(center, center) > 0.9
    %     reward = reward - 10;
    % end
    % 
    % % Obstacle proximity penalty
    % terrain_ahead = terrain(center-1:center+1, center+1:end);
    % if mean(terrain_ahead(:)) > 0.7
    %     reward = reward - 2;
    % end
    % 
    % Small time penalty
    reward = reward - 0.01;
end

function net = buildActorNetwork(numObs)

    input = featureInputLayer(numObs, 'Name', 'observation');

    common = [
        fullyConnectedLayer(256, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(256, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
    ];

    meanPath = [
        fullyConnectedLayer(4, 'Name', 'mean_fc')
        tanhLayer('Name', 'mean_tanh')
        scalingLayer('Scale', 10, 'Name', 'mean_scale')
    ];

    stdPath = [
        fullyConnectedLayer(4, 'Name', 'std_fc')
        softplusLayer('Name', 'std_softplus')
    ];

    net = layerGraph(input);
    net = addLayers(net, common);
    net = addLayers(net, meanPath);
    net = addLayers(net, stdPath);

    net = connectLayers(net, 'observation', 'fc1');
    net = connectLayers(net, 'relu2', 'mean_fc');
    net = connectLayers(net, 'relu2', 'std_fc');
end


function net = buildCriticNetwork(numObs)

    statePath = [
        featureInputLayer(numObs, 'Name', 'state')
        fullyConnectedLayer(256, 'Name', 'state_fc1')
        reluLayer('Name', 'state_relu1')
    ];

    actionPath = [
        featureInputLayer(4, 'Name', 'action')
        fullyConnectedLayer(256, 'Name', 'action_fc1')
        reluLayer('Name', 'action_relu1')
    ];

    commonPath = [
        additionLayer(2, 'Name', 'add')
        fullyConnectedLayer(256, 'Name', 'fc_common')
        reluLayer('Name', 'relu_common')
        fullyConnectedLayer(1, 'Name', 'output')
    ];

    net = layerGraph(statePath);
    net = addLayers(net, actionPath);
    net = addLayers(net, commonPath);

    net = connectLayers(net, 'state_relu1', 'add/in1');
    net = connectLayers(net, 'action_relu1', 'add/in2');
end


function [idx, target] = findPathTarget(path, pos, current_idx, lookahead)
    dists = sqrt(sum((path - pos).^2, 2));
    [~, closest] = min(dists);
    
    idx = max(current_idx, closest);
    
    for i = idx:size(path,1)
        if norm(path(i,:) - pos) >= lookahead
            target = path(i,:);
            idx = i;
            return;
        end
    end
    
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