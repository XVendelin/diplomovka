%% SAC Robot Navigation Training - A* Path Following
% Uses Soft Actor-Critic (SAC) reinforcement learning to train robot navigation

clear; clc; close all;
addpath("kinematika_MR");

%% ========== LOAD MAP ==========
coords = [280 400; 280 520; 400 520; 400 400];
map = druhy('image.jpg', coords);
res = 0.1;  % map resolution [m/cell]

%% ========== CONFIGURATION ==========
start_pos = [10; 18] * res;
goal_pos  = [100; 10] * res;

obs_size   = 20;
obs_radius = obs_size*res/2;

dt           = 0.05;

%% ========== CREATE RL ENVIRONMENT ==========
fprintf('\nCreating RL environment...\n');

% Define observation info
local_terrain_size = obs_size * obs_size;
numObs = 7 + local_terrain_size;  % sin/cos theta, distances, velocities, terrain

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
envData.start_pos = start_pos;
envData.goal_pos = goal_pos;
envData.obs_radius = obs_radius;
envData.obs_size = obs_size;
envData.dt = dt;

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
agentOpts = rlSACAgentOptions(...
    'SampleTime', dt, ...
    'DiscountFactor', 0.99, ...
    'ExperienceBufferLength', 1e6, ...
    'MiniBatchSize', 256*2, ...
    'NumWarmStartSteps', 5000, ...
    'SequenceLength', 10, ...      
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
addpath("kinematika_MR");
fprintf('\n=== Testing Trained Agent ===\n');
test_start = [10; 18] * res;
test_goal = [100; 10] * res;

state = [test_start; 0; 0; 0; 0; 0];
trajectory = state(1:2)';

max_test_steps = 20000;

figure('Position', [100, 100, 1500, 600]);

for step = 1:max_test_steps
    x = state(1);
    y = state(2);
    theta = state(3);


    local_terrain = extractLocalTerrain(x, y, map, res, obs_radius, obs_size);

    terrain_vec = reshape(local_terrain, [], 1);

    goal_dx = test_goal(1) - x;
    goal_dy = test_goal(2) - y;
    goal_heading_error = wrapToPi(atan2(goal_dy, goal_dx) - theta);
    dist_to_goal = hypot(goal_dx, goal_dy);


    obs = [
        sin(theta);
        cos(theta);
        sin(goal_heading_error);
        cos(goal_heading_error);
        dist_to_goal / norm(test_goal - test_start);
        state(4) / 5;
        state(5) / 5;
        terrain_vec
        ];


    % Get action from trained agent
    M = getAction(agent, obs);
    M = M{1};  % Extract from cell array
    M = max(min(M(:), 10), -10);

    state = trackedRobotDynamics(state, M, dt);
    trajectory = [trajectory; state(1:2)'];

    % --- VISUALIZATION ---
    if mod(step, 1) == 0
        subplot(1,3,1);
        imagesc(map); colormap gray; hold on;
        plot(trajectory(:,2)/res, trajectory(:,1)/res, 'g-', 'LineWidth', 2);
        plot(test_start(2)/res, test_start(1)/res, 'go', ...
            'MarkerSize', 8, 'MarkerFaceColor', 'g');
        plot(test_goal(2)/res, test_goal(1)/res, 'r*', ...
            'MarkerSize', 18, 'LineWidth', 2);
        drawTrackedRobot(x, y, theta, 0.25*2, 0.25, res);

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
    persistent state trajectory step_count
    
    if isempty(state)
        [state, trajectory, step_count] = resetEnv(envData);
    end
    
    step_count = step_count + 1;
    
    % Apply action
    M = action;
    state = trackedRobotDynamics(state, M, envData.dt);
    
    x = state(1); y = state(2); theta = state(3);
    trajectory = [trajectory; x, y];
    
    % Get observation
    
    local_terrain = extractLocalTerrain(x, y, envData.map, envData.res, ...
                                       envData.obs_radius, envData.obs_size);
    
    goal_dx = envData.goal_pos(1) - x;
    goal_dy = envData.goal_pos(2) - y;
    goal_heading_error = wrapToPi(atan2(goal_dy, goal_dx) - theta);
    dist_to_goal   = norm(envData.goal_pos - [x; y]);
    
    terrain_vec = reshape(local_terrain, [], 1);
    
    nextObs = [
        sin(theta);
        cos(theta);
        sin(goal_heading_error);
        cos(goal_heading_error);
        dist_to_goal / norm(envData.goal_pos - envData.start_pos);
        state(4) / 5;
        state(5) / 5;
        terrain_vec
    ];
    
    % Calculate reward
    reward = calculateReward(state, envData.goal_pos, local_terrain, ...
                         dist_to_goal, goal_heading_error);
    
    % Check termination + goal reward
    isDone = false;
    if dist_to_goal < 0.1
        reward = reward + 1000;  % Big bonus for reaching goal
        isDone = true;
    elseif step_count >= 5000
        isDone = true;
    elseif any(isnan(state)) || any(isinf(state)) % Non real number penalty
        reward = reward - 100;
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
    [state, ~, ~] = resetEnv(envData);

    x     = state(1);
    y     = state(2);
    theta = state(3);

    % Local terrain observation
    local_terrain = extractLocalTerrain( ...
        x, y, envData.map, envData.res, ...
        envData.obs_radius, envData.obs_size);

    terrain_vec = reshape(local_terrain, [], 1);

    % Goal-related quantities
    goal_dx = envData.goal_pos(1) - x;
    goal_dy = envData.goal_pos(2) - y;
    dist_to_goal = hypot(goal_dx, goal_dy);

    goal_heading_error = wrapToPi(atan2(goal_dy, goal_dx) - theta);

    % Initial observation
    initObs = [
        sin(theta);
        cos(theta);
        sin(goal_heading_error);
        cos(goal_heading_error);
        dist_to_goal / norm(envData.goal_pos - envData.start_pos);
        state(4) / 5;
        state(5) / 5;
        terrain_vec
    ];

    loggedSignals = [];
end


function [state, trajectory, step_count] = resetEnv(envData)

    init_pos = envData.start_pos;
    init_theta = randn * 0.2;

    state = [init_pos; init_theta; 0; 0; 0.22; 0];
    trajectory = init_pos';
    step_count = 0;
end


%% ========== HELPER FUNCTIONS ==========

function reward = calculateReward(state, goal_pos, terrain, ...
                                 dist_to_goal, heading_error)
    res = 0.1;
    sizex= ceil(0.5/res/2);
    sizey= ceil(0.5/res/2);
    center=ceil(size(terrain,1)/2);

    hitbox = terrain (center-sizey:center+sizey, center-sizex:center+sizex);
    hit_count = sum(hitbox(:)==1);

    % v = state(4);  
    % z = state(6);
    % zdot = state(7);
    
    % vzdialenost
    reward = -0.2 * dist_to_goal;

    % zasah do vinica
    reward = reward - 2 * hit_count;
    
    % smerovanie
    reward = reward - 0.1 * abs(heading_error);

    % rychlost
    % terrain_difficulty = mean(terrain(:));  
    % safe_speed = 1.0 * (1 - terrain_difficulty);
    % speed_penalty = (v - safe_speed)^2;
    % reward = reward - 0.5 * speed_penalty;

    % stabilita zmeny vysky
    % reward = reward - 0.2 * abs(z - 0.22);
    % reward = reward - 0.1 * abs(zdot);

    % casova penalizacia
    reward = reward - 0.01;

end


function net = buildActorNetwork(numObs)
    commonPath = [
        sequenceInputLayer(numObs, 'Name', 'observation')
        fullyConnectedLayer(256, 'Name', 'fc_pre_lstm')
        reluLayer('Name', 'relu_pre')
        lstmLayer(128, 'OutputMode', 'sequence', 'Name', 'lstm') 
        fullyConnectedLayer(128, 'Name', 'fc_common3')
        reluLayer('Name', 'relu_common3')
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
    
    net = layerGraph(commonPath);
    net = addLayers(net, meanPath);
    net = addLayers(net, stdPath);
    net = connectLayers(net, 'relu_common3', 'mean_fc');
    net = connectLayers(net, 'relu_common3', 'std_fc');
end

function net = buildCriticNetwork(numObs)
    statePath = [
        sequenceInputLayer(numObs, 'Name', 'state')
        fullyConnectedLayer(256, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
    ];
    
    actionPath = [
        sequenceInputLayer(4, 'Name', 'action')
        fullyConnectedLayer(256, 'Name', 'fc2')
        reluLayer('Name', 'relu_act')
    ];
    
    commonPath = [
        additionLayer(2, 'Name', 'add')
        lstmLayer(128, 'OutputMode', 'sequence', 'Name', 'critic_lstm')
        fullyConnectedLayer(128, 'Name', 'fc3')
        reluLayer('Name', 'relu3')
        fullyConnectedLayer(1, 'Name', 'output')
    ];
    
    net = layerGraph(statePath);
    net = addLayers(net, actionPath);
    net = addLayers(net, commonPath);
    net = connectLayers(net, 'relu1', 'add/in1');
    net = connectLayers(net, 'relu_act', 'add/in2');
end


function local_map = extractLocalTerrain(x,y, map, res, radius, grid_size)
    half = floor(grid_size/2);
    local_map = zeros(grid_size, grid_size);
    
    for i = 1:grid_size
        for j = 1:grid_size
            wx = x + (j - half - 1) * radius / half;
            wy = y + (i - half - 1) * radius / half;
            
            ix = round(wx/res) + 1;
            iy = round(wy/res) + 1;
            
            if ix >= 1 && ix <= size(map,2) && iy >= 1 && iy <= size(map,1)
                local_map(j,i) = map(ix, iy);
            else
                local_map(j,i) = 1.0;
            end
        end
    end
end

function drawTrackedRobot(x, y, theta, L, y_offset, res)

    % Robot corners (rectangle)
    corners_robot = [
        -L/2, -y_offset;
         L/2, -y_offset;
         L/2,  y_offset;
        -L/2,  y_offset
    ];

    % Rotation matrix
    R = [cos(theta) -sin(theta);
         sin(theta)  cos(theta)];

    % Transform corners to world coordinates
    corners_world = (R * corners_robot')';
    corners_world(:,1) = corners_world(:,1) + x;
    corners_world(:,2) = corners_world(:,2) + y;

    % World → map indices
    col = corners_world(:,2) / res;   % Y → column
    row = corners_world(:,1) / res;   % X → row

    % Draw filled rectangle
    patch( ...
        col, row, 'r', ...
        'EdgeColor', 'r', ...
        'LineWidth', 0.1);

    % --- Draw center-to-front line ---
    front_x = x + (L/2) * cos(theta);  % front point X
    front_y = y + (L/2) * sin(theta);  % front point Y

    line([y, front_y]/res, [x, front_x]/res, 'Color', 'k', 'LineWidth', 1);  % black line
end


