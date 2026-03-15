%% SAC Robot Navigation Training - A* Path Following

clear; clc; close all;

%% ========== LOAD MAP ==========
addpath("kinematika_MR");
coords1 = [280 400; 280 520; 400 520; 400 400];
map1 = druhy('image.jpg', coords1);
% coords2 = [475 1100; 475 1200; 575 1200; 575 1100];
% map2 = druhy('image.jpg', coords2);
res = 0.1;  % map resolution [m/cell]
% imagesc(map); colormap gray

%% ========== CONFIGURATION ==========
% -------- Map 1 routes --------
routes1(1).start = [10; 18] * res;
routes1(1).goal  = [100; 10] * res;

% routes1(2).start = [10; 38] * res;
% routes1(2).goal  = [100; 28] * res;


% -------- Map 2 routes --------
% routes2(1).start = [85; 10] * res;
% routes2(1).goal  = [69; 95] * res;
% 
% routes2(2).start = [56; 6] * res;
% routes2(2).goal  = [39; 92] * res;
% 
% routes2(3).start = [64; 10] * res;
% routes2(3).goal  = [39; 92] * res;

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
actInfo = rlNumericSpec([2 1], 'LowerLimit', -10, 'UpperLimit', 10);
actInfo.Name = 'Motor Torques';

% Create environment data structure to pass to functions
envData = struct();
envData.scenarios(1).map    = map1;
envData.scenarios(1).routes = routes1;

% envData.scenarios(2).map    = map2;
% envData.scenarios(2).routes = routes2;

envData.res        = res;
envData.obs_radius = obs_radius;
envData.obs_size   = obs_size;
envData.dt         = dt;



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

%% ========== COPY WEIGHTS FROM TRAINED 4-OUTPUT AGENT ==========
fprintf('Loading pretrained agent and transferring weights...\n');
oldData = load('savedAgents_uznefunguje\Agent939.mat');
oldAgent = oldData.saved_agent;

% --- Actor ---
oldActorParams = getLearnableParameters(getActor(oldAgent));
newActorParams  = getLearnableParameters(actor);
numSharedActor  = numel(newActorParams) - 4; % last 4 = mean_fc2 W+b, std_fc2 W+b
for i = 1:numSharedActor
    newActorParams{i} = oldActorParams{i};
end
actor = setLearnableParameters(actor, newActorParams);

% --- Critic 1 ---
oldCritics  = getCritic(oldAgent);
oldC1Params = getLearnableParameters(oldCritics(1));
newC1Params = getLearnableParameters(critic1);
for i = 1:numel(newC1Params)
    if isequal(size(newC1Params{i}), size(oldC1Params{i}))
        newC1Params{i} = oldC1Params{i};
    else
        fprintf('Critic1 param %d: size mismatch, keeping random init\n', i);
    end
end
critic1 = setLearnableParameters(critic1, newC1Params);

% --- Critic 2 ---
oldC2Params = getLearnableParameters(oldCritics(2));
newC2Params = getLearnableParameters(critic2);
for i = 1:numel(newC2Params)
    if isequal(size(newC2Params{i}), size(oldC2Params{i}))
        newC2Params{i} = oldC2Params{i};
    else
        fprintf('Critic2 param %d: size mismatch, keeping random init\n', i);
    end
end
critic2 = setLearnableParameters(critic2, newC2Params);
fprintf('Weight transfer complete.\n');

% SAC agent options
agentOpts = rlSACAgentOptions(...
    'SampleTime', dt, ...
    'DiscountFactor', 0.99, ...
    'ExperienceBufferLength', 1e6, ...
    'MiniBatchSize', 1024*4, ...
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
    'MaxStepsPerEpisode', 200, ...
    'ScoreAveragingWindowLength', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 5000, ...
    'UseParallel', false, ...
    'SaveAgentCriteria', "EpisodeReward", ...
    "SaveAgentValue", 3000, ...
    "SaveAgentDirectory", 'savedAgents1pas');

%% ========== TRAIN AGENT ==========
fprintf('\n=== Starting SAC Training ===\n');
agent = setLearnableParameters(agent, dlupdate(@gpuArray, getLearnableParameters(agent)));
trainingStats = train(agent, env, trainOpts);

%% ========== GET BEST AGENT =======
foldername = 'savedAgents_uznefunguje';
offset = 900;

files = dir(fullfile(foldername, 'Agent*.mat'));

if ~isempty(files)
    [~, idx] = sort([files.datenum], 'descend');
    newestFile = fullfile(files(idx(1)).folder, files(idx(1)).name);
    
    % Load the data
    data = load(newestFile);
    rewards = data.savedAgentResult.EpisodeReward;
    [maxVal, maxIdx] = max(rewards(offset+1:end));
    maxIdx=maxIdx+offset;
else
    error('No agent files found in the directory.');
end

fprintf('Best agent: %i | Reward: %.1f\n', maxIdx, maxVal);
fileName = sprintf('Agent%d.mat', maxIdx);
filePath = fullfile(foldername, fileName);
data=load(filePath);
agent = data.saved_agent;
save('bestAgent.mat', 'agent');

%% ========== TEST TRAINED AGENT ==========
close all;
addpath("kinematika_MR");
fprintf('\n=== Testing Trained Agent ===\n');
greedyPolicy = getGreedyPolicy(agent);
reset(greedyPolicy);
% map = druhy("image.jpg", [700 500; 700 600; 800 600; 800 500]);
map = imread("extraction.png");
if size(map,3) == 3
    map = rgb2gray(map);
end

map=im2double(map);

test_start = [87; 10] * res;
test_goal = [85; 90] * res;

test_goals = [75 76 64 60; 
            90 22 26 95];

% test_start = [85; 10] * res;
% test_goal = [69; 95] * res;

state = [test_start; pi/2; 0; 0; 0; 0];
% state(1:2)  = [test_start];

trajectory = state(1:2)';

max_test_steps = 2000;

figure('Position', [100, 100, 1500, 600]);
a=0;

for step = 1:max_test_steps
    x = state(1);
    y = state(2);
    theta = state(3);
    v = state(4);
    omega = state(5);


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
    M = getAction(greedyPolicy, obs);
    M = M{1};  % Extract from cell array
    M = max(min(M(:), 10), -10);
    M = [M;M];

    state = trackedRobotDynamics(state, M, dt);
    trajectory = [trajectory; state(1:2)'];

    % --- VISUALIZATION ---
    if mod(step, 4) == 0
        subplot(1,4,1);
        imagesc(map); colormap gray; hold on;
        plot(trajectory(:,2)/res, trajectory(:,1)/res, 'g-', 'LineWidth', 2);
        % plot(test_start(2)/res, test_start(1)/res, 'go', ...
        %     'MarkerSize', 8, 'MarkerFaceColor', 'g');
        plot(test_goal(2)/res, test_goal(1)/res, 'r*', ...
            'MarkerSize', 18, 'LineWidth', 2);
        drawTrackedRobot(x, y, theta, 0.25, 0.25, res);

        title(sprintf('Step %d | Goal: %.2fm', step, dist_to_goal));
        axis equal; axis tight; hold off;

        subplot(1,4,2);
        bar(M); ylim([-12, 12]); grid on;
        title('Motor Torques'); ylabel('Nm');
        xticklabels({'FL','FR','RL','RR'});

        subplot(1,4,3);
        bar([v, omega]); ylim([-5, 5]); grid on;
        title('Speeds');
        xticklabels({'Linear','Angular'});
        ylabel('m/s   |   rad/s');

        subplot(1,4,4);
        imagesc(local_terrain, [min(map(:)) max(map(:))]);
        colormap(gca, 'gray'); colorbar;
        title('Local View');
        axis equal; axis tight;

        drawnow;
    end

    if dist_to_goal < 0.3
        if a >= size(test_goals, 2)
            fprintf('SUCCESS! Goal reached in %d steps (%.2fm)\n', step, dist_to_goal);
            break;
        end
        a=a+1;
        test_goal = test_goals(:,a) * res;
        test_start = state(1:2);
        reset(greedyPolicy);
    end

    % if step > 200 && norm(trajectory(end,:) - trajectory(end-50,:)) < 0.5
    %     fprintf('Stuck at step %d (%.2fm from goal)\n', step, dist_to_goal);
    %     break;
    % end
end


%% ========== ENVIRONMENT FUNCTIONS ==========

function [nextObs, reward, isDone, loggedSignals] = stepFcn(action, loggedSignals, envData)

    
    state      = loggedSignals.state;
    trajectory = loggedSignals.trajectory;
    step_count = loggedSignals.step_count + 1;

    M = [action;action];
    state = trackedRobotDynamics(state, M, envData.dt);

    x     = state(1);
    y     = state(2);
    theta = state(3);
    trajectory = [trajectory; x, y];

    local_terrain = extractLocalTerrain(x, y, loggedSignals.map, envData.res, ...
                                        envData.obs_radius, envData.obs_size);
    terrain_vec = reshape(local_terrain, [], 1);

    goal_dx = loggedSignals.goal_pos(1) - x;
    goal_dy = loggedSignals.goal_pos(2) - y;
    dist_to_goal = hypot(goal_dx, goal_dy);
    heading_error = wrapToPi(atan2(goal_dy, goal_dx) - theta);

    nextObs = [
        sin(theta);
        cos(theta);
        sin(heading_error);
        cos(heading_error);
        dist_to_goal / norm(loggedSignals.goal_pos - loggedSignals.start_pos);
        state(4)/5; 
        state(5)/5;
        terrain_vec
    ];

    reward = calculateReward(state, loggedSignals.goal_pos, local_terrain, ...
                             dist_to_goal, heading_error);

    isDone = false;

    if dist_to_goal < 0.3
        reward = reward + 5000;  % Big bonus for reaching goal
        isDone = true;
    elseif step_count >= 5000
        isDone = true;
    elseif any(isnan(state)) || any(isinf(state))
        reward = reward - 100;
        isDone = true;
    end

    % --- Check Out of Bounds ---
    [mapRows, mapCols] = size(loggedSignals.map);
    maxX = mapCols * envData.res;
    maxY = mapRows * envData.res;

    if x < 0 || x > maxX || y < 0 || y > maxY
        isDone = true;
    end

    % --- Check if stuck ---
    if size(trajectory,1) > 100
        if norm(trajectory(end,:) - trajectory(end-50,:)) < 0.2
            isDone = true;
        end
    end

    % --- Update loggedSignals for next step ---
    loggedSignals.state      = state;
    loggedSignals.trajectory = trajectory;
    loggedSignals.step_count = step_count;
end

function [initObs, loggedSignals] = resetFcn(envData)

    % ----- Pick map -----
    scenario_id = randi(numel(envData.scenarios));
    scenario = envData.scenarios(scenario_id);

    % ----- Pick route -----
    route_id = randi(numel(scenario.routes));
    route = scenario.routes(route_id);

    % ----- Optional flip -----
    if rand < 2
        start_pos = route.start;
        goal_pos  = route.goal;
        flipped = false;
    else
        start_pos = route.goal;
        goal_pos  = route.start;
        flipped = true;
    end

    % ----- Store episode constants -----
    loggedSignals.map        = scenario.map;
    loggedSignals.start_pos  = start_pos;
    loggedSignals.goal_pos   = goal_pos;
    loggedSignals.scenario   = scenario_id;
    loggedSignals.route_id   = route_id;
    loggedSignals.flipped    = flipped;

    % ----- Initialize robot -----
    % init_theta = rand * 2*pi - pi;
    init_theta=0;
    % init_v = rand * 0.5;
    init_v=0;
    state = [start_pos; init_theta; init_v; 0; 0; 0];

    loggedSignals.state      = state;
    loggedSignals.trajectory = start_pos';
    loggedSignals.step_count = 0;

    % ----- Initial observation -----
    x = state(1); y = state(2); theta = state(3);

    local_terrain = extractLocalTerrain( ...
        x, y, loggedSignals.map, ...
        envData.res, envData.obs_radius, envData.obs_size);

    terrain_vec = reshape(local_terrain, [], 1);

    goal_dx = goal_pos(1) - x;
    goal_dy = goal_pos(2) - y;
    dist_to_goal = hypot(goal_dx, goal_dy);
    heading_error = wrapToPi(atan2(goal_dy, goal_dx) - theta);

    initObs = [
        sin(theta); cos(theta);
        sin(heading_error); cos(heading_error);
        dist_to_goal / norm(goal_pos - start_pos);
        0; 0;
        terrain_vec
    ];
    % fprintf("Episode start | Scenario %d | Route %d | Flipped %d\n", ...
    %     loggedSignals.scenario, ...
    %     loggedSignals.route_id, ...
    %     loggedSignals.flipped);

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

    v = state(4);
    theta = state(3);
    % z = state(6);
    % zdot = state(7);
    
    % vzdialenost
    reward = -0.2 * dist_to_goal;

    % zasah do vinica
    reward = reward - 20 * hit_count;
    
    % smerovanie
    reward = reward - 5 * abs(heading_error);

    % % rychlost
    % terrain_difficulty = mean(hitbox(:),"all");  
    % safe_speed = 1 * (1 - terrain_difficulty);
    % speed_penalty = 2 * (v - safe_speed)^4;
    % reward = reward - speed_penalty - theta^2*0.1;
    % 
    % % spomaliť blízko k cieľu
    % if dist_to_goal < 2.5
    %     max_allowed_speed = 0.5;
    %     if v > max_allowed_speed
    %         reward = reward - 0.05* (v - max_allowed_speed)^2;
    %     end
    % end
    % 
    % % stabilita zmeny vysky
    % % reward = reward - 0.2 * abs(z - 0.22);
    % % reward = reward - 0.1 * abs(zdot);
    % 
    % % casova penalizacia
    % % reward = reward - 0.01;

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
        fullyConnectedLayer(2, 'Name', 'mean_fc')
        tanhLayer('Name', 'mean_tanh')
        scalingLayer('Scale', 10, 'Name', 'mean_scale')
    ];
    
    stdPath = [
        fullyConnectedLayer(2, 'Name', 'std_fc')
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
        sequenceInputLayer(2, 'Name', 'action')
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

    corners_robot = [
        -L, -y_offset;
         L, -y_offset;
         L,  y_offset;
        -L,  y_offset
    ];

    R = [cos(theta) -sin(theta);
         sin(theta)  cos(theta)];

    corners_world = (R * corners_robot')';
    corners_world(:,1) = corners_world(:,1) + x;
    corners_world(:,2) = corners_world(:,2) + y;

    col = corners_world(:,2) / res;
    row = corners_world(:,1) / res;

    patch( ...
        col, row, 'r', ...
        'EdgeColor', 'r', ...
        'LineWidth', 0.1);

    front_x = x + (L) * cos(theta);
    front_y = y + (L) * sin(theta);

    line([y, front_y]/res, [x, front_x]/res, 'Color', 'k', 'LineWidth', 1);
end


