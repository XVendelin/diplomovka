%% cnn_height_optimal_planner.m
% Trains a tiny CNN that sees a 5x5 local patch and learns to choose an
% 8-way action that follows a path minimizing height changes (not Euclidean length).

clc; clear; close all;

%% ---------- LOAD MAP & WAYPOINTS (use your existing code) ----------
coords = [280 400; 280 520; 
         400 520; 400 400];
map = druhy('image.jpg', coords);

% map values: 0..1, with 1 meaning unaccessible
obstacleMask = (map >= 1.0);

%% Define waypoints
waypoints = [10 20;
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
            size(map,1)-20, size(map,2)-15]; % goal


%% ---------- PARAMETERS ----------
patchSize = 5;                 % 5x5 local vision
pad = floor(patchSize/2);
nDirs = 8;
maxSamplesPerPair = 3000;      % limit per waypoint pair
alphaMove = 0.01;              % small weight for Euclidean move cost
betaHeight = 1.0;              % weight for absolute height change (we want to minimize this)
rng(0);

%% ---------- PAD MAP FOR PATCH EXTRACTION ----------
paddedMap = padarray(map, [pad pad], 1);    % pad with obstacles (1)
paddedFree = ~ (paddedMap >= 1.0);          % free cells in padded map

%% ---------- BUILD DATASET from optimal (height-change) paths ----------
X = []; Y = [];
fprintf('Building dataset from waypoint pairs (optimal wrt height-change)...\n');
for k = 1:size(waypoints,1)-1
    s = waypoints(k,:);
    g = waypoints(k+1,:);
    % ensure start/goal inside and not obstacle
    if map(s(1),s(2))>=1 || map(g(1),g(2))>=1
        warning('Waypoint %d or %d on obstacle - skipping pair.', k, k+1);
        continue;
    end
    path = dijkstra_min_height(map, s, g, alphaMove, betaHeight);
    if isempty(path)
        warning('No path found between waypoint %d and %d', k, k+1);
        continue;
    end
    L = size(path,1);
    % extract samples along path (cur -> next)
    maxSamps = min(L-1, maxSamplesPerPair);
    idxs = round(linspace(1, L-1, maxSamps)); % subsample if very long
    for ii = idxs
        cur = path(ii,:);
        nxt = path(ii+1,:);
        r = cur(1) + pad;
        c = cur(2) + pad;
        patch = paddedMap(r-pad:r+pad, c-pad:c+pad);
        X(:,: ,1, end+1) = single(patch); %#ok<SAGROW>
        Y(end+1) = directionToLabel(nxt - cur); %#ok<SAGROW>
    end
end

if size(X,4) > numel(Y)
    X = X(:,:,:,1:numel(Y));    % trim extra X
end

if isempty(Y)
    error('No training samples. Check map/waypoints or padding.');
end

Y = categorical(Y,1:8,{'N','NE','E','SE','S','SW','W','NW'});

fprintf('Built dataset: %d samples\n', numel(Y));

%% ---------- DEFINE TINY CNN (input 5x5x1) ----------
layers = [
    imageInputLayer([patchSize patchSize 1],'Normalization','none','Name','input')
    convolution2dLayer(3,16,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    convolution2dLayer(3,32,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(64,'Name','fc1')
    reluLayer('Name','relu3')
    fullyConnectedLayer(8,'Name','fc_out')
    softmaxLayer('Name','soft')
    classificationLayer('Name','class')
];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-2, ...
    'MaxEpochs',500, ...
    'MiniBatchSize',128, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

%% --- DEBUG SHAPE CHECK ---
fprintf("\nChecking dataset shapes...\n");

if ndims(X) ~= 4
    error("X must be 4-D: H×W×C×N. Your X has %d dimensions.", ndims(X));
end

nX = size(X,4);
nY = numel(Y);

fprintf("X samples: %d\n", nX);
fprintf("Y labels : %d\n", nY);

if nX ~= nY
    error("Dataset mismatch: X has %d samples but Y has %d labels.", nX, nY);
end

fprintf("Dataset OK. Training starting...\n");


fprintf('Training network (this may take a moment)...\n');
net = trainNetwork(X, Y, layers, options);

%% ---------- INFERENCE: roll out CNN-guided path between every waypoint pair ----------
fullPath = [];
for k = 1:size(waypoints,1)-1
    start = waypoints(k,:);
    goal  = waypoints(k+1,:);
    fprintf('Rolling from %d->%d\n', k, k+1);
    path_net = rollout_with_cnn(start, goal, map, paddedMap, net, patchSize, pad, alphaMove, betaHeight);
    if isempty(path_net)
        warning('CNN failed to produce a path; falling back to optimal Dijkstra path for this pair.');
        path_net = dijkstra_min_height(map, start, goal, alphaMove, betaHeight);
    end
    % append (but avoid repeating last waypoint)
    if isempty(fullPath)
        fullPath = path_net;
    else
        % avoid duplicating last point of current fullPath if it equals first of new
        if isequal(fullPath(end,:), path_net(1,:))
            fullPath = [fullPath; path_net(2:end,:)]; %#ok<AGROW>
        else
            fullPath = [fullPath; path_net]; %#ok<AGROW>
        end
    end
end

%% ---------- DISPLAY final path ----------
figure;
imshow(map, []); colormap('turbo'); colorbar;
hold on;
plot(fullPath(:,2), fullPath(:,1), 'w-', 'LineWidth', 2);
plot(waypoints(:,2), waypoints(:,1), 'go', 'MarkerSize',10,'MarkerFaceColor','g');
title('CNN Path Optimized for Minimal Height Changes');

%% ------------------ SUPPORT FUNCTIONS ------------------

function label = directionToLabel(d)
    % d = [dr dc] where dr,dc in {-1,0,1} and not [0,0]
    % returns 1..8 mapping N,NE,E,SE,S,SW,W,NW
    mapping = {
        [-1 0], 1;  % N
        [-1 1], 2;  % NE
        [0 1],  3;  % E
        [1 1],  4;  % SE
        [1 0],  5;  % S
        [1 -1], 6;  % SW
        [0 -1], 7;  % W
        [-1 -1],8}; % NW
    label = NaN;
    for ii=1:8
        if isequal(d, mapping{ii,1})
            label = mapping{ii,2};
            return;
        end
    end
    if all(d==0)
        label = 3; % arbitrary (shouldn't happen)
    else
        error('Invalid direction [%d %d] for labeling.', d(1), d(2));
    end
end

function path = dijkstra_min_height(map, start, goal, alphaMove, betaHeight)
    % Dijkstra on grid using cost: alpha*moveDist + beta*abs(height diff)
    % map: height values 0..1, with 1 = obstacle
    % start, goal: [row col]
    [nR, nC] = size(map);
    startIdx = sub2ind([nR,nC], start(1), start(2));
    goalIdx  = sub2ind([nR,nC], goal(1), goal(2));
    impass = (map >= 1.0);
    if impass(start(1),start(2)) || impass(goal(1),goal(2))
        path = [];
        return;
    end

    N = nR*nC;
    dist = inf(N,1);
    prev = zeros(N,1);
    visited = false(N,1);
    dist(startIdx) = 0;

    % neighbor offsets
    offs = [-1 0; -1 1; 0 1; 1 1; 1 0; 1 -1; 0 -1; -1 -1];
    moveCost = [1 sqrt(2) 1 sqrt(2) 1 sqrt(2) 1 sqrt(2)];

    while true
        % pick unvisited node with smallest dist
        [dmin, u] = min(dist + (visited.*1e12)); %#ok<ASGLU>
        if isinf(dmin)
            break;
        end
        if u == goalIdx
            break;
        end
        visited(u) = true;
        [ur, uc] = ind2sub([nR nC], u);
        curH = map(ur,uc);
        for k = 1:8
            vr = ur + offs(k,1);
            vc = uc + offs(k,2);
            if vr < 1 || vr > nR || vc < 1 || vc > nC
                continue;
            end
            if impass(vr,vc)
                continue;
            end
            v = sub2ind([nR nC], vr, vc);
            if visited(v), continue; end
            hcost = betaHeight * abs(map(vr,vc) - curH);
            mcost = alphaMove * moveCost(k);
            alt = dist(u) + hcost + mcost;
            if alt < dist(v)
                dist(v) = alt;
                prev(v) = u;
            end
        end
    end

    if isinf(dist(goalIdx))
        path = [];
        return;
    end

    % reconstruct path
    nodes = goalIdx;
    cur = goalIdx;
    while cur ~= startIdx
        cur = prev(cur);
        nodes(end+1) = cur; %#ok<AGROW>
        if cur == 0
            path = [];
            return;
        end
    end
    nodes = flip(nodes);
    [rs, cs] = ind2sub([nR nC], nodes);
    path = [rs(:), cs(:)];
end

function path = rollout_with_cnn(start, goal, map, paddedMap, net, patchSize, pad, alphaMove, betaHeight)
    % Greedy rollout using CNN predictions. If stuck or loops detected,
    % return [] (caller will fallback to Dijkstra)
    maxSteps = numel(map)*2;
    pos = start;
    path = pos;
    visited = false(size(map));
    visited(pos(1), pos(2)) = true;
    for steps = 1:maxSteps
        % if reached goal
        if isequal(pos, goal)
            return;
        end
        % extract patch centered at pos (use paddedMap)
        r = pos(1) + pad;
        c = pos(2) + pad;
        patch = paddedMap(r-pad:r+pad, c-pad:c+pad);
        patch = single(patch);
        % predict
        pred = classify(net, patch);
        % convert label to move order preference (probability-based)
        probs = predict(net, patch);
        [~, order] = sort(probs,'descend');
        moved = false;
        for oi = 1:length(order)
            lab = order(oi);
            move = labelToDirection(lab);
            nxt = pos + move;
            if nxt(1) < 1 || nxt(1) > size(map,1) || nxt(2) < 1 || nxt(2) > size(map,2)
                continue;
            end
            if map(nxt(1),nxt(2)) >= 1.0
                continue;
            end
            if visited(nxt(1),nxt(2))
                continue;
            end
            % accept this move
            pos = nxt;
            path = [path; pos]; %#ok<AGROW>
            visited(pos(1), pos(2)) = true;
            moved = true;
            break;
        end
        if ~moved
            % couldn't find valid predicted move -> give up (caller fallbacks)
            path = [];
            return;
        end
    end
    % exceeded max steps
    path = [];
end

function d = labelToDirection(label)
    % 1..8 -> [dr dc]
    switch label
        case 1, d = [-1 0];  % N
        case 2, d = [-1 1];  % NE
        case 3, d = [0 1];   % E
        case 4, d = [1 1];   % SE
        case 5, d = [1 0];   % S
        case 6, d = [1 -1];  % SW
        case 7, d = [0 -1];  % W
        case 8, d = [-1 -1]; % NW
        otherwise, d = [0 0];
    end
end
