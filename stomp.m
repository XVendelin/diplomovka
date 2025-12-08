clc; clear; close all;

%% Load map
coords = [280 400; 280 520; 
         400 520; 400 400];
map = druhy('image.jpg', coords);

nmap = map; 
nmap(nmap ~= 1) = 0;

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


%% Compute CHOMP paths between all waypoints
fullPath = [];
for i = 1:(size(waypoints,1)-1)
    start = waypoints(i,:);
    goal = waypoints(i+1,:);
    segment = stomp_height(map, start, goal);
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
%% Display full path
figure;
imshow(map, []);
hold on;
plot(fullPath(:,2), fullPath(:,1), 'r-', 'LineWidth', 2);
plot(waypoints(:,2), waypoints(:,1), 'go', 'MarkerSize',10,'MarkerFaceColor','g');
title('CHOMP Optimized Path with Height-Aware Cost');

%% stomp
function path = stomp_height(map, start, goal)

    [rows, cols] = size(map);

    % ==== Check start & goal ====
    if map(start(1), start(2)) >= 1 || map(goal(1), goal(2)) >= 1
        path = [];
        warning('Start or goal is in an obstacle!');
        return;
    end

    % ==== STOMP parameters ====
    N = 300;          % number of path points
    K = 20;           % noisy rollouts
    step_size = 0.04;
    noise_std = 2.0;  % noise magnitude
    lambda = 10;      % height cost weight
    alpha = 0.2;      % smoothness weight

    % ===== Initial straight path =====
    path = [linspace(start(1), goal(1), N)' , linspace(start(2), goal(2), N)'];

    % ===== smoothing kernel (Gaussian) =====
    kernel = fspecial('gaussian', [21 1], 3);   % 1D smoothing
    kernel = kernel / sum(kernel);

    maxIter = 200;

    for iter = 1:maxIter
        noisy_paths = zeros(N,2,K);
        rollout_cost = zeros(K,1);

        for k = 1:K

            % === Generate noisy trajectories ===
            noise = noise_std * randn(N,2);

            % smooth noise by convolution
            noise(:,1) = conv(noise(:,1), kernel, 'same');
            noise(:,2) = conv(noise(:,2), kernel, 'same');

            noisy_paths(:,:,k) = path + noise;

            % keep endpoints fixed
            noisy_paths(1,:,k) = start;
            noisy_paths(end,:,k) = goal;

            % === Compute cost of noisy rollout ===
            c = 0;
            for j = 2:N-1
                r = round(noisy_paths(j,1,k));
                c2 = round(noisy_paths(j,2,k));

                if r < 1 || r > rows || c2 < 1 || c2 > cols
                    continue;
                end

                h = map(r,c2);

                c = c + lambda * h^2;    % terrain penalty
            end

            % smoothness cost (finite differences)
            d1 = diff(noisy_paths(:,1,k));
            d2 = diff(noisy_paths(:,2,k));
            c = c + alpha * ( sum(d1.^2) + sum(d2.^2) );

            rollout_cost(k) = c;
        end

        % === convert costs to probability weights ===
        costs = rollout_cost - min(rollout_cost);
        probs = exp(-costs);
        probs = probs / sum(probs);

        % === weighted update ===
        update = zeros(N,2);
        for k = 1:K
            update = update + probs(k)*(noisy_paths(:,:,k) - path);
        end

        % === apply update ===
        path = path + step_size * update;

        % enforce boundary and endpoints
        path(1,:) = start;
        path(end,:) = goal;

        path(:,1) = min(max(path(:,1),1), rows);
        path(:,2) = min(max(path(:,2),1), cols);
    end
end
