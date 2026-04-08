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
map = imread("image - Copy.jpg");
if size(map,3) == 3
    map = rgb2gray(map);
end

map=im2double(map);

res=0.1;

test_start = [15; 13] * res;
test_goal = [102; 12] * res;

test_goals = [97 58 42 45 102 100 77 34; 
             28 32 45 57 60 78 78 77];

% test_start = [85; 10] * res;
% test_goal = [69; 95] * res;

state = [test_start; 0; 0; 0; 0; 0];
% state(1:2)  = [test_start];

trajectory = state(1:2)';

max_test_steps = 2000;

% figure('Position', [100, 100, 1500, 600]);
a=0;
prev_M = 0;

for step = 1:max_test_steps
    x = state(1); y = state(2); theta = state(3); v = state(4); omega = state(5);

    local_terrain = extractLocalTerrain(x, y, map, res, obs_radius, obs_size);
    terrain_vec = reshape(local_terrain, [], 1);

    goal_dx = test_goal(1) - x;
    goal_dy = test_goal(2) - y;
    goal_heading_error = wrapToPi(atan2(goal_dy, goal_dx) - theta);
    dist_to_goal = hypot(goal_dx, goal_dy);

    obs = [sin(theta); cos(theta); sin(goal_heading_error); cos(goal_heading_error); ...
           dist_to_goal / norm(test_goal - test_start); state(4)/5; state(5)/5; terrain_vec];

    M_cell = getAction(greedyPolicy, obs);
    M = M_cell{1}; 
    M = max(min(M(:), 10), -10);

    state = trackedRobotDynamics(state, M, dt);
    trajectory = [trajectory; state(1:2)'];

    % --- METRIC CALCULATION (Inside Loop) ---
    res = 0.1;
    sizex= ceil(0.5/res/2);
    sizey= ceil(0.5/res/2);
    center=ceil(size(local_terrain,1)/2);

    hitbox = local_terrain (center-sizey:center+sizey, center-sizex:center+sizex);
    hit_count = sum(hitbox(:)==1);

    metrics_safety(step) = sum(hitbox(:))/numel(hitbox); 
    metrics_hitcount(step) = hit_count;
    metrics_distance(step) = dist_to_goal;
    metrics_speed(step) = v;
    
    if step == 1
        metrics_smooth(step) = 0;
    else
        metrics_smooth(step) = abs(wrapToPi(theta - prev_theta));
    end
    prev_theta = theta;

    metrics_smooth(step) = sum(abs(M(:) - prev_M(:)));
    metrics_steps(step) = step;
    % ----------------------------------------
    
    prev_M = M; % Update prev_M for the next step

    % --- VISUALIZATION ---
    if mod(step, 4) == 5
        subplot(1,4,1);
        imagesc(map); colormap gray; hold on;
        plot(trajectory(:,2)/res, trajectory(:,1)/res, 'g-', 'LineWidth', 1);
        % plot(test_start(2)/res, test_start(1)/res, 'go', ...
        %     'MarkerSize', 8, 'MarkerFaceColor', 'g');
        plot(test_goal(2)/res, test_goal(1)/res, 'r*', ...
            'MarkerSize', 15, 'LineWidth', 1);
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
end


subplot(1,4,1);
hold on;
plot(waypoints(:,2), waypoints(:,1), 'r*', 'MarkerSize', 15, 'LineWidth', 1);


ax = subplot(1,4,1); 
drawnow; pause(0.01);
frame = getframe(ax);
im = frame2im(frame);
imwrite(im, 'test2.png');

%% ========== FINAL PERFORMANCE SUMMARY ==========
figure(2);
plot(metrics_steps, smoothdata(metrics_smooth, 'movmean', 10), 'r', 'LineWidth', 1);
grid on;
title('Path Smoothness (Heading Change)');
xlabel('Path Step'); 
ylabel('Change in Direction (deg)');

figure(3);
plot(metrics_steps, metrics_safety*100);
average_safety = sum(metrics_safety)/length(metrics_safety);
hold on;
plot([0, 474], [average_safety, average_safety]*100, 'r--');
hold off;
title('Safety Metric Along Path');
xlabel('Path Step'); ylabel('Obstacle Density in Local Window (%)');
legend('Current Safety', sprintf('Average: %.1f%%', average_safety*100));
grid on;

figure(4);
plot(metrics_steps, metrics_distance);
figure(5);
plot(metrics_steps, metrics_safety*10);
hold on;
plot(metrics_steps, metrics_speed);


%% ========== HELPER FUNCTIONS ==========


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


