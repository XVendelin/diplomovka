%% ========== TEST TRAINED AGENT (WITH VIDEO + GIF + MULTI-GOAL) ==========
close all;
addpath("kinematika_MR");
fprintf('\n=== Testing Trained Agent with Video Recording ===\n');

% --- Load Agent and Map ---
% Note: Ensure 'agent', 'res', 'obs_radius', 'obs_size', 'dt' are in workspace or loaded
greedyPolicy = getGreedyPolicy(agent);
reset(greedyPolicy);

map = imread("extraction3.jpg");
if size(map,3) == 3
    map = rgb2gray(map);
end
map = im2double(map);

% --- Goal Sequence Setup ---
test_goals = [102 97 58 42 45 102 100 77 34; 
             12 28 32 45 57 60 78 78 77];
goal_idx = 1;

% Initialize Start State
test_start = [15; 13] * res;
current_goal = test_goals(:, goal_idx) * res;

state = [test_start; 0; 0; 0; 0; 0];
trajectory = state(1:2)';
max_test_steps = 5000; % Increased for multiple goals

%% ---------- VIDEO + GIF SETUP ----------
videoFile = 'agent_multi_goal.mp4';
gifFile   = 'agent_multi_goal.gif';

vwriter = VideoWriter(videoFile, 'MPEG-4');
vwriter.FrameRate = 10;
open(vwriter);

gifDelay = 1 / vwriter.FrameRate;
isFirstGifFrame = true;
% ---------------------------------------

figure('Position', [100, 100, 1500, 600]);

for step = 1:max_test_steps
    x     = state(1);
    y     = state(2);
    theta = state(3);
    v     = state(4);
    omega = state(5);

    % --- Observation Prep ---
    local_terrain = extractLocalTerrain(x, y, map, res, obs_radius, obs_size);
    terrain_vec   = reshape(local_terrain, [], 1);

    goal_dx = current_goal(1) - x;
    goal_dy = current_goal(2) - y;
    goal_heading_error = wrapToPi(atan2(goal_dy, goal_dx) - theta);
    dist_to_goal = hypot(goal_dx, goal_dy);

    obs = [
        sin(theta);
        cos(theta);
        sin(goal_heading_error);
        cos(goal_heading_error);
        dist_to_goal / norm(current_goal - test_start);
        state(4) / 5;
        state(5) / 5;
        terrain_vec
    ];

    % --- Get Action ---
    M = getAction(greedyPolicy, obs);
    M = M{1};
    M = max(min(M(:), 10), -10);

    % --- Update Physics ---
    state = trackedRobotDynamics(state, M, dt);
    trajectory = [trajectory; state(1:2)'];

    %% ---------- VISUALIZATION (Every 4 steps) ----------
    if mod(step, 2) == 0
        clf;
        
        % 1. Global Map
        subplot(1,4,1);
        imagesc(map); colormap gray; hold on;
        plot(trajectory(:,2)/res, trajectory(:,1)/res, 'g-', 'LineWidth', 2);
        plot(current_goal(2)/res, current_goal(1)/res, 'r*', 'MarkerSize', 18, 'LineWidth', 2);
        drawTrackedRobot(x, y, theta, 0.25, 0.25, res);
        title(sprintf('Step %d | Goal %d | Dist: %.2fm', step, goal_idx, dist_to_goal));
        axis equal tight; hold off;

        % 2. Motor Torques
        subplot(1,4,2);
        bar(M); ylim([-12 12]); grid on;
        title('Motor Torques (Nm)');
        xticklabels({'FL','FR','RL','RR'});

        % 3. Velocities
        subplot(1,4,3);
        bar([v omega]); ylim([-5 5]); grid on;
        title('Speeds');
        xticklabels({'Linear','Angular'});
        ylabel('m/s | rad/s');

        % 4. Local Sensor View
        subplot(1,4,4);
        imagesc(local_terrain, [min(map(:)) max(map(:))]);
        colormap(gca, 'gray'); colorbar;
        title('Local Terrain View');
        axis equal tight;

        drawnow;

        %% ---------- RECORD FRAME ----------
        frame = getframe(gcf);
        writeVideo(vwriter, frame);

        % GIF Logic
        [im, cm] = rgb2ind(frame2im(frame), 256);
        if isFirstGifFrame
            imwrite(im, cm, gifFile, 'gif', 'LoopCount', inf, 'DelayTime', gifDelay);
            isFirstGifFrame = false;
        else
            imwrite(im, cm, gifFile, 'gif', 'WriteMode', 'append', 'DelayTime', gifDelay);
        end
    end

    %% ---------- LOGIC: GOAL REACHED / STUCK ----------
    if dist_to_goal < 0.3
        if goal_idx >= size(test_goals, 2)
            fprintf('FINAL SUCCESS! All goals reached in %d steps\n', step);
            break;
        else
            fprintf('Goal %d reached! Moving to next...\n', goal_idx);
            goal_idx = goal_idx + 1;
            current_goal = test_goals(:, goal_idx) * res;
            test_start = state(1:2);
            reset(greedyPolicy);
        end
    end

    % Optional: Stuck detection
    if step > 500 && norm(trajectory(end,:) - trajectory(end-100,:)) < 0.1
        fprintf('Robot appears stuck at step %d. Terminating.\n', step);
        break;
    end
end

%% ---------- CLEANUP ----------
close(vwriter);
fprintf('\nFiles saved:\n- %s\n- %s\n', videoFile, gifFile);
% close all; % Uncomment to close figure window automatically

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

    % Robot corners (rectangle)
    corners_robot = [
        -L, -y_offset;
         L, -y_offset;
         L,  y_offset;
        -L,  y_offset
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
    front_x = x + (L) * cos(theta);  % front point X
    front_y = y + (L) * sin(theta);  % front point Y

    line([y, front_y]/res, [x, front_x]/res, 'Color', 'k', 'LineWidth', 1);  % black line
end


